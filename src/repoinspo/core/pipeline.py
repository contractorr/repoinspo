"""Pipeline orchestration for scouting portable ideas."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from litellm import acompletion, aembedding

from repoinspo.config import Settings, get_settings
from repoinspo.core.analysis import (
    analyze_repo,
    compare_repos,
    extract_features,
    generate_search_strategies,
    prioritize_ideas,
)
from repoinspo.core.embeddings import EmbeddingIndex
from repoinspo.core.github import GitHubClient
from repoinspo.core.ingestion import DEFAULT_INGESTER, IngestCallable, ingest_repo
from repoinspo.models import (
    IngestedRepo,
    RepoMetadata,
    ScoutResult,
    SearchFilters,
    SearchStrategy,
    TokenBudget,
)

logger = logging.getLogger(__name__)


async def find_similar_repos(
    repo_url: str,
    n: int,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
    github_client: GitHubClient | None = None,
    embedding_func: Any = aembedding,
    ingester: IngestCallable | None = None,
    strategies: list[SearchStrategy] | None = None,
) -> list[RepoMetadata]:
    """Find similar repositories using keyword search and embedding reranking.

    If strategies are provided, runs each strategy's query and merges results.
    Falls back to static _build_search_query when no strategies are given.
    """

    config = settings or get_settings()
    own_client = github_client is None
    client = github_client or GitHubClient(
        token=config.github_token.get_secret_value() if config.github_token else None,
        cache_path=config.cache_dir / "github_cache.sqlite3",
    )
    try:
        seed = await _prepare_ingested_repo(
            repo_url,
            config,
            client,
            ingester=ingester,
        )

        if strategies:
            per_query_limit = min(max(n * 3, 10), 30)
            search_tasks = [
                client.search_repos(s.query, per_page=per_query_limit)
                for s in strategies
            ]
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            seen: set[str] = set()
            candidates: list[RepoMetadata] = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Strategy search failed: %s", result)
                    continue
                for repo in result:
                    if repo.full_name != seed.metadata.full_name and repo.full_name not in seen:
                        seen.add(repo.full_name)
                        candidates.append(repo)
        else:
            query = GitHubClient._build_search_query(seed.metadata, filters or SearchFilters())
            search_limit = min(max(n * 5, 10), 100)
            candidates = await client.search_repos(query, per_page=search_limit)
            candidates = [
                repo for repo in candidates if repo.full_name != seed.metadata.full_name
            ]

        if not candidates:
            return []

        try:
            index = EmbeddingIndex(config.embedding_model, embedding_func=embedding_func)
            reranked = await index.rerank(
                query=_repo_rerank_text(seed.metadata),
                items=candidates,
                texts=[_repo_rerank_text(repo) for repo in candidates],
                top_k=n,
            )
            return reranked[:n]
        except Exception as exc:
            logger.warning(
                "Embedding rerank unavailable; falling back to keyword search",
                exc_info=exc,
            )
            return candidates[:n]
    finally:
        if own_client:
            await client.close()


async def scout_ideas(
    repo_url: str,
    n_similar: int,
    target_context: str | None = None,
    budget: int | TokenBudget | None = None,
    filters: SearchFilters | None = None,
    settings: Settings | None = None,
    github_client: GitHubClient | None = None,
    completion_func: Any = acompletion,
    embedding_func: Any = aembedding,
    ingester: IngestCallable | None = None,
) -> ScoutResult:
    """Run the full scouting pipeline from a seed repository."""

    config = settings or get_settings()
    token_budget = budget if isinstance(budget, TokenBudget) else TokenBudget(
        max_tokens_per_run=budget or config.default_token_budget
    )
    notes: list[str] = []
    own_client = github_client is None
    client = github_client or GitHubClient(
        token=config.github_token.get_secret_value() if config.github_token else None,
        cache_path=config.cache_dir / "github_cache.sqlite3",
    )

    try:
        effective_n = n_similar
        if token_budget.max_tokens_per_run < 20000:
            effective_n = min(n_similar, 2)
            if effective_n < n_similar:
                notes.append("Reduced similar repo count due to low token budget.")

        seed = await _prepare_ingested_repo(repo_url, config, client, ingester=ingester)
        seed_analysis = await analyze_repo(
            seed,
            token_budget,
            settings=config,
            completion_func=completion_func,
        )

        search_strategies = await generate_search_strategies(
            seed_analysis,
            token_budget,
            settings=config,
            completion_func=completion_func,
        )
        if search_strategies:
            notes.append(
                f"Generated {len(search_strategies)} search strategies "
                f"({sum(1 for s in search_strategies if s.strategy_type == 'lateral')} lateral)."
            )

        similar_repos = await find_similar_repos(
            repo_url=repo_url,
            n=effective_n,
            filters=filters or SearchFilters(),
            settings=config,
            github_client=client,
            embedding_func=embedding_func,
            ingester=ingester,
            strategies=search_strategies or None,
        )

        ingest_tasks = [
            _prepare_ingested_repo(str(repo.html_url), config, client, ingester=ingester)
            for repo in similar_repos
        ]
        ingested_results = await asyncio.gather(*ingest_tasks, return_exceptions=True)

        ingested_repos: list[IngestedRepo] = []
        for repo, result in zip(similar_repos, ingested_results, strict=True):
            if isinstance(result, Exception):
                notes.append(f"Failed to ingest {repo.full_name}: {result}")
                continue
            ingested_repos.append(result)

        feature_reports = []
        for ingested in ingested_repos:
            if token_budget.exhausted:
                notes.append("Token budget exhausted before feature extraction completed.")
                break
            feature_reports.append(
                await extract_features(
                    ingested,
                    target_context,
                    token_budget,
                    settings=config,
                    completion_func=completion_func,
                )
            )

        comparisons = []
        if token_budget.remaining_tokens >= 4000:
            for ingested in ingested_repos[:2]:
                comparisons.append(
                    await compare_repos(
                        seed,
                        ingested,
                        token_budget,
                        settings=config,
                        completion_func=completion_func,
                    )
                )
        else:
            notes.append("Skipped repo comparisons due to low remaining token budget.")

        prioritized = []
        if feature_reports and not token_budget.exhausted:
            prioritized = await prioritize_ideas(
                feature_reports,
                target_context,
                token_budget,
                settings=config,
                completion_func=completion_func,
            )
        elif not feature_reports:
            notes.append("No feature reports were produced.")

        return ScoutResult(
            seed_repo=seed.metadata,
            seed_analysis=seed_analysis,
            similar_repos=similar_repos,
            feature_reports=feature_reports,
            prioritized_ideas=prioritized,
            comparisons=comparisons,
            search_strategies=search_strategies,
            budget=token_budget,
            partial=bool(notes) or token_budget.exhausted,
            notes=notes,
        )
    finally:
        if own_client:
            await client.close()


async def _prepare_ingested_repo(
    repo_url: str,
    settings: Settings,
    github_client: GitHubClient,
    ingester: IngestCallable | None = None,
) -> IngestedRepo:
    metadata = await _resolve_metadata(repo_url, settings, github_client, ingester=ingester)
    ingested = await ingest_repo(
        metadata,
        max_tokens=settings.max_tokens,
        github_token=settings.github_token.get_secret_value() if settings.github_token else None,
        ingester=ingester or DEFAULT_INGESTER,
    )
    if _is_github_url(repo_url):
        readme = await github_client.get_readme(metadata.full_name)
        if readme:
            ingested = ingested.model_copy(update={"readme": readme})
    return ingested


async def _resolve_metadata(
    repo_url: str,
    settings: Settings,
    github_client: GitHubClient,
    ingester: IngestCallable | None = None,
) -> RepoMetadata:
    if not _is_github_url(repo_url):
        ingested = await ingest_repo(
            repo_url,
            max_tokens=settings.max_tokens,
            github_token=(
                settings.github_token.get_secret_value() if settings.github_token else None
            ),
            ingester=ingester or DEFAULT_INGESTER,
        )
        return ingested.metadata

    full_name = _github_full_name(repo_url)
    repos = await github_client.get_repo_metadata([full_name])
    if repos:
        return repos[0]
    raise RuntimeError(f"Repository metadata not found for {repo_url}")


def _repo_rerank_text(repo: RepoMetadata) -> str:
    parts = [repo.full_name]
    if repo.description:
        parts.append(repo.description)
    if repo.language:
        parts.append(repo.language)
    if repo.topics:
        parts.append(" ".join(repo.topics))
    return " | ".join(parts)


def _is_github_url(repo_url: str) -> bool:
    parsed = urlparse(repo_url)
    return parsed.netloc.endswith("github.com")


def _github_full_name(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
    return f"{path_parts[0]}/{path_parts[1].removesuffix('.git')}"
