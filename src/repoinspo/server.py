"""MCP server entrypoints for repoinspo."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, TypedDict
from urllib.parse import urlparse

from mcp.server.fastmcp import Context, FastMCP

from repoinspo.config import Settings, get_settings
from repoinspo.core.analysis import analyze_repo as analyze_repo_impl
from repoinspo.core.analysis import compare_repos as compare_repos_impl
from repoinspo.core.analysis import extract_features as extract_features_impl
from repoinspo.core.github import GitHubClient
from repoinspo.core.ingestion import ingest_repo
from repoinspo.models import SearchFilters, TokenBudget


class AppState(TypedDict):
    github_client: GitHubClient
    settings: Settings


def create_mcp_server(settings: Settings | None = None) -> FastMCP:
    """Create the FastMCP server with shared lifespan state."""

    configure_logging()
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        async with GitHubClient(
            token=config.github_token.get_secret_value() if config.github_token else None,
            cache_path=config.cache_dir / "github_cache.sqlite3",
        ) as github_client:
            yield AppState(github_client=github_client, settings=config)

    server = FastMCP(
        name="repoinspo",
        instructions="Analyze repositories, find similar repos, and extract portable ideas.",
        lifespan=lifespan,
        log_level="ERROR",
    )

    @server.tool()
    async def analyze_repo(
        repo_url: str,
        budget: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        state = _state_from_context(ctx, config)
        ingested = await _prepare_ingested_repo(repo_url, state)
        analysis = await analyze_repo_impl(
            ingested,
            TokenBudget(max_tokens_per_run=budget or state["settings"].default_token_budget),
            settings=state["settings"],
        )
        return analysis.model_dump(mode="json")

    @server.tool()
    async def find_similar_repos(
        repo_url: str,
        n: int = 5,
        filters: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        state = _state_from_context(ctx, config)
        from repoinspo.core.pipeline import find_similar_repos as find_similar_repos_impl

        repos = await find_similar_repos_impl(
            repo_url=repo_url,
            n=n,
            filters=SearchFilters.model_validate(filters or {}),
            settings=state["settings"],
            github_client=state["github_client"],
        )
        return [repo.model_dump(mode="json") for repo in repos]

    @server.tool()
    async def extract_features(
        repo_url: str,
        target_context: str | None = None,
        budget: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        state = _state_from_context(ctx, config)
        ingested = await _prepare_ingested_repo(repo_url, state)
        features = await extract_features_impl(
            ingested,
            target_context,
            TokenBudget(max_tokens_per_run=budget or state["settings"].default_token_budget),
            settings=state["settings"],
        )
        return features.model_dump(mode="json")

    @server.tool()
    async def compare_repos(
        repo_a_url: str,
        repo_b_url: str,
        budget: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        state = _state_from_context(ctx, config)
        repo_a = await _prepare_ingested_repo(repo_a_url, state)
        repo_b = await _prepare_ingested_repo(repo_b_url, state)
        comparison = await compare_repos_impl(
            repo_a,
            repo_b,
            TokenBudget(max_tokens_per_run=budget or state["settings"].default_token_budget),
            settings=state["settings"],
        )
        return comparison.model_dump(mode="json")

    @server.tool()
    async def scout_ideas(
        repo_url: str,
        n_similar: int = 5,
        target_context: str | None = None,
        budget: int | None = None,
        filters: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        state = _state_from_context(ctx, config)
        from repoinspo.core.pipeline import scout_ideas as scout_ideas_impl

        result = await scout_ideas_impl(
            repo_url=repo_url,
            n_similar=n_similar,
            target_context=target_context,
            budget=budget or state["settings"].default_token_budget,
            filters=SearchFilters.model_validate(filters or {}),
            settings=state["settings"],
            github_client=state["github_client"],
        )
        return result.model_dump(mode="json")

    return server


def run_server(transport: str = "stdio") -> None:
    server = create_mcp_server()
    mapped_transport = "streamable-http" if transport == "http" else transport
    server.run(transport=mapped_transport)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _state_from_context(ctx: Context | None, settings: Settings) -> AppState:
    if ctx is None:
        raise RuntimeError("MCP context is required for server tool execution")
    return ctx.request_context.lifespan_context or AppState(
        github_client=GitHubClient(
            token=settings.github_token.get_secret_value() if settings.github_token else None,
            cache_path=settings.cache_dir / "github_cache.sqlite3",
        ),
        settings=settings,
    )


async def _prepare_ingested_repo(repo_url: str, state: AppState):
    metadata = await _resolve_metadata(repo_url, state)
    ingested = await ingest_repo(
        metadata,
        max_tokens=state["settings"].max_tokens,
        github_token=state["settings"].github_token.get_secret_value()
        if state["settings"].github_token
        else None,
    )
    if _is_github_url(repo_url):
        readme = await state["github_client"].get_readme(metadata.full_name)
        if readme:
            ingested = ingested.model_copy(update={"readme": readme})
    return ingested


async def _resolve_metadata(repo_url: str, state: AppState):
    if not _is_github_url(repo_url):
        return (await ingest_repo(repo_url, max_tokens=state["settings"].max_tokens)).metadata
    full_name = _github_full_name(repo_url)
    repos = await state["github_client"].get_repo_metadata([full_name])
    if repos:
        return repos[0]
    raise RuntimeError(f"Repository metadata not found for {repo_url}")


def _is_github_url(repo_url: str) -> bool:
    parsed = urlparse(repo_url)
    return parsed.netloc.endswith("github.com")


def _github_full_name(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
    return f"{path_parts[0]}/{path_parts[1].removesuffix('.git')}"
