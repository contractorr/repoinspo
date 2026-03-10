"""LLM analysis helpers built on litellm."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from litellm import acompletion
from pydantic import BaseModel

from repoinspo.config import Settings, get_settings
from repoinspo.core.council import council_query, synthesize
from repoinspo.models import (
    FeatureExtractionResult,
    IngestedRepo,
    PortableIdea,
    RepoAnalysis,
    RepoComparison,
    SearchStrategy,
    TokenBudget,
)
from repoinspo.prompts import (
    ANALYZE_REPO_FULL,
    ANALYZE_REPO_SHORT,
    COMPARE_REPOS_FULL,
    COMPARE_REPOS_SHORT,
    EXTRACT_FEATURES_FULL,
    EXTRACT_FEATURES_SHORT,
    GENERATE_SEARCH_STRATEGIES,
    PRIORITIZE_IDEAS_FULL,
    PRIORITIZE_IDEAS_SHORT,
)

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


async def analyze_repo(
    ingested: IngestedRepo,
    budget: TokenBudget,
    model: str | None = None,
    settings: Settings | None = None,
    completion_func: Any = acompletion,
) -> RepoAnalysis:
    """Analyze a repository and return structured metadata."""

    prompt = _select_prompt_variant(budget, ANALYZE_REPO_SHORT, ANALYZE_REPO_FULL)
    payload = {
        "repo": ingested.metadata.model_dump(mode="json"),
        "summary": ingested.readme,
        "file_tree": ingested.file_tree,
        "content": ingested.content,
    }
    data = await _json_completion(
        prompt=prompt,
        payload=payload,
        budget=budget,
        model=model,
        settings=settings,
        completion_func=completion_func,
    )
    return RepoAnalysis.model_validate({"repo": ingested.metadata, **data})


async def extract_features(
    ingested: IngestedRepo,
    target_context: str | None,
    budget: TokenBudget,
    model: str | None = None,
    settings: Settings | None = None,
    completion_func: Any = acompletion,
) -> FeatureExtractionResult:
    """Extract portable features from a repository."""

    prompt = _select_prompt_variant(budget, EXTRACT_FEATURES_SHORT, EXTRACT_FEATURES_FULL)
    payload = {
        "repo": ingested.metadata.model_dump(mode="json"),
        "target_context": target_context,
        "summary": ingested.readme,
        "file_tree": ingested.file_tree,
        "content": ingested.content,
    }
    data = await _json_completion(
        prompt=prompt,
        payload=payload,
        budget=budget,
        model=model,
        settings=settings,
        completion_func=completion_func,
    )
    return FeatureExtractionResult.model_validate(
        {
            "repo": ingested.metadata,
            "target_context": target_context,
            **data,
        }
    )


async def compare_repos(
    repo_a: IngestedRepo,
    repo_b: IngestedRepo,
    budget: TokenBudget,
    model: str | None = None,
    settings: Settings | None = None,
    completion_func: Any = acompletion,
) -> RepoComparison:
    """Compare two repositories and summarize overlap and differentiation."""

    prompt = _select_prompt_variant(budget, COMPARE_REPOS_SHORT, COMPARE_REPOS_FULL)
    payload = {
        "repo_a": {
            "repo": repo_a.metadata.model_dump(mode="json"),
            "summary": repo_a.readme,
            "file_tree": repo_a.file_tree,
            "content": repo_a.content,
        },
        "repo_b": {
            "repo": repo_b.metadata.model_dump(mode="json"),
            "summary": repo_b.readme,
            "file_tree": repo_b.file_tree,
            "content": repo_b.content,
        },
    }
    data = await _json_completion(
        prompt=prompt,
        payload=payload,
        budget=budget,
        model=model,
        settings=settings,
        completion_func=completion_func,
    )
    return RepoComparison.model_validate(
        {
            "repo_a": repo_a.metadata,
            "repo_b": repo_b.metadata,
            **data,
        }
    )


async def prioritize_ideas(
    feature_reports: list[FeatureExtractionResult],
    target_context: str | None,
    budget: TokenBudget,
    model: str | None = None,
    settings: Settings | None = None,
    completion_func: Any = acompletion,
) -> list[PortableIdea]:
    """Prioritize extracted features into portable implementation ideas."""

    prompt = _select_prompt_variant(budget, PRIORITIZE_IDEAS_SHORT, PRIORITIZE_IDEAS_FULL)
    payload = {
        "target_context": target_context,
        "feature_reports": [report.model_dump(mode="json") for report in feature_reports],
    }
    data = await _json_completion(
        prompt=prompt,
        payload=payload,
        budget=budget,
        model=model,
        settings=settings,
        completion_func=completion_func,
    )
    ideas = [PortableIdea.model_validate(item) for item in data.get("prioritized_ideas", [])]

    # Cap ideas per source repo — no single repo > 50% of total
    if len(ideas) > 1:
        max_per_repo = max(len(ideas) // 2, 1)
        repo_counts: dict[str, int] = {}
        capped: list[PortableIdea] = []
        for idea in ideas:
            count = repo_counts.get(idea.source_repo, 0)
            if count < max_per_repo:
                capped.append(idea)
                repo_counts[idea.source_repo] = count + 1
        ideas = capped

    return ideas


async def generate_search_strategies(
    seed_analysis: RepoAnalysis,
    budget: TokenBudget,
    settings: Settings | None = None,
    completion_func: Any = acompletion,
) -> list[SearchStrategy]:
    """Generate diverse search strategies from seed analysis via LLM."""

    if budget.remaining_tokens < 2000:
        return []

    payload = seed_analysis.model_dump(mode="json")
    try:
        data = await _json_completion(
            prompt=GENERATE_SEARCH_STRATEGIES,
            payload=payload,
            budget=budget,
            model=None,
            settings=settings,
            completion_func=completion_func,
        )
    except Exception as exc:
        logger.warning("Search strategy generation failed; using static fallback", exc_info=exc)
        return []

    strategies = []
    for item in data.get("strategies", []):
        try:
            strategies.append(SearchStrategy.model_validate(item))
        except Exception:
            logger.debug("Skipping invalid search strategy: %s", item)
    return strategies


def _parse_json_response(
    raw_content: str,
    model_type: type[ModelT] | None = None,
) -> ModelT | dict[str, Any]:
    """Parse JSON returned by a model, tolerating markdown code fences."""

    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start_indices = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if start_indices:
        start = min(start_indices)
        end = max(text.rfind("}"), text.rfind("]"))
        text = text[start : end + 1]

    data = json.loads(text)
    if model_type is None:
        return data
    return model_type.model_validate(data)


async def _json_completion(
    prompt: str,
    payload: dict[str, Any],
    budget: TokenBudget,
    model: str | None,
    settings: Settings | None,
    completion_func: Any,
) -> dict[str, Any]:
    config = settings or get_settings()
    if budget.exhausted:
        raise RuntimeError("Token budget exhausted before completion request")

    user_message = {"role": "user", "content": json.dumps(payload, default=str)}
    if model is None and config.council_mode:
        responses, failures = await council_query(
            prompt=prompt,
            messages=[user_message],
            models=config.llm_models,
            budget=budget,
            completion_func=completion_func,
        )
        if failures:
            logger.warning("Council completed with failed models: %s", ", ".join(failures))
        content = await synthesize(
            prompt=prompt,
            responses=responses,
            synthesizer_model=config.llm_models[0],
            budget=budget,
            model_count=len(config.llm_models),
            completion_func=completion_func,
        )
    else:
        selected_model = model or config.llm_models[0]
        response = await completion_func(
            model=selected_model,
            messages=[
                {"role": "system", "content": prompt},
                user_message,
            ],
            max_tokens=min(config.max_tokens, budget.remaining_tokens),
        )
        budget.record_usage(_extract_total_tokens(response))
        content = _extract_response_text(response)
        logger.debug("Received model response from %s", selected_model)
    parsed = _parse_json_response(content)
    if not isinstance(parsed, dict):
        raise TypeError("Expected JSON object response from model")
    return parsed


def _select_prompt_variant(budget: TokenBudget, short_prompt: str, full_prompt: str) -> str:
    return short_prompt if budget.remaining_tokens < 4000 else full_prompt


def _extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        content = response["choices"][0]["message"]["content"]
    else:
        content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_total_tokens(response: Any) -> int | None:
    if isinstance(response, dict):
        usage = response.get("usage") or {}
        return usage.get("total_tokens")
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is not None:
        return total_tokens
    if isinstance(usage, dict):
        return usage.get("total_tokens")
    return None
