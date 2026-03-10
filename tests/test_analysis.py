from __future__ import annotations

from repoinspo.config import Settings
from repoinspo.core.analysis import (
    _parse_json_response,
    analyze_repo,
    compare_repos,
    extract_features,
    prioritize_ideas,
)
from repoinspo.models import IngestedRepo, RepoMetadata, TokenBudget


def _make_ingested_repo(name: str) -> IngestedRepo:
    metadata = RepoMetadata(
        full_name=f"octo/{name}",
        name=name,
        owner="octo",
        html_url=f"https://github.com/octo/{name}",
        stars=42,
        language="Python",
    )
    return IngestedRepo(
        metadata=metadata,
        source_url=str(metadata.html_url),
        readme="summary",
        file_tree="tree",
        content="content",
        token_estimate=10,
    )


async def _analysis_completion(**_: object) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": """
                    ```json
                    {
                      "purpose": "Analyze repos",
                      "architecture": "Async services",
                      "features": ["search", "analysis"],
                      "tech_stack": ["python", "httpx"],
                      "notable_patterns": ["dependency injection"],
                      "summary": "Good repo"
                    }
                    ```
                    """
                }
            }
        ],
        "usage": {"total_tokens": 120},
    }


async def _feature_completion(**_: object) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": """
                    {
                      "features": [
                        {
                          "name": "Hybrid search",
                          "description": "REST plus reranking",
                          "portability_score": 9,
                          "rationale": "Easy to reuse",
                          "implementation_notes": "Keep API boundaries thin",
                          "source_files": ["core/github.py"]
                        }
                      ]
                    }
                    """
                }
            }
        ],
        "usage": {"total_tokens": 90},
    }


async def _comparison_completion(**_: object) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": """
                    {
                      "common_patterns": ["async HTTP"],
                      "unique_to_a": ["MCP"],
                      "unique_to_b": ["web UI"],
                      "recommendation": "Reuse the ingestion path from repo A."
                    }
                    """
                }
            }
        ],
        "usage": {"total_tokens": 75},
    }


async def _council_completion(**kwargs: object) -> dict[str, object]:
    model = str(kwargs["model"])
    messages = kwargs["messages"]
    if model == "model-a" and '"responses"' in str(messages[1]["content"]):
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "purpose": "Analyze repos",
                          "architecture": "Layered async services",
                          "features": ["search", "analysis"],
                          "tech_stack": ["python"],
                          "notable_patterns": ["cache", "async"],
                          "summary": "Synthesized"
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 60},
        }
    if model == "model-a":
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "purpose": "A",
                          "architecture": "Layered",
                          "features": ["search"],
                          "tech_stack": ["python"],
                          "notable_patterns": ["cache"],
                          "summary": "First view"
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 40},
        }
    if model == "model-b":
        raise RuntimeError("timeout")
    assert model == "model-c"
    if len(messages) == 2:
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "purpose": "C",
                          "architecture": "Layered",
                          "features": ["analysis"],
                          "tech_stack": ["python"],
                          "notable_patterns": ["async"],
                          "summary": "Second view"
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 45},
        }
    raise AssertionError(f"Unexpected council call for model={model!r}")


def test_token_budget_tracks_usage() -> None:
    budget = TokenBudget(max_tokens_per_run=100)

    budget.record_usage(40)
    budget.record_usage(10)

    assert budget.tokens_used == 50
    assert budget.remaining_tokens == 50
    assert budget.exhausted is False


def test_parse_json_response_handles_markdown_fences() -> None:
    parsed = _parse_json_response("```json\n{\"hello\": \"world\"}\n```")

    assert parsed == {"hello": "world"}


async def test_analyze_repo_returns_structured_analysis() -> None:
    budget = TokenBudget(max_tokens_per_run=500)

    analysis = await analyze_repo(
        _make_ingested_repo("example"),
        budget,
        model="gpt-4o-mini",
        completion_func=_analysis_completion,
    )

    assert analysis.purpose == "Analyze repos"
    assert analysis.repo.full_name == "octo/example"
    assert budget.tokens_used == 120


async def test_extract_features_returns_feature_list() -> None:
    budget = TokenBudget(max_tokens_per_run=500)

    result = await extract_features(
        _make_ingested_repo("example"),
        "building an MCP tool",
        budget,
        completion_func=_feature_completion,
    )

    assert result.features[0].name == "Hybrid search"
    assert budget.tokens_used == 90


async def test_compare_repos_returns_repo_comparison() -> None:
    budget = TokenBudget(max_tokens_per_run=500)

    comparison = await compare_repos(
        _make_ingested_repo("repo-a"),
        _make_ingested_repo("repo-b"),
        budget,
        completion_func=_comparison_completion,
    )

    assert comparison.common_patterns == ["async HTTP"]
    assert comparison.recommendation.startswith("Reuse")


async def test_analyze_repo_uses_council_mode_with_partial_failure() -> None:
    budget = TokenBudget(max_tokens_per_run=500)
    settings = Settings(LLM_MODELS="model-a,model-b,model-c", COUNCIL_ENABLED="true")

    analysis = await analyze_repo(
        _make_ingested_repo("example"),
        budget,
        settings=settings,
        completion_func=_council_completion,
    )

    assert analysis.summary == "Synthesized"
    assert budget.tokens_used == 145


async def test_prioritize_ideas_returns_portable_ideas() -> None:
    budget = TokenBudget(max_tokens_per_run=500)

    async def _prioritize_completion(**_: object) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "prioritized_ideas": [
                            {
                              "title": "Hybrid search",
                              "description": "Combine keyword and embeddings.",
                              "priority_score": 8,
                              "rationale": "High leverage",
                              "source_repo": "octo/example",
                              "related_features": ["Hybrid search"]
                            }
                          ]
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 50},
        }

    ideas = await prioritize_ideas(
        [
            await extract_features(
                _make_ingested_repo("example"),
                "building an MCP tool",
                TokenBudget(max_tokens_per_run=500),
                completion_func=_feature_completion,
            )
        ],
        "building an MCP tool",
        budget,
        completion_func=_prioritize_completion,
    )

    assert ideas[0].title == "Hybrid search"
