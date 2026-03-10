from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from repoinspo import cli
from repoinspo.config import Settings
from repoinspo.core.analysis import generate_search_strategies
from repoinspo.core.pipeline import find_similar_repos, scout_ideas
from repoinspo.models import (
    FeatureExtractionResult,
    PortableIdea,
    RepoAnalysis,
    RepoMetadata,
    ScoutResult,
    SearchFilters,
    SearchStrategy,
    TokenBudget,
)


def test_settings_parse_comma_separated_models(tmp_path: Path) -> None:
    settings = Settings(
        LLM_MODELS="anthropic/claude-sonnet-4-20250514,gpt-4o",
        CACHE_DIR=str(tmp_path),
    )

    assert settings.llm_models == ["anthropic/claude-sonnet-4-20250514", "gpt-4o"]
    assert settings.cache_dir == tmp_path
    assert settings.council_mode is True


class FakeGitHubClient:
    async def get_repo_metadata(self, full_names: list[str]) -> list[RepoMetadata]:
        return [
            RepoMetadata(
                full_name=full_name,
                name=full_name.split("/")[1],
                owner=full_name.split("/")[0],
                html_url=f"https://github.com/{full_name}",
                description=f"Description for {full_name}",
                stars=100,
                language="Python",
                topics=["mcp", "analysis"],
            )
            for full_name in full_names
        ]

    async def search_repos(
        self,
        query: str,
        sort: str | None = None,
        per_page: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RepoMetadata]:
        del sort, per_page, filters, query
        return [
            RepoMetadata(
                full_name="octo/sim-one",
                name="sim-one",
                owner="octo",
                html_url="https://github.com/octo/sim-one",
                description="First similar repo",
                stars=90,
                language="Python",
                topics=["mcp"],
            ),
            RepoMetadata(
                full_name="octo/sim-two",
                name="sim-two",
                owner="octo",
                html_url="https://github.com/octo/sim-two",
                description="Second similar repo",
                stars=80,
                language="Python",
                topics=["mcp", "analysis"],
            ),
        ]

    async def get_readme(self, full_name: str) -> str:
        return f"# {full_name}"


async def _fake_ingester(source: str, token: str | None = None) -> tuple[str, str, str]:
    del token
    return f"summary for {source}", "tree", f"content for {source}"


async def _fake_completion(**kwargs: object) -> dict[str, object]:
    prompt = str(kwargs["messages"][0]["content"])
    if "generate diverse github search strategies" in prompt.lower():
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "strategies": [
                            {
                              "query": "topic:mcp language:python",
                              "strategy_type": "direct",
                              "rationale": "Find similar MCP tools"
                            },
                            {
                              "query": "plugin architecture extensible",
                              "strategy_type": "lateral",
                              "rationale": "Cross-domain plugin patterns"
                            }
                          ]
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 30},
        }
    if "analyze software repositories" in prompt.lower():
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "purpose": "Analyze repos",
                          "architecture": "Async pipeline",
                          "features": ["search"],
                          "tech_stack": ["python"],
                          "notable_patterns": ["budgeting"],
                          "summary": "Seed summary",
                          "strengths": ["clean async architecture"],
                          "weaknesses": ["limited test coverage"],
                          "opportunities": ["plugin system", "caching layer"]
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 50},
        }
    if "extract portable features" in prompt.lower():
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "features": [
                            {
                              "name": "Async ingestion",
                              "description": "Parallel repo ingestion",
                              "portability_score": 8,
                              "rationale": "Useful pattern",
                              "implementation_notes": "Use gather",
                              "source_files": ["core/pipeline.py"]
                            }
                          ]
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 40},
        }
    if "compare two repositories" in prompt.lower():
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "common_patterns": ["async"],
                          "unique_to_a": ["budget"],
                          "unique_to_b": ["ui"],
                          "recommendation": "Borrow async ingestion."
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 30},
        }
    return {
        "choices": [
            {
                "message": {
                    "content": """
                    {
                      "prioritized_ideas": [
                        {
                          "title": "Async ingestion",
                          "description": "Parallel repository ingestion",
                          "priority_score": 9,
                          "rationale": "High impact",
                          "source_repo": "octo/sim-one",
                          "related_features": ["Async ingestion"]
                        }
                      ]
                    }
                    """
                }
            }
        ],
        "usage": {"total_tokens": 35},
    }


async def _fake_embedding(model: str, input: list[str]) -> dict[str, object]:
    del model
    vectors = []
    for text in input:
        if "analysis" in text:
            vectors.append({"embedding": [1.0, 0.0]})
        else:
            vectors.append({"embedding": [0.5, 0.5]})
    return {"data": vectors}


async def test_find_similar_repos_reranks_candidates() -> None:
    settings = Settings(CACHE_DIR="cache", LLM_MODELS="gpt-4o-mini")

    repos = await find_similar_repos(
        repo_url="https://github.com/octo/seed",
        n=2,
        settings=settings,
        github_client=FakeGitHubClient(),
        embedding_func=_fake_embedding,
        ingester=_fake_ingester,
    )

    assert [repo.full_name for repo in repos] == ["octo/sim-two", "octo/sim-one"]


async def test_scout_ideas_orchestrates_pipeline() -> None:
    settings = Settings(CACHE_DIR="cache", LLM_MODELS="gpt-4o-mini", DEFAULT_TOKEN_BUDGET="20000")

    result = await scout_ideas(
        repo_url="https://github.com/octo/seed",
        n_similar=2,
        target_context="building an MCP tool",
        settings=settings,
        github_client=FakeGitHubClient(),
        completion_func=_fake_completion,
        embedding_func=_fake_embedding,
        ingester=_fake_ingester,
    )

    assert result.seed_analysis.purpose == "Analyze repos"
    assert result.seed_analysis.strengths == ["clean async architecture"]
    assert result.seed_analysis.weaknesses == ["limited test coverage"]
    assert result.seed_analysis.opportunities == ["plugin system", "caching layer"]
    assert len(result.search_strategies) == 2
    assert result.search_strategies[0].strategy_type == "direct"
    assert result.search_strategies[1].strategy_type == "lateral"
    assert len(result.similar_repos) == 2
    assert len(result.feature_reports) == 2
    assert result.prioritized_ideas[0].title == "Async ingestion"


def test_cli_renders_json_output(monkeypatch) -> None:
    async def fake_scout_ideas(**_: object) -> ScoutResult:
        repo = RepoMetadata(
            full_name="octo/seed",
            name="seed",
            owner="octo",
            html_url="https://github.com/octo/seed",
        )
        analysis = RepoAnalysis(
            repo=repo,
            purpose="Analyze repos",
            architecture="Async pipeline",
            features=[],
            tech_stack=[],
            notable_patterns=[],
            summary="Seed summary",
        )
        return ScoutResult(
            seed_repo=repo,
            seed_analysis=analysis,
            similar_repos=[],
            feature_reports=[FeatureExtractionResult(repo=repo, features=[])],
            prioritized_ideas=[
                PortableIdea(
                    title="Async ingestion",
                    description="Parallel repo ingestion",
                    priority_score=9,
                    rationale="High impact",
                    source_repo="octo/sim-one",
                    related_features=[],
                )
            ],
            comparisons=[],
            budget=TokenBudget(max_tokens_per_run=1000),
            partial=False,
            notes=[],
        )

    monkeypatch.setattr("repoinspo.core.pipeline.scout_ideas", fake_scout_ideas)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["run", "https://github.com/octo/seed", "--output", "json"])

    assert result.exit_code == 0
    assert '"title": "Async ingestion"' in result.stdout


async def test_generate_search_strategies_returns_strategies() -> None:
    budget = TokenBudget(max_tokens_per_run=10000)
    repo = RepoMetadata(
        full_name="octo/seed",
        name="seed",
        owner="octo",
        html_url="https://github.com/octo/seed",
        stars=100,
        language="Python",
    )
    analysis = RepoAnalysis(
        repo=repo,
        purpose="Analyze repos",
        architecture="Async pipeline",
        strengths=["clean API"],
        weaknesses=["no caching"],
        opportunities=["plugin system"],
    )

    async def _strategy_completion(**_: object) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": """
                        {
                          "strategies": [
                            {
                              "query": "topic:analysis language:python",
                              "strategy_type": "direct",
                              "rationale": "Same domain"
                            },
                            {
                              "query": "plugin architecture extensible",
                              "strategy_type": "lateral",
                              "rationale": "Cross-domain patterns"
                            }
                          ]
                        }
                        """
                    }
                }
            ],
            "usage": {"total_tokens": 40},
        }

    strategies = await generate_search_strategies(
        analysis, budget, completion_func=_strategy_completion
    )

    assert len(strategies) == 2
    assert strategies[0].strategy_type == "direct"
    assert strategies[1].strategy_type == "lateral"
    assert budget.tokens_used == 40


async def test_generate_search_strategies_returns_empty_on_low_budget() -> None:
    budget = TokenBudget(max_tokens_per_run=1000, tokens_used=999)
    repo = RepoMetadata(
        full_name="octo/seed",
        name="seed",
        owner="octo",
        html_url="https://github.com/octo/seed",
    )
    analysis = RepoAnalysis(repo=repo, purpose="test", architecture="test")

    strategies = await generate_search_strategies(analysis, budget)

    assert strategies == []


async def test_find_similar_repos_uses_strategies() -> None:
    settings = Settings(CACHE_DIR="cache", LLM_MODELS="gpt-4o-mini")
    strategies = [
        SearchStrategy(query="plugin architecture", strategy_type="lateral", rationale="test"),
    ]

    repos = await find_similar_repos(
        repo_url="https://github.com/octo/seed",
        n=2,
        settings=settings,
        github_client=FakeGitHubClient(),
        embedding_func=_fake_embedding,
        ingester=_fake_ingester,
        strategies=strategies,
    )

    assert len(repos) == 2
