from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
import pytest

from repoinspo.core.github import GitHubClient
from repoinspo.models import RepoMetadata, SearchFilters


def test_filters_translate_to_github_qualifiers() -> None:
    filters = SearchFilters(
        created_after=datetime(2024, 1, 1),
        created_before=datetime(2024, 12, 31),
        pushed_after=datetime(2025, 1, 1),
        min_stars=10,
        max_stars=100,
        language="python",
        topics=["mcp", "llm"],
        archived=False,
        license="mit",
    )

    qualifiers = GitHubClient._filters_to_qualifiers(filters)

    assert qualifiers == [
        "created:>=2024-01-01",
        "created:<=2024-12-31",
        "pushed:>=2025-01-01",
        "stars:10..100",
        "language:python",
        "topic:mcp",
        "topic:llm",
        "archived:false",
        "license:mit",
    ]


def test_build_search_query_uses_seed_repo_defaults() -> None:
    seed_repo = RepoMetadata(
        full_name="contractorr/repoinspo",
        name="repoinspo",
        owner="contractorr",
        html_url="https://github.com/contractorr/repoinspo",
        stars=200,
        language="Python",
        topics=["mcp", "analysis"],
    )

    query = GitHubClient._build_search_query(seed_repo)

    assert query == "topic:mcp topic:analysis language:Python stars:>=100"


@pytest.mark.asyncio
async def test_search_repos_maps_rest_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search/repositories"
        assert request.url.params["q"] == "repo analysis language:python"
        return httpx.Response(
            200,
            headers={"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "9999999999"},
            json={
                "items": [
                    {
                        "full_name": "octo/example",
                        "name": "example",
                        "owner": {"login": "octo"},
                        "html_url": "https://github.com/octo/example",
                        "description": "Example",
                        "stargazers_count": 42,
                        "language": "Python",
                        "topics": ["analysis"],
                        "archived": False,
                        "license": {"spdx_id": "MIT"},
                        "created_at": "2025-01-01T00:00:00Z",
                        "pushed_at": "2025-01-02T00:00:00Z",
                        "default_branch": "main",
                    }
                ]
            },
            request=request,
        )

    async with GitHubClient(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ),
        cache_path=tmp_path / "cache.sqlite3",
    ) as client:
        repos = await client.search_repos(
            "repo analysis",
            filters=SearchFilters(language="python"),
        )

    assert len(repos) == 1
    assert repos[0].full_name == "octo/example"
    assert repos[0].stars == 42


@pytest.mark.asyncio
async def test_get_repo_metadata_batches_graphql_results(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graphql"
        assert "repo_0" in request.content.decode("utf-8")
        return httpx.Response(
            200,
            headers={"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "9999999999"},
            json={
                "data": {
                    "repo_0": {
                        "name": "example",
                        "nameWithOwner": "octo/example",
                        "url": "https://github.com/octo/example",
                        "description": "Example",
                        "stargazerCount": 5,
                        "isArchived": False,
                        "createdAt": "2025-01-01T00:00:00Z",
                        "pushedAt": "2025-01-02T00:00:00Z",
                        "primaryLanguage": {"name": "Python"},
                        "repositoryTopics": {"nodes": [{"topic": {"name": "mcp"}}]},
                        "licenseInfo": {"spdxId": "MIT", "name": "MIT License", "key": "mit"},
                        "defaultBranchRef": {"name": "main"},
                        "owner": {"login": "octo"},
                    }
                }
            },
            request=request,
        )

    async with GitHubClient(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ),
        cache_path=tmp_path / "cache.sqlite3",
    ) as client:
        repos = await client.get_repo_metadata(["octo/example"])

    assert [repo.full_name for repo in repos] == ["octo/example"]
    assert repos[0].topics == ["mcp"]


@pytest.mark.asyncio
async def test_get_readme_uses_etag_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={
                    "ETag": '"v1"',
                    "X-RateLimit-Remaining": "10",
                    "X-RateLimit-Reset": "9999999999",
                },
                text="# README",
                request=request,
            )
        assert request.headers["If-None-Match"] == '"v1"'
        return httpx.Response(
            304,
            headers={"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "9999999999"},
            request=request,
        )

    async with GitHubClient(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ),
        cache_path=tmp_path / "cache.sqlite3",
    ) as client:
        first = await client.get_readme("octo/example")
        second = await client.get_readme("octo/example")

    assert first == "# README"
    assert second == "# README"
    assert calls == 2


@pytest.mark.asyncio
async def test_rate_limit_sleep_triggers_when_remaining_is_low(tmp_path: Path) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-RateLimit-Remaining": "4", "X-RateLimit-Reset": "9999999999"},
            json={"items": []},
            request=request,
        )

    async with GitHubClient(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ),
        cache_path=tmp_path / "cache.sqlite3",
        sleep_func=fake_sleep,
    ) as client:
        await client.search_repos("repo analysis")

    assert sleeps
