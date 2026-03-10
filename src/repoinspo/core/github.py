"""Async GitHub client helpers."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import time
from typing import Any

import httpx

from repoinspo.models import RepoMetadata, SearchFilters

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CachedReadme:
    full_name: str
    etag: str | None
    content: str


class GitHubClient:
    """Async wrapper around GitHub REST and GraphQL endpoints."""

    def __init__(
        self,
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
        cache_path: Path | None = None,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=self._build_headers(token),
            timeout=30.0,
        )
        if client is not None and token:
            self._client.headers.update(self._build_headers(token))
        self._sleep = sleep_func
        self._cache_path = cache_path or Path(".cache") / "repoinspo" / "github_cache.sqlite3"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._cache_path)
        self._db.row_factory = sqlite3.Row
        self._initialize_cache()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._db.close()
        if self._owns_client:
            await self._client.aclose()

    async def search_repos(
        self,
        query: str,
        sort: str | None = None,
        per_page: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RepoMetadata]:
        """Search repositories with optional GitHub qualifiers."""

        search_query = self._combine_query(query, filters)
        params: dict[str, Any] = {"q": search_query, "per_page": per_page}
        if sort and sort != "best match":
            params["sort"] = sort
        response = await self._request("GET", "/search/repositories", params=params)
        payload = response.json()
        return [self._repo_from_rest(item) for item in payload.get("items", [])]

    async def get_repo_metadata(self, full_names: Sequence[str]) -> list[RepoMetadata]:
        """Fetch repository metadata in a single GraphQL batch query."""

        if not full_names:
            return []

        query, aliases = self._build_graphql_query(full_names)
        response = await self._request("POST", "/graphql", json={"query": query})
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL errors: {payload['errors']}")

        data = payload.get("data", {})
        results: list[RepoMetadata] = []
        for alias in aliases:
            repo_data = data.get(alias)
            if repo_data is None:
                continue
            results.append(self._repo_from_graphql(repo_data))
        return results

    async def get_readme(self, full_name: str) -> str | None:
        """Fetch the README content and reuse cached content on 304 responses."""

        cached = self._get_cached_readme(full_name)
        headers = {"Accept": "application/vnd.github.raw+json"}
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag

        try:
            response = await self._request("GET", f"/repos/{full_name}/readme", headers=headers)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 304 and cached:
                return cached.content
            if exc.response.status_code == 404:
                return None
            raise

        etag = response.headers.get("ETag")
        content = response.text
        self._cache_readme(full_name, etag, content)
        return content

    def _initialize_cache(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS readme_cache (
                full_name TEXT PRIMARY KEY,
                etag TEXT,
                content TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    @staticmethod
    def _build_headers(token: str | None) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "repoinspo",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        response = await self._client.request(method, url, **kwargs)
        await self._respect_rate_limit(response)
        if response.status_code == 304:
            raise httpx.HTTPStatusError("Not modified", request=response.request, response=response)
        response.raise_for_status()
        return response

    async def _respect_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_at = response.headers.get("X-RateLimit-Reset")
        if remaining is None or reset_at is None:
            return
        try:
            remaining_value = int(remaining)
            reset_value = int(reset_at)
        except ValueError:
            return
        if remaining_value >= 5:
            return
        delay = max(reset_value - int(time()), 0)
        if delay <= 0:
            return
        logger.warning("GitHub rate limit low; sleeping for %s seconds", delay)
        await self._sleep(delay)

    def _combine_query(self, query: str, filters: SearchFilters | None) -> str:
        parts = [query.strip()] if query.strip() else []
        if filters:
            parts.extend(self._filters_to_qualifiers(filters))
        return " ".join(part for part in parts if part)

    def _get_cached_readme(self, full_name: str) -> CachedReadme | None:
        row = self._db.execute(
            "SELECT full_name, etag, content FROM readme_cache WHERE full_name = ?",
            (full_name,),
        ).fetchone()
        if row is None:
            return None
        return CachedReadme(full_name=row["full_name"], etag=row["etag"], content=row["content"])

    def _cache_readme(self, full_name: str, etag: str | None, content: str) -> None:
        self._db.execute(
            """
            INSERT INTO readme_cache (full_name, etag, content, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(full_name) DO UPDATE SET
                etag = excluded.etag,
                content = excluded.content,
                fetched_at = excluded.fetched_at
            """,
            (full_name, etag, content, datetime.now(UTC).isoformat()),
        )
        self._db.commit()

    @classmethod
    def _filters_to_qualifiers(cls, filters: SearchFilters) -> list[str]:
        qualifiers: list[str] = []
        if filters.created_after:
            qualifiers.append(f"created:>={cls._format_date(filters.created_after)}")
        if filters.created_before:
            qualifiers.append(f"created:<={cls._format_date(filters.created_before)}")
        if filters.pushed_after:
            qualifiers.append(f"pushed:>={cls._format_date(filters.pushed_after)}")
        if filters.min_stars is not None and filters.max_stars is not None:
            qualifiers.append(f"stars:{filters.min_stars}..{filters.max_stars}")
        elif filters.min_stars is not None:
            qualifiers.append(f"stars:>={filters.min_stars}")
        elif filters.max_stars is not None:
            qualifiers.append(f"stars:<={filters.max_stars}")
        if filters.language:
            qualifiers.append(f"language:{filters.language}")
        qualifiers.extend(f"topic:{topic}" for topic in filters.topics)
        if filters.archived is not None:
            qualifiers.append(f"archived:{str(filters.archived).lower()}")
        if filters.license:
            qualifiers.append(f"license:{filters.license}")
        return qualifiers

    @classmethod
    def _build_search_query(
        cls,
        seed_repo: RepoMetadata,
        filters: SearchFilters | None = None,
    ) -> str:
        qualifiers: list[str] = []
        if seed_repo.topics:
            qualifiers.extend(f"topic:{topic}" for topic in seed_repo.topics[:3])
        if seed_repo.language and not (filters and filters.language):
            qualifiers.append(f"language:{seed_repo.language}")
        has_star_filter = filters and (
            filters.min_stars is not None or filters.max_stars is not None
        )
        if seed_repo.stars and not has_star_filter:
            qualifiers.append(f"stars:>={max(seed_repo.stars // 2, 1)}")
        if filters:
            qualifiers.extend(cls._filters_to_qualifiers(filters))
        if not qualifiers:
            qualifiers.append(seed_repo.name)
        return " ".join(qualifiers)

    @staticmethod
    def _format_date(value: datetime) -> str:
        return value.date().isoformat()

    @staticmethod
    def _build_graphql_query(full_names: Sequence[str]) -> tuple[str, list[str]]:
        aliases: list[str] = []
        blocks: list[str] = []
        for index, full_name in enumerate(full_names):
            owner, name = full_name.split("/", 1)
            alias = f"repo_{index}"
            aliases.append(alias)
            blocks.append(
                f"""
                {alias}: repository(owner: "{owner}", name: "{name}") {{
                  name
                  nameWithOwner
                  url
                  description
                  stargazerCount
                  isArchived
                  createdAt
                  pushedAt
                  primaryLanguage {{ name }}
                  repositoryTopics(first: 20) {{ nodes {{ topic {{ name }} }} }}
                  licenseInfo {{ key name spdxId }}
                  defaultBranchRef {{ name }}
                  owner {{ login }}
                }}
                """
            )
        return f"query {{{''.join(blocks)}}}", aliases

    @staticmethod
    def _repo_from_rest(payload: dict[str, Any]) -> RepoMetadata:
        license_payload = payload.get("license") or {}
        owner_payload = payload.get("owner") or {}
        return RepoMetadata.model_validate(
            {
                "full_name": payload["full_name"],
                "name": payload["name"],
                "owner": owner_payload.get("login", payload["full_name"].split("/", 1)[0]),
                "html_url": payload["html_url"],
                "description": payload.get("description"),
                "stars": payload.get("stargazers_count", 0),
                "language": payload.get("language"),
                "topics": payload.get("topics") or [],
                "archived": payload.get("archived", False),
                "license": license_payload.get("spdx_id")
                or license_payload.get("name")
                or license_payload.get("key"),
                "created_at": payload.get("created_at"),
                "pushed_at": payload.get("pushed_at"),
                "default_branch": payload.get("default_branch"),
            }
        )

    @staticmethod
    def _repo_from_graphql(payload: dict[str, Any]) -> RepoMetadata:
        license_payload = payload.get("licenseInfo") or {}
        topics_payload = payload.get("repositoryTopics", {}).get("nodes", [])
        return RepoMetadata.model_validate(
            {
                "full_name": payload["nameWithOwner"],
                "name": payload["name"],
                "owner": payload.get("owner", {}).get(
                    "login",
                    payload["nameWithOwner"].split("/", 1)[0],
                ),
                "html_url": payload["url"],
                "description": payload.get("description"),
                "stars": payload.get("stargazerCount", 0),
                "language": (payload.get("primaryLanguage") or {}).get("name"),
                "topics": [node["topic"]["name"] for node in topics_payload if node.get("topic")],
                "archived": payload.get("isArchived", False),
                "license": license_payload.get("spdxId")
                or license_payload.get("name")
                or license_payload.get("key"),
                "created_at": payload.get("createdAt"),
                "pushed_at": payload.get("pushedAt"),
                "default_branch": (payload.get("defaultBranchRef") or {}).get("name"),
            }
        )
