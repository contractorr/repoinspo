from __future__ import annotations

from repoinspo.core.ingestion import estimate_tokens, ingest_repo


async def _fake_ingester(_: str, token: str | None = None) -> tuple[str, str, str]:
    assert token == "secret"
    return "summary", "tree", "content"


async def _long_ingester(_: str, token: str | None = None) -> tuple[str, str, str]:
    del token
    return "summary", "tree", "word " * 1000


async def _failing_ingester(_: str, token: str | None = None) -> tuple[str, str, str]:
    del token
    raise RuntimeError("boom")


async def _fallback_runner(source: str) -> tuple[str, str, str]:
    assert source == "https://github.com/octo/example"
    return "", "", "fallback content"


async def test_ingest_repo_uses_gitingest_output() -> None:
    ingested = await ingest_repo(
        "https://github.com/octo/example",
        max_tokens=200,
        github_token="secret",
        ingester=_fake_ingester,
    )

    assert ingested.metadata.full_name == "octo/example"
    assert ingested.readme == "summary"
    assert ingested.file_tree == "tree"
    assert ingested.content == "content"
    assert ingested.truncated is False


async def test_ingest_repo_truncates_content_to_budget() -> None:
    ingested = await ingest_repo(
        "https://github.com/octo/example",
        max_tokens=50,
        ingester=_long_ingester,
    )

    assert ingested.truncated is True
    assert estimate_tokens((ingested.readme or "") + ingested.file_tree + ingested.content) <= 50


async def test_ingest_repo_falls_back_to_repomix() -> None:
    ingested = await ingest_repo(
        "https://github.com/octo/example",
        max_tokens=200,
        ingester=_failing_ingester,
        fallback_runner=_fallback_runner,
    )

    assert ingested.content == "fallback content"
    assert ingested.readme is None
