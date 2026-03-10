"""Repository ingestion helpers."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

import tiktoken
from gitingest import ingest_async

from repoinspo.models import IngestedRepo, RepoMetadata

logger = logging.getLogger(__name__)

IngestResult = tuple[str, str, str]
IngestCallable = Callable[..., Awaitable[IngestResult]]
FallbackCallable = Callable[[str], Awaitable[IngestResult]]
DEFAULT_INGESTER: IngestCallable = ingest_async


async def ingest_repo(
    repo: RepoMetadata | str,
    max_tokens: int,
    github_token: str | None = None,
    ingester: IngestCallable = DEFAULT_INGESTER,
    fallback_runner: FallbackCallable | None = None,
) -> IngestedRepo:
    """Ingest a repository with gitingest and repomix fallback support."""

    metadata = _metadata_from_source(repo)
    source = str(metadata.html_url)
    try:
        summary, file_tree, content = await ingester(source, token=github_token)
    except Exception as exc:
        logger.warning("gitingest failed for %s, trying repomix fallback", source, exc_info=exc)
        runner = fallback_runner or _run_repomix
        summary, file_tree, content = await runner(source)

    summary, file_tree, content, token_estimate, truncated = _truncate_sections(
        summary=summary,
        file_tree=file_tree,
        content=content,
        max_tokens=max_tokens,
    )

    return IngestedRepo(
        metadata=metadata,
        source_url=source,
        readme=summary or None,
        file_tree=file_tree,
        content=content,
        token_estimate=token_estimate,
        truncated=truncated,
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count using a stable tokenizer, with a character fallback."""

    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(len(text) // 4, 1)


async def _run_repomix(source: str) -> IngestResult:
    executable = shutil.which("repomix")
    if executable is None:
        raise RuntimeError("repomix is not installed and gitingest ingestion failed")

    process = await asyncio.create_subprocess_exec(
        executable,
        "--remote",
        "--compress",
        source,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or "repomix failed")
    return "", "", stdout.decode("utf-8", errors="replace")


def _truncate_sections(
    summary: str,
    file_tree: str,
    content: str,
    max_tokens: int,
) -> tuple[str, str, str, int, bool]:
    summary_tokens = estimate_tokens(summary)
    tree_tokens = estimate_tokens(file_tree)
    content_tokens = estimate_tokens(content)
    total_tokens = summary_tokens + tree_tokens + content_tokens
    if total_tokens <= max_tokens:
        return summary, file_tree, content, total_tokens, False

    remaining = max_tokens
    truncated = False

    summary, used, did_truncate = _truncate_text(summary, remaining)
    remaining = max(remaining - used, 0)
    truncated = truncated or did_truncate

    file_tree, used, did_truncate = _truncate_text(file_tree, remaining)
    remaining = max(remaining - used, 0)
    truncated = truncated or did_truncate

    content, _, did_truncate = _truncate_text(content, remaining)
    truncated = truncated or did_truncate

    total = estimate_tokens(summary + file_tree + content)
    while total > max_tokens and content:
        content = content[:-1]
        total = estimate_tokens(summary + file_tree + content)
        truncated = True
    return summary, file_tree, content, total, truncated


def _truncate_text(text: str, max_tokens: int) -> tuple[str, int, bool]:
    token_count = estimate_tokens(text)
    if token_count <= max_tokens:
        return text, token_count, False
    if max_tokens <= 0:
        return "", 0, bool(text)

    ratio = max_tokens / token_count
    character_limit = max(1, int(len(text) * ratio))
    truncated = text[:character_limit]
    while truncated and estimate_tokens(truncated) > max_tokens:
        truncated = truncated[:-1]
    return truncated, estimate_tokens(truncated), True


def _metadata_from_source(repo: RepoMetadata | str) -> RepoMetadata:
    if isinstance(repo, RepoMetadata):
        return repo
    parsed = urlparse(repo)
    if parsed.scheme and parsed.netloc:
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2:
            owner, name = path_parts[:2]
            full_name = f"{owner}/{name.removesuffix('.git')}"
            return RepoMetadata(
                full_name=full_name,
                name=name.removesuffix(".git"),
                owner=owner,
                html_url=repo,
            )
    path = Path(repo)
    return RepoMetadata(
        full_name=path.name,
        name=path.name,
        owner="local",
        html_url=path.resolve().as_uri(),
    )
