"""CLI entrypoints for repoinspo."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from repoinspo.config import Settings, get_settings
from repoinspo.models import ScoutResult, SearchFilters

app = typer.Typer(no_args_is_help=True, help="Analyze repos and surface portable ideas.")
console = Console()


@app.command("run")
def run_command(
    url: Annotated[str, typer.Argument(help="Repository URL to analyze.")],
    n: Annotated[int, typer.Option("-n", help="Number of similar repos to inspect.")] = 5,
    context: Annotated[
        str | None,
        typer.Option("--context", help="Target project context."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override the default model."),
    ] = None,
    output: Annotated[
        Literal["pretty", "json", "md"],
        typer.Option("--output"),
    ] = "pretty",
    budget: Annotated[int | None, typer.Option("--budget", min=1)] = None,
    created_after: Annotated[str | None, typer.Option("--created-after")] = None,
    created_before: Annotated[str | None, typer.Option("--created-before")] = None,
    pushed_after: Annotated[str | None, typer.Option("--pushed-after")] = None,
    min_stars: Annotated[int | None, typer.Option("--min-stars", min=0)] = None,
    max_stars: Annotated[int | None, typer.Option("--max-stars", min=0)] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    topic: Annotated[list[str] | None, typer.Option("--topic")] = None,
    archived: Annotated[bool | None, typer.Option("--archived")] = None,
    license_name: Annotated[str | None, typer.Option("--license")] = None,
) -> None:
    """Run the full scouting pipeline from the CLI."""

    result = asyncio.run(
        _run_pipeline(
            repo_url=url,
            n_similar=n,
            target_context=context,
            model=model,
            output=output,
            budget=budget,
            filters=_build_search_filters(
                created_after=created_after,
                created_before=created_before,
                pushed_after=pushed_after,
                min_stars=min_stars,
                max_stars=max_stars,
                language=language,
                topic=topic,
                archived=archived,
                license_name=license_name,
            ),
        )
    )
    _render_result(result, output)


@app.command("serve")
def serve_command(
    transport: Annotated[Literal["stdio", "http"], typer.Option("--transport")] = "stdio",
) -> None:
    """Serve repoinspo as an MCP server."""

    from repoinspo.server import run_server

    run_server(transport=transport)


def main() -> None:
    app()


async def _run_pipeline(
    repo_url: str,
    n_similar: int,
    target_context: str | None,
    model: str | None,
    output: Literal["pretty", "json", "md"],
    budget: int | None,
    filters: SearchFilters,
) -> ScoutResult:
    del output
    from repoinspo.core.pipeline import scout_ideas

    settings = _resolve_settings(model)
    return await scout_ideas(
        repo_url=repo_url,
        n_similar=n_similar,
        target_context=target_context,
        budget=budget or settings.default_token_budget,
        filters=filters,
        settings=settings,
    )


def _resolve_settings(model: str | None) -> Settings:
    settings = get_settings()
    if not model:
        return settings
    return settings.model_copy(update={"llm_models": [model], "council_enabled": False})


def _build_search_filters(
    *,
    created_after: str | None,
    created_before: str | None,
    pushed_after: str | None,
    min_stars: int | None,
    max_stars: int | None,
    language: str | None,
    topic: list[str] | None,
    archived: bool | None,
    license_name: str | None,
) -> SearchFilters:
    return SearchFilters(
        created_after=_to_datetime(created_after),
        created_before=_to_datetime(created_before),
        pushed_after=_to_datetime(pushed_after),
        min_stars=min_stars,
        max_stars=max_stars,
        language=language,
        topics=topic or [],
        archived=archived,
        license=license_name,
    )


def _to_datetime(value: date | None) -> datetime | None:
    if value is None:
        return None
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _render_result(result: ScoutResult, output: Literal["pretty", "json", "md"]) -> None:
    if output == "json":
        console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))
        return
    if output == "md":
        console.print(_render_markdown(result))
        return
    _render_pretty(result)


def _render_pretty(result: ScoutResult) -> None:
    summary = Table(title="Seed Repository")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Repository", result.seed_repo.full_name)
    summary.add_row("Purpose", result.seed_analysis.purpose)
    summary.add_row("Architecture", result.seed_analysis.architecture)
    summary.add_row("Summary", result.seed_analysis.summary or "")
    console.print(summary)

    similar = Table(title="Similar Repositories")
    similar.add_column("Repository")
    similar.add_column("Stars", justify="right")
    similar.add_column("Language")
    for repo in result.similar_repos:
        similar.add_row(repo.full_name, str(repo.stars), repo.language or "")
    console.print(similar)

    ideas = Table(title="Prioritized Ideas")
    ideas.add_column("Idea")
    ideas.add_column("Score", justify="right")
    ideas.add_column("Source")
    for idea in result.prioritized_ideas:
        ideas.add_row(idea.title, str(idea.priority_score), idea.source_repo)
    console.print(ideas)


def _render_markdown(result: ScoutResult) -> str:
    lines = [
        f"# {result.seed_repo.full_name}",
        "",
        f"- Purpose: {result.seed_analysis.purpose}",
        f"- Architecture: {result.seed_analysis.architecture}",
        "",
        "## Similar Repositories",
    ]
    lines.extend(f"- {repo.full_name} ({repo.stars} stars)" for repo in result.similar_repos)
    lines.append("")
    lines.append("## Prioritized Ideas")
    lines.extend(
        f"- {idea.title} ({idea.priority_score}/10) from {idea.source_repo}"
        for idea in result.prioritized_ideas
    )
    return "\n".join(lines)
