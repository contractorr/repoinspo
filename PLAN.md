# repoinspo — Implementation Plan

## Context

No existing tool does end-to-end "understand repo → find similar repos → read code → prioritize portable ideas." Pieces exist (SimRepo for similarity, Repomix/GitIngest for ingestion, litellm for LLM abstraction) but nobody composes them. We're building the glue + the analysis layer.

## Stack

- Python 3.11+, `pyproject.toml` with hatchling
- CLI: Typer + Rich
- MCP server: FastMCP (`mcp[cli]`)
- LLM: litellm (Claude, GPT-4, Ollama — configurable, multi-model council mode)
- GitHub: httpx async (REST search + GraphQL batch metadata)
- Embeddings: litellm.embedding() + FAISS IndexFlatIP
- Ingestion: gitingest (Python-native), repomix subprocess fallback
- Config: pydantic-settings (reads `.env`)
- Cache: SQLite (ETag + response cache)

## File Structure

```
repoinspo/
├── pyproject.toml
├── .env.example
├── tests/
│   ├── conftest.py
│   ├── test_github.py
│   ├── test_ingestion.py
│   ├── test_analysis.py
│   └── test_pipeline.py
└── src/repoinspo/
    ├── __init__.py
    ├── cli.py             # Typer: `run` + `serve` subcommands
    ├── server.py          # FastMCP + @mcp.tool decorators
    ├── config.py          # Settings(BaseSettings)
    ├── models.py          # All Pydantic models
    ├── prompts.py         # LLM prompt templates
    └── core/
        ├── __init__.py
        ├── github.py      # GitHubClient: REST search, GraphQL batch, ETag cache
        ├── ingestion.py   # gitingest wrapper + repomix fallback + truncation
        ├── analysis.py    # analyze_repo, extract_features, compare_repos via litellm
        ├── council.py     # multi-model LLM council: fan-out, collect, synthesize
        ├── embeddings.py  # EmbeddingIndex: FAISS + litellm.embedding()
        └── pipeline.py    # scout_ideas: full orchestration
```

## Phases

### Phase 1: Skeleton + Config + Models
- [ ] Init repo, pyproject.toml w/ all deps
- [ ] `config.py` — Settings (github_token, llm_models, api_keys, embedding_model, cache_dir, max_tokens, default_token_budget, council_enabled)
- [ ] Support multiple API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc. all in `.env`
- [ ] `llm_models` is a list (e.g. `["anthropic/claude-sonnet-4-20250514", "gpt-4o", "gemini/gemini-2.5-pro"]`) — single model = normal mode, multiple = council mode
- [ ] `models.py` — RepoMetadata, IngestedRepo, RepoAnalysis, ExtractedFeature, PortableIdea, ScoutResult, SearchFilters, TokenBudget
- [ ] `prompts.py` — stub constants
- [ ] Verify `pip install -e .` works

### Phase 2: GitHub Client
- [ ] `core/github.py` — async GitHubClient w/ httpx
- [ ] `search_repos(query, sort, per_page, filters)` via REST `GET /search/repositories`
- [ ] `SearchFilters` model — created_after, created_before, pushed_after, min_stars, max_stars, language, topics, archived (bool), license
- [ ] Filters map to GitHub search qualifiers: `created:>2023-01-01`, `stars:100..5000`, `language:python`, `archived:false`, etc.
- [ ] `get_repo_metadata(full_names)` via GraphQL batch
- [ ] `get_readme(full_name)` w/ ETag caching in SQLite
- [ ] Rate-limit middleware (sleep when X-RateLimit-Remaining < 5)
- [ ] `_build_search_query(seed_repo, filters)` — constructs `topic:X language:Y stars:>N created:>DATE`

### Phase 3: Ingestion
- [ ] `core/ingestion.py` — `ingest_repo(repo, max_tokens)`
- [ ] Primary: `gitingest.ingest_async(url)`
- [ ] Token estimation, truncation w/ truncated flag
- [ ] Fallback: `subprocess repomix --remote --compress` if available

### Phase 4: LLM Analysis (core value)
- [ ] `core/analysis.py` — all litellm.acompletion() calls, budget-aware
- [ ] `TokenBudget` model — max_tokens_per_run (total ceiling), track usage across calls via `litellm` response `usage.total_tokens`, stop early when budget exhausted
- [ ] Budget controls depth: low budget → analyze fewer repos, use shorter prompts, skip comparison step. High budget → full deep analysis of all candidates
- [ ] `analyze_repo(ingested, budget)` → RepoAnalysis
- [ ] `extract_features(ingested, target_context, budget)` → FeatureExtractionResult
- [ ] `compare_repos(a, b, budget)` → RepoComparison
- [ ] `prompts.py` — finalize ANALYZE_REPO, EXTRACT_FEATURES, COMPARE_REPOS, PRIORITIZE_IDEAS templates (short + full variants)
- [ ] `_parse_json_response()` — strip markdown fences, validate w/ Pydantic

### Phase 4b: LLM Council
- [ ] `core/council.py` — multi-model council orchestration
- [ ] `council_query(prompt, messages, models, budget)` — fans out same prompt to N models via `asyncio.gather` + `litellm.acompletion()`, collects all responses
- [ ] `synthesize(responses, synthesizer_model)` — takes N model outputs, calls a designated synthesizer model to merge: find consensus, flag disagreements, produce unified output
- [ ] Council mode auto-activates when `len(settings.llm_models) > 1`
- [ ] `analysis.py` calls route through council when enabled — same interface, council is transparent to callers
- [ ] Budget split: total budget divided across N models + 1 synthesis call. Each model gets `budget / (N + 1)` tokens
- [ ] Handles partial failure: if one model errors/times out, council proceeds with remaining responses

### Phase 5: MVP CLI
- [ ] `cli.py` — Typer app
- [ ] `repoinspo run <url> [-n 5] [--context "..."] [--model "..."] [--output pretty|json|md] [--budget 100000] [--created-after 2023-01-01] [--min-stars 50] [--language python]`
- [ ] `repoinspo serve [--transport stdio|http]`
- [ ] Smoke test end-to-end

### Phase 6: MCP Server (Claude Code integration)
- [ ] `server.py` — FastMCP w/ lifespan (shared GitHubClient)
- [ ] 5 tools: `analyze_repo`, `find_similar_repos`, `extract_features`, `compare_repos`, `scout_ideas`
- [ ] All logging to stderr (never stdout in MCP mode)
- [ ] Test w/ `fastmcp dev`
- [ ] Claude Code config snippet for `.claude/settings.json`:
  ```json
  { "mcpServers": { "repoinspo": { "command": "repoinspo", "args": ["serve"] } } }
  ```
- [ ] Also works w/ Claude Desktop `claude_desktop_config.json` same format

### Phase 7: Embeddings + Vector Reranking
- [ ] `core/embeddings.py` — EmbeddingIndex wrapping FAISS IndexFlatIP
- [ ] Embed repo descriptions via litellm.aembedding()
- [ ] Hybrid search: REST keyword results → vector rerank → top-k
- [ ] Persist index to `~/.repoinspo/index.faiss`

### Phase 8: Full Pipeline + Polish
- [ ] `core/pipeline.py` — `scout_ideas()` orchestration (analyze → search → ingest top-N → extract → prioritize)
- [ ] Parallel ingestion of similar repos via asyncio.gather
- [ ] Rich table output for CLI
- [ ] README w/ Claude Desktop MCP config snippet
- [ ] GitHub Actions CI (ruff + pytest)

## Key MCP Tools

| Tool | Input | Output | Description |
|---|---|---|---|
| `analyze_repo` | `repo_url` | RepoAnalysis | Purpose, architecture, features, tech stack |
| `find_similar_repos` | `repo_url, n, filters?` | list[RepoMetadata] | Keyword search + embedding rerank, filterable |
| `extract_features` | `repo_url, target_context?` | list[ExtractedFeature] | Portable features w/ portability scores 1-10 |
| `compare_repos` | `repo_a_url, repo_b_url` | RepoComparison | Common patterns, unique features, recommendation |
| `scout_ideas` | `repo_url, n_similar, target_context?, budget?, filters?` | ScoutResult | Full pipeline → prioritized portable ideas |

## Key Architectural Decisions

- **JSON via instruction, not function-calling** — litellm unified API can't rely on provider-specific `tools=` consistently. Instruct model to return JSON, parse w/ Pydantic.
- **Hybrid search** — GitHub REST keyword search (fast, up to 1000 results) then embed + FAISS cosine rerank top-100. Cheaper and more reliable than pure embedding search.
- **Single GitHubClient via MCP lifespan** — one httpx.AsyncClient at startup, reused by all tool calls. Avoids TCP overhead, respects connection pools.
- **Two-pass analysis** — shallow scan (README + file tree) for discovery, deep scan (full content via gitingest) only for top-N candidates. Controls cost.
- **Token budget system** — `TokenBudget(max_tokens=100000)` passed through the pipeline. Each litellm call's `response.usage.total_tokens` is accumulated. Pipeline adapts: low budget → fewer repos analyzed, shorter prompts, skip compare step. Budget exhaustion stops gracefully and returns partial results. Default budget configurable in `.env` or per-run via `--budget`.
- **Search filters** — `SearchFilters` Pydantic model maps directly to GitHub search qualifiers. CLI exposes as flags (`--created-after`, `--min-stars`, `--language`), MCP tools accept as optional dict. Filters applied in `_build_search_query()` before any API call.
- **LLM council** — when multiple models configured, same prompt fans out to all models in parallel via `asyncio.gather`. A synthesizer model merges responses: consensus items get boosted, disagreements flagged. Transparent to callers — `analysis.py` functions have same interface regardless of single/council mode. Partial failure tolerant.
- **MCP-first design** — MCP server is the primary integration point. Works from Claude Code (`repoinspo serve` in `.claude/settings.json`), Claude Desktop, Cursor, or any MCP client. CLI is a convenience wrapper that calls the same core functions.

## Verification

1. `pip install -e ".[dev]" && pytest` — unit tests w/ mocked litellm + httpx
2. `repoinspo run https://github.com/tiangolo/fastapi --context "building a web framework" -n 3` — end-to-end CLI
3. `fastmcp dev src/repoinspo/server.py` — MCP Inspector smoke test
4. Claude Desktop config: add `repoinspo serve` as MCP server, test `scout_ideas` tool call
5. `pytest -m integration` — real API calls (requires GITHUB_TOKEN + at least one LLM API key)
6. Council test: set `LLM_MODELS=anthropic/claude-sonnet-4-20250514,gpt-4o` in `.env`, run `repoinspo run <url>` — verify output includes synthesis from both models

## Resolved Questions

1. **BigQuery co-stargazer** — **Deferred entirely.** REST search + embedding rerank is enough. No GCP dependency. Revisit post-MVP only if keyword search proves too noisy.
2. **Embedding model portability** — **Graceful degradation.** Runtime check for embedding support; skip vector reranking and use pure keyword search when unavailable (e.g., Ollama without embedding model).
3. **Persistent vs ephemeral FAISS index** — **Ephemeral.** Built per-run, discarded after. No eviction logic, no stale data. Revisit if repeated usage pattern emerges.
4. **Repo name** — **repoinspo**
