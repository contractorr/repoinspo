# repoinspo — Implementation Plan

## Context

No existing tool does end-to-end "understand repo → find similar repos → read code → prioritize portable ideas." Pieces exist (SimRepo for similarity, Repomix/GitIngest for ingestion, litellm for LLM abstraction) but nobody composes them. We're building the glue + the analysis layer.

## Stack

- Python 3.11+, `pyproject.toml` with hatchling
- CLI: Typer + Rich
- MCP server: FastMCP (`mcp[cli]`)
- LLM: litellm (Claude, GPT-4, Ollama — configurable)
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
        ├── embeddings.py  # EmbeddingIndex: FAISS + litellm.embedding()
        └── pipeline.py    # scout_ideas: full orchestration
```

## Phases

### Phase 1: Skeleton + Config + Models
- [ ] Init repo, pyproject.toml w/ all deps
- [ ] `config.py` — Settings (github_token, llm_model, embedding_model, cache_dir, max_tokens)
- [ ] `models.py` — RepoMetadata, IngestedRepo, RepoAnalysis, ExtractedFeature, PortableIdea, ScoutResult
- [ ] `prompts.py` — stub constants
- [ ] Verify `pip install -e .` works

### Phase 2: GitHub Client
- [ ] `core/github.py` — async GitHubClient w/ httpx
- [ ] `search_repos(query, sort, per_page)` via REST `GET /search/repositories`
- [ ] `get_repo_metadata(full_names)` via GraphQL batch
- [ ] `get_readme(full_name)` w/ ETag caching in SQLite
- [ ] Rate-limit middleware (sleep when X-RateLimit-Remaining < 5)
- [ ] `_build_search_query(seed_repo)` — constructs `topic:X language:Y stars:>N`

### Phase 3: Ingestion
- [ ] `core/ingestion.py` — `ingest_repo(repo, max_tokens)`
- [ ] Primary: `gitingest.ingest_async(url)`
- [ ] Token estimation, truncation w/ truncated flag
- [ ] Fallback: `subprocess repomix --remote --compress` if available

### Phase 4: LLM Analysis (core value)
- [ ] `core/analysis.py` — all litellm.acompletion() calls
- [ ] `analyze_repo(ingested)` → RepoAnalysis
- [ ] `extract_features(ingested, target_context)` → FeatureExtractionResult
- [ ] `compare_repos(a, b)` → RepoComparison
- [ ] `prompts.py` — finalize ANALYZE_REPO, EXTRACT_FEATURES, COMPARE_REPOS, PRIORITIZE_IDEAS templates
- [ ] `_parse_json_response()` — strip markdown fences, validate w/ Pydantic

### Phase 5: MVP CLI
- [ ] `cli.py` — Typer app
- [ ] `repoinspo run <url> [-n 5] [--context "..."] [--model "..."] [--output pretty|json|md]`
- [ ] `repoinspo serve [--transport stdio|http]`
- [ ] Smoke test end-to-end

### Phase 6: MCP Server
- [ ] `server.py` — FastMCP w/ lifespan (shared GitHubClient)
- [ ] 5 tools: `analyze_repo`, `find_similar_repos`, `extract_features`, `compare_repos`, `scout_ideas`
- [ ] All logging to stderr (never stdout in MCP mode)
- [ ] Test w/ `fastmcp dev`

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
| `find_similar_repos` | `repo_url, n` | list[RepoMetadata] | Keyword search + embedding rerank |
| `extract_features` | `repo_url, target_context?` | list[ExtractedFeature] | Portable features w/ portability scores 1-10 |
| `compare_repos` | `repo_a_url, repo_b_url` | RepoComparison | Common patterns, unique features, recommendation |
| `scout_ideas` | `repo_url, n_similar, target_context?` | ScoutResult | Full pipeline → prioritized portable ideas |

## Key Architectural Decisions

- **JSON via instruction, not function-calling** — litellm unified API can't rely on provider-specific `tools=` consistently. Instruct model to return JSON, parse w/ Pydantic.
- **Hybrid search** — GitHub REST keyword search (fast, up to 1000 results) then embed + FAISS cosine rerank top-100. Cheaper and more reliable than pure embedding search.
- **Single GitHubClient via MCP lifespan** — one httpx.AsyncClient at startup, reused by all tool calls. Avoids TCP overhead, respects connection pools.
- **Two-pass analysis** — shallow scan (README + file tree) for discovery, deep scan (full content via gitingest) only for top-N candidates. Controls cost.

## Verification

1. `pip install -e ".[dev]" && pytest` — unit tests w/ mocked litellm + httpx
2. `repoinspo run https://github.com/tiangolo/fastapi --context "building a web framework" -n 3` — end-to-end CLI
3. `fastmcp dev src/repoinspo/server.py` — MCP Inspector smoke test
4. Claude Desktop config: add `repoinspo serve` as MCP server, test `scout_ideas` tool call
5. `pytest -m integration` — real API calls (requires GITHUB_TOKEN + ANTHROPIC_API_KEY)

## Resolved Questions

1. **BigQuery co-stargazer** — **Deferred entirely.** REST search + embedding rerank is enough. No GCP dependency. Revisit post-MVP only if keyword search proves too noisy.
2. **Embedding model portability** — **Graceful degradation.** Runtime check for embedding support; skip vector reranking and use pure keyword search when unavailable (e.g., Ollama without embedding model).
3. **Persistent vs ephemeral FAISS index** — **Ephemeral.** Built per-run, discarded after. No eviction logic, no stale data. Revisit if repeated usage pattern emerges.
4. **Repo name** — **repoinspo**
