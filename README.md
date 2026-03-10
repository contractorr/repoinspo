# repoinspo

`repoinspo` analyzes a repository, finds similar GitHub projects, ingests their code, and prioritizes portable implementation ideas.

It ships as both:

- a CLI for one-off runs
- an MCP server for Claude Code, Claude Desktop, Cursor, and other MCP clients

## What it does

Given a seed repository, `repoinspo` will:

1. fetch repository metadata and README data from GitHub
2. ingest the repository contents with `gitingest`
3. analyze the codebase with `litellm`
4. search GitHub for similar repositories
5. rerank candidates with embeddings plus FAISS when embeddings are available
6. extract portable features from similar repos
7. prioritize the best ideas for reuse

The implementation is async across GitHub, ingestion, embeddings, and LLM calls.

## Stack

- Python 3.11+
- Typer + Rich for the CLI
- FastMCP via `mcp[cli]`
- `httpx` for GitHub REST and GraphQL
- `litellm` for completions and embeddings
- `gitingest` with `repomix` fallback
- SQLite README cache with ETag support
- FAISS for vector reranking

## Installation

```bash
pip install -e ".[dev]"
```

If you want the `repomix` fallback path, install `repomix` separately so it is available on `PATH`.

## Configuration

Copy `.env.example` to `.env` and fill in the values you need:

```env
GITHUB_TOKEN=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
LLM_MODELS=anthropic/claude-sonnet-4-20250514
EMBEDDING_MODEL=text-embedding-3-small
CACHE_DIR=.cache/repoinspo
MAX_TOKENS=8192
DEFAULT_TOKEN_BUDGET=100000
COUNCIL_ENABLED=true
```

Notes:

- `GITHUB_TOKEN` is strongly recommended for GitHub API limits.
- `LLM_MODELS` accepts a comma-separated list.
- One configured model means normal mode.
- Multiple configured models plus `COUNCIL_ENABLED=true` enable council mode.
- If embeddings are unavailable for the configured provider or model, `repoinspo` falls back to keyword-only search results.

## CLI

Show help:

```bash
repoinspo --help
```

Run the full pipeline:

```bash
repoinspo run https://github.com/tiangolo/fastapi
```

Run with context, output selection, and search filters:

```bash
repoinspo run https://github.com/tiangolo/fastapi \
  -n 3 \
  --context "building a web framework" \
  --output pretty \
  --budget 100000 \
  --created-after 2023-01-01 \
  --min-stars 50 \
  --language python \
  --topic api
```

Output modes:

- `pretty`: Rich tables
- `json`: structured machine-readable output
- `md`: markdown summary

## MCP server

Start the server over stdio:

```bash
repoinspo serve
```

Start the server over HTTP transport:

```bash
repoinspo serve --transport http
```

Available MCP tools:

- `analyze_repo`
- `find_similar_repos`
- `extract_features`
- `compare_repos`
- `scout_ideas`

Claude Code or Claude Desktop config:

```json
{
  "mcpServers": {
    "repoinspo": {
      "command": "repoinspo",
      "args": ["serve"]
    }
  }
}
```

## Behavior

- GitHub search uses REST search first, then embedding reranking when available.
- README fetches are cached in SQLite with ETag reuse.
- Ingestion uses `gitingest` first and falls back to `repomix --remote --compress` when available.
- LLM responses are requested as JSON and validated with Pydantic models.
- Token usage is tracked across analysis calls and the pipeline degrades work when the budget is low.
- In council mode, prompts fan out to multiple models and then synthesize into a single JSON result.

## Development

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

The current test suite covers:

- GitHub client behavior
- ingestion and truncation
- analysis parsing and council behavior
- pipeline orchestration

## Repository layout

```text
src/repoinspo/
  cli.py
  server.py
  config.py
  models.py
  prompts.py
  core/
    github.py
    ingestion.py
    analysis.py
    council.py
    embeddings.py
    pipeline.py
tests/
```
