"""Prompt templates used by the LLM analysis layer."""

ANALYZE_REPO_SHORT = """
You analyze software repositories.
Return JSON only, without markdown fences.
Schema:
{
  "purpose": "string",
  "architecture": "string",
  "features": ["string"],
  "tech_stack": ["string"],
  "notable_patterns": ["string"],
  "summary": "string"
}
Focus on the clearest high-signal findings only.
""".strip()

ANALYZE_REPO_FULL = """
You analyze software repositories in detail.
Return JSON only, without markdown fences.
Schema:
{
  "purpose": "string",
  "architecture": "string",
  "features": ["string"],
  "tech_stack": ["string"],
  "notable_patterns": ["string"],
  "summary": "string",
  "strengths": ["string — what the repo does well architecturally or functionally"],
  "weaknesses": ["string — gaps, tech debt, missing capabilities"],
  "opportunities": ["string — short concepts suitable for cross-domain search, e.g. 'behavioral learning loop', 'event-driven scraping'"]
}
Use the repository summary, tree, and file contents to infer architecture
and implementation choices. For each field:
- strengths: identify 2-4 things the repo does well (architecture, DX, patterns)
- weaknesses: identify 2-4 gaps, limitations, or areas of tech debt
- opportunities: list 3-5 short conceptual phrases describing capabilities this
  project could gain from cross-domain inspiration. These should be abstract enough
  to search for in unrelated domains (e.g. "progressive disclosure UI",
  "self-healing pipeline", "federated plugin architecture").
""".strip()

EXTRACT_FEATURES_SHORT = """
You extract portable features from a repository.
Return JSON only, without markdown fences.
Schema:
{
  "features": [
    {
      "name": "string",
      "description": "string",
      "portability_score": 1,
      "rationale": "string",
      "implementation_notes": "string",
      "source_files": ["string"]
    }
  ]
}
Prefer fewer, higher-value features when context is limited.
""".strip()

EXTRACT_FEATURES_FULL = """
You extract portable features from a repository for reuse in another project.
Return JSON only, without markdown fences.
Schema:
{
  "features": [
    {
      "name": "string",
      "description": "string",
      "portability_score": 1,
      "rationale": "string",
      "implementation_notes": "string",
      "source_files": ["string"]
    }
  ]
}
Explain why each feature is portable and include likely implementation anchors.
""".strip()

COMPARE_REPOS_SHORT = """
You compare two repositories.
Return JSON only, without markdown fences.
Schema:
{
  "common_patterns": ["string"],
  "unique_to_a": ["string"],
  "unique_to_b": ["string"],
  "recommendation": "string"
}
Keep the comparison concise and decision-oriented.
""".strip()

COMPARE_REPOS_FULL = """
You compare two repositories in detail.
Return JSON only, without markdown fences.
Schema:
{
  "common_patterns": ["string"],
  "unique_to_a": ["string"],
  "unique_to_b": ["string"],
  "recommendation": "string"
}
Focus on architectural overlap, meaningful divergences, and which ideas transfer cleanly.
""".strip()

PRIORITIZE_IDEAS_SHORT = """
You prioritize portable ideas for reuse.
Return JSON only, without markdown fences.
Schema:
{
  "prioritized_ideas": [
    {
      "title": "string",
      "description": "string",
      "priority_score": 1,
      "rationale": "string",
      "source_repo": "string",
      "related_features": ["string"],
      "implementation_complexity": "low|medium|high — brief explanation",
      "expected_impact": "what adopting this would unlock",
      "adaptation_notes": "how to adapt for the target repo"
    }
  ]
}
Consolidate near-duplicate ideas across repos.
""".strip()

PRIORITIZE_IDEAS_FULL = """
You prioritize portable implementation ideas across repositories.
Return JSON only, without markdown fences.
Schema:
{
  "prioritized_ideas": [
    {
      "title": "string",
      "description": "string",
      "priority_score": 1,
      "rationale": "string — 2-3 sentences minimum explaining WHY this is valuable",
      "source_repo": "string",
      "related_features": ["string"],
      "implementation_complexity": "low|medium|high — brief explanation of effort required",
      "expected_impact": "what adopting this would unlock for the target project",
      "adaptation_notes": "specific guidance on how to adapt this for the target repo"
    }
  ]
}
Score each idea on these axes:
- Reuse leverage: how much value is gained vs effort to integrate
- Implementation effort: complexity relative to a solo developer
- Strategic fit: alignment with the target project's goals and architecture

Consolidate near-duplicate ideas that appear across multiple repos into a single
entry, crediting all source repos in the rationale.
Write detailed rationale (2-3 sentences minimum) explaining the reasoning.
""".strip()

GENERATE_SEARCH_STRATEGIES = """
You generate diverse GitHub search strategies to find repos that could inspire
improvements to a target project. Generate two types:

1. DIRECT strategies: repos solving similar problems (same domain, similar tech)
2. LATERAL strategies: repos from DIFFERENT domains that implement transferable
   patterns. Think cross-domain: a music recommendation engine might inspire a
   news recommendation system; a game engine's ECS architecture might inspire
   a data pipeline; a fintech fraud detector might inspire anomaly detection
   in monitoring.

Return JSON only, without markdown fences.
Schema:
{
  "strategies": [
    {
      "query": "GitHub search query string (use keywords, NOT topic: qualifiers for lateral)",
      "strategy_type": "direct|lateral|conceptual",
      "rationale": "why this query finds repos with transferable ideas"
    }
  ]
}

Generate 4-6 strategies total: ~2 direct, ~2-3 lateral, ~1 conceptual.
For lateral queries, REMOVE language constraints — great ideas transcend languages.
For direct queries, keep language if relevant.
""".strip()
