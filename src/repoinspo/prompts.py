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
  "summary": "string"
}
Use the repository summary, tree, and file contents to infer architecture
and implementation choices.
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
      "related_features": ["string"]
    }
  ]
}
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
      "rationale": "string",
      "source_repo": "string",
      "related_features": ["string"]
    }
  ]
}
Rank ideas by reuse value, implementation leverage, and fit for the target context.
""".strip()
