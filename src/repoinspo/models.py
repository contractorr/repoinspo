"""Core Pydantic models used across the project."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RepoMetadata(BaseModel):
    """Metadata describing a GitHub repository."""

    model_config = ConfigDict(extra="forbid")

    full_name: str
    name: str
    owner: str
    html_url: HttpUrl | str
    description: str | None = None
    stars: int = Field(default=0, ge=0)
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    archived: bool = False
    license: str | None = None
    created_at: datetime | None = None
    pushed_at: datetime | None = None
    default_branch: str | None = None


class IngestedRepo(BaseModel):
    """Repository content prepared for LLM analysis."""

    model_config = ConfigDict(extra="forbid")

    metadata: RepoMetadata
    source_url: HttpUrl | str
    readme: str | None = None
    file_tree: str = ""
    content: str = ""
    token_estimate: int = Field(default=0, ge=0)
    truncated: bool = False
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RepoAnalysis(BaseModel):
    """Structured analysis of a repository."""

    model_config = ConfigDict(extra="forbid")

    repo: RepoMetadata
    purpose: str
    architecture: str
    features: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    notable_patterns: list[str] = Field(default_factory=list)
    summary: str | None = None


class ExtractedFeature(BaseModel):
    """A feature extracted from an analyzed repository."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    portability_score: int = Field(ge=1, le=10)
    rationale: str
    implementation_notes: str | None = None
    source_files: list[str] = Field(default_factory=list)


class FeatureExtractionResult(BaseModel):
    """A set of extracted features for a repository."""

    model_config = ConfigDict(extra="forbid")

    repo: RepoMetadata
    target_context: str | None = None
    features: list[ExtractedFeature] = Field(default_factory=list)


class PortableIdea(BaseModel):
    """A prioritized idea worth reusing in another project."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    priority_score: int = Field(ge=1, le=10)
    rationale: str
    source_repo: str
    related_features: list[str] = Field(default_factory=list)


class RepoComparison(BaseModel):
    """Comparison between two repositories."""

    model_config = ConfigDict(extra="forbid")

    repo_a: RepoMetadata
    repo_b: RepoMetadata
    common_patterns: list[str] = Field(default_factory=list)
    unique_to_a: list[str] = Field(default_factory=list)
    unique_to_b: list[str] = Field(default_factory=list)
    recommendation: str


class SearchFilters(BaseModel):
    """Optional filters mapped to GitHub search qualifiers."""

    model_config = ConfigDict(extra="forbid")

    created_after: datetime | None = None
    created_before: datetime | None = None
    pushed_after: datetime | None = None
    min_stars: int | None = Field(default=None, ge=0)
    max_stars: int | None = Field(default=None, ge=0)
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    archived: bool | None = None
    license: str | None = None


class TokenBudget(BaseModel):
    """Track token usage across multiple LLM calls."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_tokens_per_run: int = Field(ge=1)
    tokens_used: int = Field(default=0, ge=0)

    @property
    def remaining_tokens(self) -> int:
        return max(self.max_tokens_per_run - self.tokens_used, 0)

    @property
    def exhausted(self) -> bool:
        return self.tokens_used >= self.max_tokens_per_run

    def record_usage(self, total_tokens: int | None) -> int:
        if total_tokens is None:
            return self.tokens_used
        self.tokens_used += max(total_tokens, 0)
        return self.tokens_used


class ScoutResult(BaseModel):
    """The final output of the repo scouting pipeline."""

    model_config = ConfigDict(extra="forbid")

    seed_repo: RepoMetadata
    seed_analysis: RepoAnalysis
    similar_repos: list[RepoMetadata] = Field(default_factory=list)
    feature_reports: list[FeatureExtractionResult] = Field(default_factory=list)
    prioritized_ideas: list[PortableIdea] = Field(default_factory=list)
    comparisons: list[RepoComparison] = Field(default_factory=list)
    budget: TokenBudget
    partial: bool = False
    notes: list[str] = Field(default_factory=list)
