"""Multi-model council orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from litellm import acompletion

from repoinspo.models import TokenBudget

logger = logging.getLogger(__name__)


async def council_query(
    prompt: str,
    messages: list[dict[str, str]],
    models: list[str],
    budget: TokenBudget,
    completion_func: Any = acompletion,
) -> tuple[list[dict[str, str]], list[str]]:
    """Fan out one prompt across multiple models and keep successful responses."""

    if not models:
        raise ValueError("Council query requires at least one model")
    if budget.exhausted:
        raise RuntimeError("Token budget exhausted before council query")

    per_model_budget = max(budget.remaining_tokens // (len(models) + 1), 1)
    full_messages = [{"role": "system", "content": prompt}, *messages]
    tasks = [
        completion_func(model=model_name, messages=full_messages, max_tokens=per_model_budget)
        for model_name in models
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    responses: list[dict[str, str]] = []
    failures: list[str] = []
    for model_name, result in zip(models, raw_results, strict=True):
        if isinstance(result, Exception):
            failures.append(model_name)
            logger.warning("Council model %s failed", model_name, exc_info=result)
            continue
        budget.record_usage(_extract_total_tokens(result))
        responses.append({"model": model_name, "content": _extract_response_text(result)})

    if not responses:
        raise RuntimeError("All council models failed")
    return responses, failures


async def synthesize(
    prompt: str,
    responses: list[dict[str, str]],
    synthesizer_model: str,
    budget: TokenBudget,
    model_count: int | None = None,
    completion_func: Any = acompletion,
) -> str:
    """Synthesize multiple model outputs into one unified response."""

    if budget.exhausted:
        raise RuntimeError("Token budget exhausted before synthesis")
    total_models = model_count or len(responses)
    synthesis_budget = max(budget.max_tokens_per_run // (total_models + 1), 1)
    synthesis_prompt = (
        f"{prompt}\n\n"
        "You are synthesizing multiple model outputs for the same task. "
        "Find consensus, keep disagreements only when they materially affect the answer, "
        "and return one unified JSON object with no markdown fences."
    )
    response = await completion_func(
        model=synthesizer_model,
        messages=[
            {"role": "system", "content": synthesis_prompt},
            {
                "role": "user",
                "content": json.dumps({"responses": responses}, default=str),
            },
        ],
        max_tokens=synthesis_budget,
    )
    budget.record_usage(_extract_total_tokens(response))
    return _extract_response_text(response)


def _extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        content = response["choices"][0]["message"]["content"]
    else:
        content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_total_tokens(response: Any) -> int | None:
    if isinstance(response, dict):
        usage = response.get("usage") or {}
        return usage.get("total_tokens")
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is not None:
        return total_tokens
    if isinstance(usage, dict):
        return usage.get("total_tokens")
    return None
