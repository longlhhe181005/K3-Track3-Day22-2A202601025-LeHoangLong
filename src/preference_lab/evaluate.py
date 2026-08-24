from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import PreferenceExample

_WORD_RE = re.compile(r"[a-z0-9]+")

def deterministic_score(prompt: str, response: str) -> float:
    """CPU-only, model-free proxy score for how well a response addresses a prompt.

    Combines lexical overlap between prompt and response with a mild length
    prior (longer, more substantive answers score slightly higher). Deterministic
    and reproducible, used in place of real model logprobs when no trained
    policy is available.
    """
    prompt_words = set(_WORD_RE.findall(prompt.lower()))
    response_words = _WORD_RE.findall(response.lower())
    if not response_words:
        return 0.0
    overlap = len(prompt_words.intersection(response_words))
    overlap_ratio = overlap / max(1, len(prompt_words))
    length_score = min(len(response_words), 60) / 60
    return 0.7 * overlap_ratio + 0.3 * length_score

def pairwise_accuracy(examples: list[PreferenceExample], chosen_scores: list[float], rejected_scores: list[float]) -> float:
    """Return fraction where chosen score is greater than rejected score.

    Ties (equal scores) count as half a win. Raises if the score lists don't
    match the number of examples.
    """
    if not examples:
        return 0.0
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError(
            f"Expected {len(examples)} chosen/rejected scores, "
            f"got {len(chosen_scores)}/{len(rejected_scores)}"
        )
    wins = 0.0
    for c, r in zip(chosen_scores, rejected_scores, strict=True):
        if c > r:
            wins += 1.0
        elif c == r:
            wins += 0.5
    return wins / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
