from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

_PII_PATTERNS = (
    "@",  # crude email marker
    "ssn:",
)

def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Raises a ValueError carrying the offending line number for malformed JSON,
    schema violations, or duplicate prompts. Flags obvious PII markers as a warning.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON ({exc.msg})") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"Line {line_number}: schema validation failed ({exc})") from exc

            normalized_prompt = " ".join(example.prompt.split()).casefold()
            if normalized_prompt in seen_prompts:
                raise ValueError(
                    f"Line {line_number}: duplicate prompt (first seen on line "
                    f"{seen_prompts[normalized_prompt]})"
                )
            seen_prompts[normalized_prompt] = line_number

            lowered = f"{example.prompt} {example.chosen} {example.rejected}".lower()
            if any(marker in lowered for marker in _PII_PATTERNS):
                print(f"[yellow]Warning: possible PII marker on line {line_number}[/yellow]")

            examples.append(example)
    return examples

def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Groups examples by (normalized) prompt so the same prompt never appears in both
    splits, then deterministically shuffles the groups using `seed` before cutting.
    """
    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        key = " ".join(example.prompt.split()).casefold()
        groups.setdefault(key, []).append(example)

    prompt_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(prompt_keys)

    val_count = max(1, round(len(prompt_keys) * validation_ratio)) if prompt_keys else 0
    val_keys = set(prompt_keys[:val_count])

    train: list[PreferenceExample] = []
    val: list[PreferenceExample] = []
    for key in prompt_keys:
        target = val if key in val_keys else train
        target.extend(groups[key])
    return train, val
