import json
from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected

def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)

def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.3)
    train_prompts = {" ".join(ex.prompt.split()).casefold() for ex in train}
    val_prompts = {" ".join(ex.prompt.split()).casefold() for ex in val}
    assert train_prompts.isdisjoint(val_prompts)

def test_split_is_deterministic() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train1, val1 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    train2, val2 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    assert [ex.prompt for ex in train1] == [ex.prompt for ex in train2]
    assert [ex.prompt for ex in val1] == [ex.prompt for ex in val2]

def test_load_jsonl_reports_line_number_on_bad_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"prompt":"p1","chosen":"a","rejected":"b"}\n{not valid json}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Line 2"):
        load_jsonl(bad_file)

def test_load_jsonl_rejects_duplicate_prompts(tmp_path: Path) -> None:
    dup_file = tmp_path / "dup.jsonl"
    row = {"prompt": "same prompt", "chosen": "a", "rejected": "b"}
    dup_file.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt"):
        load_jsonl(dup_file)
