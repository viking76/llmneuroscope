import json

import pytest

from app.services.ingestion import IngestionError, parse_content


def _jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows).encode()


def test_parse_jsonl_basic():
    content = _jsonl(
        [
            {"model": "m1", "prompt_id": "p1", "run": 1, "tokens": ["a", "b"], "logprobs": [-0.1, -0.2], "latency_ms": 100},
            {"model": "m1", "prompt_id": "p2", "run": 2, "tokens": ["a", "c"], "logprobs": [-0.3, -0.4], "latency_ms": 120},
        ]
    )
    records = parse_content("runs.jsonl", content)
    assert len(records) == 2
    assert records[0]["model"] == "m1"
    assert records[0]["run"] == 1
    assert records[0]["tokens"] == ["a", "b"]
    assert records[0]["logprobs"] == [-0.1, -0.2]


def test_parse_json_list_wrapper():
    content = json.dumps({"records": [{"model": "m1", "output": "hello world"}]}).encode()
    records = parse_content("runs.json", content)
    assert len(records) == 1
    assert records[0]["tokens"] == ["hello", "world"]
    assert records[0]["run"] == 1
    assert records[0]["prompt_id"] == "0"


def test_parse_csv_with_aliases():
    content = b"model_name,seed,latency,output_text\nm1,2,150,hello world\n"
    records = parse_content("runs.csv", content)
    assert len(records) == 1
    assert records[0]["model"] == "m1"
    assert records[0]["run"] == 2
    assert records[0]["latency_ms"] == 150.0
    assert records[0]["tokens"] == ["hello", "world"]


def test_unsupported_extension():
    with pytest.raises(IngestionError):
        parse_content("runs.txt", b"model\nm1\n")


def test_empty_file():
    with pytest.raises(IngestionError):
        parse_content("runs.jsonl", b"")


def test_invalid_jsonl_line():
    with pytest.raises(IngestionError, match="Ligne 2"):
        parse_content("runs.jsonl", b'{"model": "m1"}\nnot json\n')
