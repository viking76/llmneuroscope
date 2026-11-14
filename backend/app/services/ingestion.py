from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson"}
MAX_POSITIONS = 128

FIELD_ALIASES = {
    "model": ["model", "model_name", "model_id", "model_version"],
    "prompt_id": ["prompt_id", "prompt", "sample_id", "sample_index", "example_id", "case_id"],
    "run": ["run", "run_id", "seed", "iteration", "trial", "attempt"],
    "tokens": ["tokens", "token_ids", "output_tokens", "generated_tokens"],
    "logprobs": ["logprobs", "token_logprobs", "log_probs", "logprobs_list"],
    "output": ["output", "output_text", "response", "completion", "text", "answer", "generation"],
    "latency_ms": ["latency_ms", "latency", "duration_ms", "time_ms", "elapsed_ms"],
    "timestamp": ["timestamp", "created_at", "datetime", "date", "time"],
}


class IngestionError(ValueError):
    pass


def _build_column_map(columns):
    lowered = {c: str(c).strip().lower() for c in columns}
    mapping = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            for original, low in lowered.items():
                if low == alias:
                    mapping[canonical] = original
                    break
            if canonical in mapping:
                break
    return mapping


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value, default=1):
    result = _coerce_float(value)
    if result is None:
        return default
    return max(1, int(result))


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return text.split()
    return [str(value)]


def _normalize_row(raw: dict, row_index: int) -> dict | None:
    mapping = _build_column_map(raw.keys())
    model = raw.get(mapping.get("model", ""), None)
    tokens = _coerce_list(raw.get(mapping.get("tokens", "")))
    output = raw.get(mapping.get("output", ""))
    if model is None and not tokens and (output is None or str(output).strip() == ""):
        return None
    if not tokens and output:
        tokens = str(output).split()
    logprobs_raw = raw.get(mapping.get("logprobs", ""))
    logprobs = []
    if logprobs_raw is not None and str(logprobs_raw).strip() != "":
        if isinstance(logprobs_raw, list):
            logprobs = [_coerce_float(v) for v in logprobs_raw]
        else:
            logprobs = [_coerce_float(v) for v in _coerce_list(logprobs_raw)]
        logprobs = [v for v in logprobs if v is not None]
    tokens = tokens[:MAX_POSITIONS]
    logprobs = logprobs[:MAX_POSITIONS]
    return {
        "model": str(model) if model not in (None, "") else "unknown",
        "prompt_id": str(raw.get(mapping.get("prompt_id", ""), row_index)),
        "run": _coerce_int(raw.get(mapping.get("run", ""))),
        "tokens": tokens,
        "logprobs": logprobs,
        "output": str(output) if output is not None else "",
        "latency_ms": _coerce_float(raw.get(mapping.get("latency_ms", ""))),
        "timestamp": str(raw.get(mapping.get("timestamp", ""), "")),
    }


def parse_content(filename: str, content: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Format non supporté: {suffix}. Formats acceptés: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    text = content.decode("utf-8-sig", errors="replace")
    if suffix == ".csv":
        rows = _parse_csv(text)
    elif suffix == ".json":
        rows = _parse_json(text)
    else:
        rows = _parse_jsonl(text)
    records = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_row(raw, index)
        if normalized is not None:
            records.append(normalized)
    if not records:
        raise IngestionError("Aucun enregistrement exploitable trouvé dans le fichier.")
    return records


def _parse_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _parse_json(text: str) -> list[dict]:
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("records", "data", "results", "runs", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise IngestionError("JSON invalide: attendu un objet ou une liste d'enregistrements.")


def _parse_jsonl(text: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IngestionError(f"Ligne {line_number}: JSON invalide ({exc})") from exc
    return rows


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)
