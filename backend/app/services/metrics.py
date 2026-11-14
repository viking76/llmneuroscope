from __future__ import annotations

import math
from collections import Counter

import numpy as np

REPETITION_RATIO_THRESHOLD = 0.5
ANOMALY_Z_THRESHOLD = 3.0
MAX_HEATMAP_POSITIONS = 64


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def token_distribution(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        counter.update(record.get("tokens") or [])
    return counter


def js_divergence(counter_a: Counter, counter_b: Counter) -> float:
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    total_a = sum(counter_a.values()) or 1
    total_b = sum(counter_b.values()) or 1
    divergence = 0.0
    for key in keys:
        p = counter_a.get(key, 0) / total_a
        q = counter_b.get(key, 0) / total_b
        m = (p + q) / 2
        if p > 0 and m > 0:
            divergence += 0.5 * p * math.log2(p / m)
        if q > 0 and m > 0:
            divergence += 0.5 * q * math.log2(q / m)
    return divergence


def perplexity(logprobs: list[float]) -> float | None:
    if not logprobs:
        return None
    mean = float(np.mean(logprobs))
    return float(math.exp(-mean))


def repetition_ratio(tokens: list[str], n: int = 3) -> float:
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    top_count = counter.most_common(1)[0][1]
    return top_count / len(ngrams)


def _stats(values: list[float]) -> dict | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    array = np.array(clean, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _record_logprob_mean(record: dict) -> float | None:
    logprobs = record.get("logprobs") or []
    if not logprobs:
        return None
    return float(np.mean(logprobs))


def _position_entropy(records: list[dict]) -> list[float]:
    per_position = []
    for position in range(MAX_HEATMAP_POSITIONS):
        counter = Counter()
        for record in records:
            tokens = record.get("tokens") or []
            if position < len(tokens):
                counter[tokens[position]] += 1
        if not counter:
            break
        per_position.append(shannon_entropy(counter))
    return per_position


def _position_logprob(records: list[dict]) -> list[float]:
    per_position = []
    for position in range(MAX_HEATMAP_POSITIONS):
        values = []
        for record in records:
            logprobs = record.get("logprobs") or []
            if position < len(logprobs):
                values.append(logprobs[position])
        if not values:
            break
        per_position.append(float(np.mean(values)))
    return per_position


def heatmap(records: list[dict]) -> dict:
    runs = sorted({record["run"] for record in records})
    positions = range(MAX_HEATMAP_POSITIONS)
    has_data = any(record.get("tokens") for record in records)
    if not has_data:
        return {"runs": [], "positions": [], "matrix": []}
    matrix = []
    for run in runs:
        run_records = [r for r in records if r["run"] == run]
        row = []
        for position in positions:
            counter = Counter()
            for record in run_records:
                tokens = record.get("tokens") or []
                if position < len(tokens):
                    counter[tokens[position]] += 1
            row.append(round(shannon_entropy(counter), 4) if counter else None)
        if all(v is None for v in row):
            break
        matrix.append({"run": run, "values": row})
    used_positions = len(matrix[0]["values"]) if matrix else 0
    return {
        "runs": [row["run"] for row in matrix],
        "positions": list(range(used_positions)),
        "matrix": [
            {"run": row["run"], "values": row["values"][:used_positions]} for row in matrix
        ],
    }


def analyze_model(records: list[dict], model: str) -> dict:
    model_records = [r for r in records if r["model"] == model]
    runs = sorted({r["run"] for r in model_records})
    distribution = token_distribution(model_records)
    logprob_means = [_record_logprob_mean(r) for r in model_records]
    lengths = [float(len(r.get("tokens") or [])) for r in model_records]
    latencies = [r.get("latency_ms") for r in model_records]
    repetition_scores = [repetition_ratio(r.get("tokens") or []) for r in model_records]
    degenerate = [s for s in repetition_scores if s > REPETITION_RATIO_THRESHOLD]

    run_entropies = {}
    for run in runs:
        run_records = [r for r in model_records if r["run"] == run]
        run_entropies[str(run)] = round(shannon_entropy(token_distribution(run_records)), 4)

    drift = {"max_js": 0.0, "from_run": None, "to_run": None}
    if len(runs) > 1:
        baseline = token_distribution([r for r in model_records if r["run"] == runs[0]])
        for run in runs[1:]:
            current = token_distribution([r for r in model_records if r["run"] == run])
            score = js_divergence(baseline, current)
            if score > drift["max_js"]:
                drift = {"max_js": round(score, 4), "from_run": runs[0], "to_run": run}

    perplexity_value = perplexity(
        [lp for r in model_records for lp in (r.get("logprobs") or [])]
    )

    return {
        "model": model,
        "n_records": len(model_records),
        "n_runs": len(runs),
        "token_entropy": round(shannon_entropy(distribution), 4),
        "vocabulary_size": len(distribution),
        "perplexity": round(perplexity_value, 4) if perplexity_value else None,
        "logprob": _round_stats(_stats(logprob_means)),
        "length": _round_stats(_stats(lengths)),
        "latency_ms": _round_stats(_stats(latencies)),
        "repetition": {
            "mean_ratio": round(float(np.mean(repetition_scores)), 4) if repetition_scores else 0.0,
            "max_ratio": round(max(repetition_scores), 4) if repetition_scores else 0.0,
            "degeneration_rate": round(len(degenerate) / len(model_records), 4)
            if model_records
            else 0.0,
        },
        "run_entropy": run_entropies,
        "drift": drift,
        "position_entropy": [round(v, 4) for v in _position_entropy(model_records)],
        "position_logprob": [round(v, 4) for v in _position_logprob(model_records)],
    }


def _round_stats(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    return {key: round(value, 4) for key, value in stats.items()}


def _position_spread(records: list[dict]) -> list[float]:
    per_position = []
    for position in range(MAX_HEATMAP_POSITIONS):
        values = []
        for record in records:
            logprobs = record.get("logprobs") or []
            if position < len(logprobs):
                values.append(logprobs[position])
        if len(values) < 2:
            break
        per_position.append(float(np.std(values)))
    return per_position


def detect_blocking_points(records: list[dict], model: str) -> list[dict]:
    model_records = [r for r in records if r["model"] == model]
    position_logprob = _position_logprob(model_records)
    position_entropy = _position_entropy(model_records)
    if not position_logprob:
        return []
    array = np.array(position_logprob)
    mean, std = float(array.mean()), float(array.std())
    points = []
    for position, value in enumerate(position_logprob):
        z = (value - mean) / std if std > 0 else 0.0
        if z < -1.2:
            points.append(
                {
                    "position": position,
                    "type": "logprob_collapse",
                    "value": round(value, 4),
                    "z_score": round(z, 2),
                    "severity": "high" if z < -2.0 else "medium",
                }
            )
    spread = _position_spread(model_records)
    if spread:
        spread_array = np.array(spread)
        median_spread = float(np.median(spread_array))
        for position, value in enumerate(spread):
            if median_spread > 0 and value > 4 * median_spread:
                points.append(
                    {
                        "position": position,
                        "type": "position_instability",
                        "value": round(value, 4),
                        "z_score": round(value / median_spread, 2),
                        "severity": "high" if value > 8 * median_spread else "medium",
                    }
                )
    entropy_array = np.array(position_entropy)
    if len(entropy_array) > 1:
        e_mean, e_std = float(entropy_array.mean()), float(entropy_array.std())
        for position, value in enumerate(position_entropy):
            z = (value - e_mean) / e_std if e_std > 0 else 0.0
            if z > 2.0:
                points.append(
                    {
                        "position": position,
                        "type": "entropy_spike",
                        "value": round(value, 4),
                        "z_score": round(z, 2),
                        "severity": "high" if z > 2.5 else "medium",
                    }
                )
    severity_order = {"high": 0, "medium": 1, "low": 2}
    unique = {}
    for point in points:
        key = (point["position"], point["type"])
        if key not in unique or severity_order[point["severity"]] < severity_order[unique[key]["severity"]]:
            unique[key] = point
    return sorted(unique.values(), key=lambda p: severity_order[p["severity"]])


def detect_anomalies(records: list[dict], model: str) -> list[dict]:
    model_records = [r for r in records if r["model"] == model]
    anomalies = []

    logprob_means = [_record_logprob_mean(r) for r in model_records]
    if logprob_means:
        array = np.array(logprob_means)
        mean, std = float(array.mean()), float(array.std())
        if std > 0:
            for record, value in zip(model_records, logprob_means):
                z = (value - mean) / std
                if z < -ANOMALY_Z_THRESHOLD:
                    anomalies.append(
                        {
                            "type": "logprob_collapse",
                            "severity": "high" if z < -4 else "medium",
                            "model": model,
                            "prompt_id": record["prompt_id"],
                            "run": record["run"],
                            "value": round(value, 4),
                            "z_score": round(z, 2),
                        }
                    )

    lengths = [float(len(r.get("tokens") or [])) for r in model_records]
    if lengths:
        array = np.array(lengths)
        mean, std = float(array.mean()), float(array.std())
        if std > 0:
            for record, value in zip(model_records, lengths):
                z = (value - mean) / std
                if abs(z) > ANOMALY_Z_THRESHOLD:
                    anomalies.append(
                        {
                            "type": "length_outlier",
                            "severity": "medium",
                            "model": model,
                            "prompt_id": record["prompt_id"],
                            "run": record["run"],
                            "value": value,
                            "z_score": round(z, 2),
                        }
                    )

    latencies = [(r, r.get("latency_ms")) for r in model_records if r.get("latency_ms")]
    if latencies:
        values = [v for _, v in latencies]
        array = np.array(values)
        mean, std = float(array.mean()), float(array.std())
        if std > 0:
            for record, value in latencies:
                z = (value - mean) / std
                if z > ANOMALY_Z_THRESHOLD:
                    anomalies.append(
                        {
                            "type": "latency_spike",
                            "severity": "low" if z < 4 else "medium",
                            "model": model,
                            "prompt_id": record["prompt_id"],
                            "run": record["run"],
                            "value": value,
                            "z_score": round(z, 2),
                        }
                    )

    for record in model_records:
        ratio = repetition_ratio(record.get("tokens") or [])
        if ratio > REPETITION_RATIO_THRESHOLD:
            anomalies.append(
                {
                    "type": "degenerate_repetition",
                    "severity": "high" if ratio > 0.8 else "medium",
                    "model": model,
                    "prompt_id": record["prompt_id"],
                    "run": record["run"],
                    "value": round(ratio, 4),
                    "z_score": None,
                }
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(anomalies, key=lambda a: severity_order[a["severity"]])
