from __future__ import annotations

from app.services.metrics import js_divergence, token_distribution

REGRESSION_THRESHOLDS = {
    "perplexity": 0.10,
    "repetition.degeneration_rate": 0.05,
    "latency_ms.mean": 0.20,
    "token_entropy": -0.10,
}


def _get_nested(data: dict, path: str):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or current.get(part) is None:
            return None
        current = current[part]
    return current


def compare_models(analysis_a: dict, analysis_b: dict) -> dict:
    model_a, model_b = analysis_a["model"], analysis_b["model"]
    records_a = analysis_a.get("_records", [])
    records_b = analysis_b.get("_records", [])
    divergence = None
    if records_a and records_b:
        divergence = round(
            js_divergence(token_distribution(records_a), token_distribution(records_b)), 4
        )

    rows = []
    metric_paths = [
        ("perplexity", "Perplexité", "lower"),
        ("token_entropy", "Entropie des tokens", "context"),
        ("vocabulary_size", "Taille du vocabulaire", "context"),
        ("logprob.mean", "Log-prob moyenne", "higher"),
        ("length.mean", "Longueur moyenne", "context"),
        ("length.cv", "Variabilité de longueur", "lower"),
        ("latency_ms.mean", "Latence moyenne (ms)", "lower"),
        ("latency_ms.p95", "Latence p95 (ms)", "lower"),
        ("repetition.degeneration_rate", "Taux de dégénérescence", "lower"),
        ("repetition.mean_ratio", "Ratio de répétition", "lower"),
        ("drift.max_js", "Dérive max entre runs", "lower"),
    ]
    for path, label, direction in metric_paths:
        value_a = _get_nested(analysis_a, path)
        value_b = _get_nested(analysis_b, path)
        if value_a is None or value_b is None:
            continue
        delta = round(value_b - value_a, 4)
        pct = round(delta / abs(value_a) * 100, 2) if value_a else None
        status = "stable"
        if direction == "lower" and pct is not None and pct > REGRESSION_THRESHOLDS.get(path, 10) * 100:
            status = "regression"
        elif direction == "higher" and pct is not None and pct < REGRESSION_THRESHOLDS.get(path, -10) * 100:
            status = "regression"
        elif direction == "lower" and pct is not None and pct < -15:
            status = "improvement"
        elif direction == "higher" and pct is not None and pct > 15:
            status = "improvement"
        rows.append(
            {
                "metric": label,
                "key": path,
                "model_a": value_a,
                "model_b": value_b,
                "delta": delta,
                "delta_pct": pct,
                "status": status,
            }
        )

    regressions = [row["metric"] for row in rows if row["status"] == "regression"]
    improvements = [row["metric"] for row in rows if row["status"] == "improvement"]
    verdict = "stable"
    if len(regressions) >= 2:
        verdict = "regression"
    elif len(regressions) == 1 and len(improvements) == 0:
        verdict = "warning"
    elif len(improvements) > len(regressions):
        verdict = "improvement"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "token_distribution_js": divergence,
        "metrics": rows,
        "regressions": regressions,
        "improvements": improvements,
        "verdict": verdict,
    }
