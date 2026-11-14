import math
from collections import Counter

from app.services.metrics import (
    analyze_model,
    detect_anomalies,
    js_divergence,
    perplexity,
    repetition_ratio,
    shannon_entropy,
)


def _record(model="m1", run=1, tokens=None, logprobs=None, latency_ms=100, prompt_id="p1"):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "run": run,
        "tokens": tokens or [],
        "logprobs": logprobs or [],
        "output": "",
        "latency_ms": latency_ms,
        "timestamp": "",
    }


def test_shannon_entropy_uniform():
    assert shannon_entropy(Counter({"a": 1, "b": 1})) == 1.0
    assert shannon_entropy(Counter({"a": 4})) == 0.0


def test_shannon_entropy_known_value():
    entropy = shannon_entropy(Counter({"a": 2, "b": 2, "c": 4}))
    expected = -(0.25 * math.log2(0.25) * 2 + 0.5 * math.log2(0.5))
    assert math.isclose(entropy, expected, rel_tol=1e-9)


def test_js_divergence_identical_is_zero():
    counter = Counter({"a": 5, "b": 5})
    assert js_divergence(counter, counter) == 0.0


def test_js_divergence_disjoint_is_one():
    assert js_divergence(Counter({"a": 1}), Counter({"b": 1})) == 1.0


def test_perplexity():
    assert math.isclose(perplexity([0.0]), 1.0)
    assert math.isclose(perplexity([-math.log(2), -math.log(2)]), 2.0)


def test_repetition_ratio_detects_loop():
    looping = ["the"] * 10
    assert repetition_ratio(looping) > 0.9
    varied = ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert repetition_ratio(varied) < 0.2


def test_analyze_model_basic():
    records = [
        _record(run=1, tokens=["a", "b", "c"], logprobs=[-0.1, -0.2, -0.3]),
        _record(run=2, tokens=["a", "b", "d"], logprobs=[-0.1, -0.2, -0.5]),
    ]
    analysis = analyze_model(records, "m1")
    assert analysis["model"] == "m1"
    assert analysis["n_records"] == 2
    assert analysis["n_runs"] == 2
    assert analysis["perplexity"] is not None
    assert len(analysis["position_entropy"]) == 3
    assert set(analysis["run_entropy"]) == {"1", "2"}


def test_analyze_model_drift_detected():
    records = [_record(run=1, tokens=["a", "a", "a"]), _record(run=2, tokens=["b", "b", "b"])]
    analysis = analyze_model(records, "m1")
    assert analysis["drift"]["max_js"] == 1.0
    assert analysis["drift"]["to_run"] == 2


def test_detect_anomalies_repetition():
    records = [
        _record(prompt_id="ok", tokens=["a", "b", "c", "d", "e", "f"], logprobs=[-0.5] * 6),
        _record(prompt_id="loop", tokens=["x"] * 20, logprobs=[-0.5] * 20),
    ]
    anomalies = detect_anomalies(records, "m1")
    types = {a["type"] for a in anomalies}
    assert "degenerate_repetition" in types
    loop_anomaly = next(a for a in anomalies if a["prompt_id"] == "loop")
    assert loop_anomaly["severity"] == "high"
