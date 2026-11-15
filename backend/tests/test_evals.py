import os

os.environ["LLM_NEUROSCOPE_BACKEND"] = "mock"

import numpy as np
import pytest

from app.services.evals import (
    build_langfuse_traces,
    export_langfuse,
    mock_benchmark,
    quality_report,
)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_mock_benchmark_shape_and_determinism():
    first = mock_benchmark("mock-tiny-16d", ["lambada_openai", "hellaswag"], 5)
    second = mock_benchmark("mock-tiny-16d", ["lambada_openai", "hellaswag"], 5)
    assert first == second
    assert first["engine"] == "mock"
    assert set(first["tasks"]) == {"lambada_openai", "hellaswag"}
    for metrics in first["tasks"].values():
        assert 0.0 <= metrics["acc"] <= 1.0
        assert metrics["acc_stderr"] > 0.0


def test_quality_report_metrics():
    texts = ["le chat dort sur le tapis", "le chat dort sur le tapis"]
    logprobs = [[-0.5, -0.7, -0.4, -0.6, -0.5], [-0.5, -0.7, -0.4, -0.6, -0.5]]
    report = quality_report(texts, logprobs)
    assert report["n_texts"] == 2
    assert report["judge"] is None
    for entry in report["texts"]:
        assert entry["perplexity"] > 1.0
        assert 0.0 <= entry["distinct_ratio"] <= 1.0
        assert 0.0 <= entry["repetition_ratio"] <= 1.0
    assert report["aggregate"]["repetition_ratio"] == pytest.approx(
        report["texts"][0]["repetition_ratio"]
    )


def test_quality_report_flags_repetition():
    report = quality_report(["le chat chat chat chat chat"], [[-0.5, -0.6, -0.7, -0.8, -0.9]])
    assert report["texts"][0]["repetition_ratio"] > 0.5
    assert report["texts"][0]["distinct_ratio"] < 0.5


def test_build_langfuse_traces_deterministic_ids():
    import uuid

    records = [
        {
            "model": "stable-v1",
            "prompt_id": "p-001",
            "run": 1,
            "tokens": ["le", "chat", "dort"],
            "logprobs": [-0.3, -0.4, -0.5],
            "latency_ms": 120,
        }
    ]
    batches = build_langfuse_traces("stable-v1", records)
    assert len(batches) == 1
    body = batches[0]["body"]
    assert body["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "stable-v1:p-001:1"))
    assert body["scores"][0]["name"] == "mean_logprob"
    assert batches[0]["type"] == "trace"


def test_export_langfuse_local_mode(tmp_path, monkeypatch):
    import app.services.evals as evals_module

    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(evals_module, "LANGFUSE_EXPORT_DIR", tmp_path)
    records = [
        {
            "model": "stable-v1",
            "prompt_id": "p-001",
            "run": 1,
            "tokens": ["le", "chat"],
            "logprobs": [-0.3, -0.4],
        }
    ]
    result = export_langfuse("stable-v1", records)
    assert result["mode"] == "local"
    assert result["n_traces"] == 1
    path = tmp_path / result["path"].split("/")[-1]
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1


def test_api_benchmark_mock(client):
    response = client.post("/api/evals/benchmark", json={"tasks": ["lambada_openai"], "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "mock"
    assert "lambada_openai" in body["tasks"]


def test_api_benchmark_rejects_unknown_task(client):
    response = client.post("/api/evals/benchmark", json={"tasks": ["mmlu_pro"]})
    assert response.status_code == 422


def test_api_benchmark_requires_model(client):
    from app.services import probe as probe_module
    from app.services.probe import reset_probe

    os.environ["LLM_NEUROSCOPE_BACKEND"] = "torch"
    try:
        probe_module._probe = None
        response = client.post("/api/evals/benchmark", json={"tasks": ["lambada_openai"]})
        assert response.status_code == 400
    finally:
        os.environ["LLM_NEUROSCOPE_BACKEND"] = "mock"
        reset_probe()


def test_api_quality_mock(client):
    response = client.post(
        "/api/evals/quality",
        json={"texts": ["le chat dort profondément", "la science progresse chaque jour"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_texts"] == 2
    assert body["judge"] is None
    assert len(body["texts"]) == 2


def test_api_quality_judge_cloud_unavailable_note(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/api/evals/quality",
        json={"texts": ["texte simple"], "judge": "cloud"},
    )
    assert response.status_code == 200
    assert "OPENAI_API_KEY" in response.json().get("judge_note", "")


def test_api_quality_judge_local_wiring(client, monkeypatch):
    import app.services.evals as evals_module

    monkeypatch.setattr(
        evals_module,
        "run_local_judge_evals",
        lambda texts: {"judge": "deepeval/local:stub", "results": [{"text": texts[0][:60], "score": 0.8}]},
    )
    response = client.post(
        "/api/evals/quality",
        json={"texts": ["texte simple"], "judge": "local"},
    )
    assert response.status_code == 200
    assert response.json()["judge"]["judge"] == "deepeval/local:stub"


def test_api_quality_rejects_unknown_judge(client):
    response = client.post(
        "/api/evals/quality",
        json={"texts": ["texte simple"], "judge": "gpt5"},
    )
    assert response.status_code == 422


def test_local_coherence_metric_parses_judge_json():
    from deepeval.test_case import LLMTestCase

    from app.services.judge import LocalCoherenceMetric

    class StubJudge:
        def generate(self, prompt):
            return 'Voici mon évaluation: {"score": 4, "reason": "texte fluide"}'

    metric = LocalCoherenceMetric(StubJudge())
    score = metric.measure(LLMTestCase(input="", actual_output="un texte propre"))
    assert score == 0.8
    assert metric.reason == "texte fluide"
    assert metric.success is True


def test_local_coherence_metric_rejects_invalid_json():
    from deepeval.test_case import LLMTestCase

    from app.services.judge import LocalCoherenceMetric

    class BadJudge:
        def generate(self, prompt):
            return "je ne sais pas"

    metric = LocalCoherenceMetric(BadJudge())
    with pytest.raises(ValueError, match="sans JSON"):
        metric.measure(LLMTestCase(input="", actual_output="texte"))


def test_api_langfuse_export_local(client, tmp_path, monkeypatch):
    import app.services.evals as evals_module

    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(evals_module, "LANGFUSE_EXPORT_DIR", tmp_path)

    uploaded = client.post(
        "/api/datasets",
        files={
            "file": (
                "runs.jsonl",
                b'{"model": "stable-v1", "prompt_id": "p-1", "run": 1, "tokens": ["a", "b"], "logprobs": [-0.1, -0.2]}\n',
            )
        },
    )
    dataset_id = uploaded.json()["id"]
    response = client.post("/api/evals/langfuse", json={"dataset_id": dataset_id})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "local"
    assert body["n_traces"] == 1


def test_api_langfuse_ambiguous_model(client):
    uploaded = client.post(
        "/api/datasets",
        files={
            "file": (
                "runs.jsonl",
                b'{"model": "a-v1", "prompt_id": "p-1", "run": 1, "tokens": ["a"], "logprobs": [-0.1]}\n'
                b'{"model": "b-v1", "prompt_id": "p-2", "run": 1, "tokens": ["b"], "logprobs": [-0.3]}\n',
            )
        },
    )
    dataset_id = uploaded.json()["id"]
    response = client.post("/api/evals/langfuse", json={"dataset_id": dataset_id})
    assert response.status_code == 400
    response = client.post(
        "/api/evals/langfuse", json={"dataset_id": dataset_id, "model": "a-v1"}
    )
    assert response.status_code == 200
