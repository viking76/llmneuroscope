import os

os.environ["LLM_NEUROSCOPE_BACKEND"] = "mock"

import pytest

from app.services.probe import ProbeError, MockBackend, reset_probe

reset_probe()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    reset_probe()
    return TestClient(app)


def test_mock_backend_status():
    probe = MockBackend()
    status = probe.status()
    assert status["loaded"] is True
    assert status["backend"] == "mock"
    assert status["n_layers"] == 4


def test_mock_activation_stats():
    probe = MockBackend()
    result = probe.activation_stats(["bonjour le monde", "salut la famille"])
    assert result["n_texts"] == 2
    assert len(result["layers"]) == 4
    for layer in result["layers"]:
        assert 0.0 <= layer["dead_neuron_ratio"] <= 1.0
        assert len(layer["top_neurons"]) == 5


def test_mock_deterministic_activations():
    probe = MockBackend()
    first = probe.activation_stats(["texte fixe"])
    second = probe.activation_stats(["texte fixe"])
    assert first == second


def test_mock_weight_stats():
    probe = MockBackend()
    stats = probe.weight_stats("layer_0.mlp")
    assert stats["shape"] == [16, 16]
    assert len(stats["histogram"]["counts"]) == 16
    assert len(stats["preview"]) == 8
    assert stats["mean"] == pytest.approx(0.0, abs=0.05)


def test_mock_modify_and_restore():
    probe = MockBackend()
    layer = "layer_1.mlp"
    original = probe.layers[layer].copy()
    result = probe.modify(layer, 3, "scale", 2.0)
    assert result["action"] == "scale"
    assert result["after"]["norm"] > result["before"]["norm"]
    assert probe.modified is True
    restored = probe.modify(layer, None, "restore", None)
    assert restored["action"] == "restore"
    assert layer in restored["restored_layers"]
    assert probe.modified is False
    assert probe.layers[layer] == pytest.approx(original)


def test_mock_zero_neuron():
    probe = MockBackend()
    probe.modify("layer_2.mlp", 0, "zero", None)
    assert probe.layers["layer_2.mlp"][0].tolist() == [0.0] * 16
    probe.modify("layer_2.mlp", None, "restore", None)


def test_mock_generate_deterministic():
    probe = MockBackend()
    first = probe.generate("Test prompt", 8)
    second = probe.generate("Test prompt", 8)
    assert first == second
    assert len(first["tokens"]) == 8
    assert len(first["logprobs"]) == 8
    assert first["perplexity"] >= 1.0


def test_unknown_layer_raises():
    probe = MockBackend()
    with pytest.raises(ProbeError):
        probe.weight_stats("inexistant")


def test_api_status(client):
    response = client.get("/api/probe/status")
    assert response.status_code == 200
    assert response.json()["backend"] == "mock"


def test_api_full_workflow(client):
    load = client.post("/api/probe/load", json={"model_name": "mock", "device": "cpu"})
    assert load.status_code == 200

    analysis = client.post(
        "/api/probe/analyze-texts", json={"texts": ["premier texte", "second texte"]}
    )
    assert analysis.status_code == 200
    assert len(analysis.json()["layers"]) == 4

    layers = client.get("/api/probe/layers")
    assert layers.status_code == 200
    assert len(layers.json()["layers"]) == 4

    weights = client.get("/api/probe/weights/layer_0.mlp")
    assert weights.status_code == 200

    modify = client.post(
        "/api/probe/weights/modify",
        json={"layer": "layer_0.mlp", "neuron": 2, "action": "scale", "factor": 0.5},
    )
    assert modify.status_code == 200
    assert modify.json()["after"]["norm"] < modify.json()["before"]["norm"]

    restore = client.post(
        "/api/probe/weights/modify", json={"layer": "layer_0.mlp", "action": "restore"}
    )
    assert restore.status_code == 200

    generation = client.post(
        "/api/probe/generate", json={"prompt": "Bonjour", "max_new_tokens": 5}
    )
    assert generation.status_code == 200
    assert len(generation.json()["tokens"]) == 5


def test_api_modify_requires_neuron(client):
    response = client.post(
        "/api/probe/weights/modify",
        json={"layer": "layer_0.mlp", "action": "scale", "factor": 2.0},
    )
    assert response.status_code == 400


def test_export_apply_roundtrip(tmp_path):
    probe = MockBackend()
    layer = "layer_0.mlp"
    original = probe.layers[layer].copy()
    probe.modify(layer, 1, "scale", 3.0)
    modified = probe.layers[layer].copy()

    export_path = tmp_path / "edit.safetensors"
    result = probe.export_weights(export_path)
    assert export_path.exists()
    assert result["layers"] == [layer]

    probe.modify(layer, None, "restore", None)
    assert probe.layers[layer] == pytest.approx(original)

    applied = probe.apply_weights(export_path)
    assert applied["applied"] == [layer]
    assert probe.layers[layer] == pytest.approx(modified)


def test_export_requires_active_modifications(tmp_path):
    probe = MockBackend()
    with pytest.raises(ProbeError):
        probe.export_weights(tmp_path / "nothing.safetensors")


def test_apply_skips_unknown_layers(tmp_path):
    from safetensors.numpy import save_file

    import numpy as np

    probe = MockBackend()
    foreign = tmp_path / "foreign.safetensors"
    save_file({"inconnue.mlp": np.zeros((4, 4), dtype=np.float32)}, str(foreign))
    result = probe.apply_weights(foreign)
    assert result["applied"] == []
    assert result["skipped"] == ["inconnue.mlp"]


def test_api_export_and_apply(client, tmp_path):
    from app.services.probe import EXPORT_DIR

    client.post(
        "/api/probe/weights/modify",
        json={"layer": "layer_3.mlp", "neuron": 0, "action": "scale", "factor": 1.5},
    )
    export = client.post(
        "/api/probe/weights/export", json={"filename": "test-edit.safetensors"}
    )
    assert export.status_code == 200
    assert (EXPORT_DIR / "test-edit.safetensors").exists()

    client.post(
        "/api/probe/weights/modify", json={"layer": "layer_3.mlp", "action": "restore"}
    )
    apply_response = client.post(
        "/api/probe/weights/apply", json={"filename": "test-edit.safetensors"}
    )
    assert apply_response.status_code == 200
    assert apply_response.json()["applied"] == ["layer_3.mlp"]

    traversal = client.post(
        "/api/probe/weights/export", json={"filename": "../evil.safetensors"}
    )
    assert traversal.status_code == 400

    (EXPORT_DIR / "test-edit.safetensors").unlink()


def test_status_includes_edit_log(client):
    status = client.get("/api/probe/status").json()
    assert "n_edits" in status
    assert "edit_log" in status


def test_mock_attention_entropy_shape_and_bounds():
    probe = MockBackend()
    result = probe.attention_entropy(["Le chat dort.", "Explique la photosynthèse."])
    assert result["n_texts"] == 2
    assert result["n_layers"] == 4
    assert result["n_heads"] == 4
    assert len(result["head_matrix"]) == 4
    assert all(len(row) == 4 for row in result["head_matrix"])
    for row in result["head_matrix"]:
        for value in row:
            assert 0.0 <= value <= result["max_entropy"]
    for entry in result["top_heads"] + result["focused_heads"]:
        assert 0.0 <= entry["ratio"] <= 1.0


def test_mock_attention_deterministic():
    probe = MockBackend()
    first = probe.attention_entropy(["texte fixe"])
    second = probe.attention_entropy(["texte fixe"])
    assert first == second


def test_mock_attention_causal_masking():
    import math

    probe = MockBackend()
    result = probe.attention_entropy(["un deux trois quatre cinq six"])
    assert result["sequence_length"] == 6
    assert result["max_entropy"] == pytest.approx(math.log2(6), abs=1e-4)
    assert result["layer_entropy"][0] < result["max_entropy"]


def test_api_attention_endpoint(client):
    response = client.post(
        "/api/probe/attention", json={"texts": ["premier texte", "second texte"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_layers"] == 4
    assert len(body["layer_entropy"]) == 4
