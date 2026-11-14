from __future__ import annotations

import hashlib
import os
from collections import Counter

import numpy as np

DEAD_THRESHOLD = 1e-4
HISTOGRAM_BINS = 16
TOP_NEURONS = 5


class ProbeError(RuntimeError):
    pass


class MockBackend:
    """Backend de simulation déterministe, pour tester sans torch ni modèle réel."""

    backend = "mock"

    def __init__(self):
        self.model_name = "mock-tiny-16d"
        self.device = "cpu"
        self.d_model = 16
        self.vocab_size = 32
        self.layers = {}
        self.snapshots = {}
        self.modified = False
        for index in range(4):
            rng = np.random.default_rng(100 + index)
            self.layers[f"layer_{index}.mlp"] = rng.normal(0.0, 0.05, size=(self.d_model, self.d_model))

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "n_layers": len(self.layers),
            "modified": self.modified,
            "snapshot_available": bool(self.snapshots),
        }

    def layer_names(self) -> list[str]:
        return list(self.layers)

    def _text_vector(self, text: str, layer_index: int) -> np.ndarray:
        digest = hashlib.sha256(f"{layer_index}:{text}".encode()).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(float)
        vector = (values[: self.d_model] / 255.0) - 0.5
        if layer_index > 1:
            vector = vector * (1.0 + layer_index * 0.4)
        return vector

    def activation_stats(self, texts: list[str]) -> dict:
        per_layer = []
        for layer_index, layer in enumerate(self.layers):
            matrix = np.stack([self._text_vector(text, layer_index) for text in texts])
            neuron_max = np.abs(matrix).max(axis=0)
            dead = neuron_max < DEAD_THRESHOLD
            top = np.argsort(neuron_max)[::-1][:TOP_NEURONS]
            per_layer.append(
                {
                    "layer": layer,
                    "mean": float(matrix.mean()),
                    "std": float(matrix.std()),
                    "dead_neuron_ratio": round(float(dead.mean()), 4),
                    "dead_neurons": int(dead.sum()),
                    "top_neurons": [
                        {"index": int(i), "max_activation": round(float(neuron_max[i]), 4)}
                        for i in top
                    ],
                }
            )
        return {"n_texts": len(texts), "layers": per_layer}

    def weight_stats(self, layer: str) -> dict:
        if layer not in self.layers:
            raise ProbeError(f"Couche inconnue: {layer}")
        weights = self.layers[layer]
        flat = weights.flatten()
        histogram, edges = np.histogram(flat, bins=HISTOGRAM_BINS)
        return {
            "layer": layer,
            "shape": list(weights.shape),
            "mean": round(float(weights.mean()), 6),
            "std": round(float(weights.std()), 6),
            "min": round(float(weights.min()), 6),
            "max": round(float(weights.max()), 6),
            "norm": round(float(np.linalg.norm(weights)), 6),
            "dead_ratio": round(float((np.abs(flat) < DEAD_THRESHOLD).mean()), 6),
            "histogram": {
                "counts": [int(c) for c in histogram],
                "edges": [round(float(e), 6) for e in edges],
            },
            "preview": [round(float(v), 6) for v in weights[0, :8]],
        }

    def modify(self, layer: str, index: int | None, action: str, factor: float | None) -> dict:
        if layer not in self.layers:
            raise ProbeError(f"Couche inconnue: {layer}")
        weights = self.layers[layer]
        if action != "restore":
            if index is None or not 0 <= index < weights.shape[0]:
                raise ProbeError(
                    f"Index de neurone requis entre 0 et {weights.shape[0] - 1}"
                )
            if layer not in self.snapshots:
                self.snapshots[layer] = weights.copy()
            before = self.weight_stats(layer)
            if action == "scale":
                if factor is None:
                    raise ProbeError("factor requis pour l'action scale")
                weights[index] *= factor
            elif action == "zero":
                weights[index] = 0.0
            else:
                raise ProbeError(f"Action inconnue: {action}")
            self.modified = True
            return {
                "layer": layer,
                "neuron": index,
                "action": action,
                "before": before,
                "after": self.weight_stats(layer),
            }
        if not self.snapshots:
            raise ProbeError("Aucune modification à restaurer")
        restored = []
        for snapshot_layer, snapshot in self.snapshots.items():
            self.layers[snapshot_layer] = snapshot.copy()
            restored.append(snapshot_layer)
        self.snapshots = {}
        self.modified = False
        return {"action": "restore", "restored_layers": restored}

    def generate(self, prompt: str, max_new_tokens: int) -> dict:
        tokens = []
        logprobs = []
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        for step in range(max_new_tokens):
            logits = rng.normal(0.0, 2.0, size=self.vocab_size)
            logits -= logits.max()
            probabilities = np.exp(logits) / np.exp(logits).sum()
            token_id = int(rng.choice(self.vocab_size, p=probabilities))
            tokens.append(f"tok{token_id}")
            logprobs.append(round(float(np.log(probabilities[token_id])), 4))
        return {
            "prompt": prompt,
            "tokens": tokens,
            "logprobs": logprobs,
            "perplexity": round(float(np.exp(-np.mean(logprobs))), 4),
        }


class TorchBackend:
    """Backend réel: modèle open-weights chargé localement via transformers."""

    backend = "torch"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.snapshots = {}
        self.modified = False

    def load(self, model_name: str, device: str) -> dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self.torch
        resolved = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None or self.model_name != model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).to(resolved)
        self.model.eval()
        self.model_name = model_name
        self.device = resolved
        return self.status()

    def status(self) -> dict:
        if self.model is None:
            return {
                "backend": self.backend,
                "loaded": False,
                "model_name": None,
                "device": None,
                "n_layers": 0,
                "modified": False,
                "snapshot_available": False,
            }
        return {
            "backend": self.backend,
            "loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "n_layers": len(self.layer_names()),
            "modified": self.modified,
            "snapshot_available": bool(self.snapshots),
        }

    def layer_names(self) -> list[str]:
        if self.model is None:
            return []
        names = []
        for name, module in self.model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dim() == 2:
                names.append(name)
        return names

    def _require_loaded(self):
        if self.model is None:
            raise ProbeError("Aucun modèle chargé. Appelez d'abord POST /api/probe/load.")

    def activation_stats(self, texts: list[str]) -> dict:
        self._require_loaded()
        torch = self.torch
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True)
        per_layer = []
        for layer_index, hidden in enumerate(output.hidden_states):
            matrix = hidden.float().cpu().numpy().reshape(-1, hidden.shape[-1])
            neuron_max = np.abs(matrix).max(axis=0)
            dead = neuron_max < DEAD_THRESHOLD
            top = np.argsort(neuron_max)[::-1][:TOP_NEURONS]
            per_layer.append(
                {
                    "layer": f"hidden_state_{layer_index}",
                    "mean": round(float(matrix.mean()), 6),
                    "std": round(float(matrix.std()), 6),
                    "dead_neuron_ratio": round(float(dead.mean()), 4),
                    "dead_neurons": int(dead.sum()),
                    "top_neurons": [
                        {"index": int(i), "max_activation": round(float(neuron_max[i]), 4)}
                        for i in top
                    ],
                }
            )
        return {"n_texts": len(texts), "layers": per_layer}

    def _weight_tensor(self, layer: str):
        self._require_loaded()
        try:
            module = self.model.get_submodule(layer)
        except AttributeError as exc:
            raise ProbeError(f"Couche inconnue: {layer}") from exc
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            raise ProbeError(f"Couche sans poids 2D: {layer}")
        return weight

    def weight_stats(self, layer: str) -> dict:
        weight = self._weight_tensor(layer)
        flat = weight.detach().float().cpu().numpy().flatten()
        histogram, edges = np.histogram(flat, bins=HISTOGRAM_BINS)
        matrix = flat.reshape(-1)
        return {
            "layer": layer,
            "shape": list(weight.shape),
            "mean": round(float(matrix.mean()), 6),
            "std": round(float(matrix.std()), 6),
            "min": round(float(matrix.min()), 6),
            "max": round(float(matrix.max()), 6),
            "norm": round(float(np.linalg.norm(matrix)), 6),
            "dead_ratio": round(float((np.abs(matrix) < DEAD_THRESHOLD).mean()), 6),
            "histogram": {
                "counts": [int(c) for c in histogram],
                "edges": [round(float(e), 6) for e in edges],
            },
            "preview": [round(float(v), 6) for v in flat[:8]],
        }

    def modify(self, layer: str, index: int | None, action: str, factor: float | None) -> dict:
        weight = self._weight_tensor(layer)
        torch = self.torch
        if action != "restore":
            if index is None or not 0 <= index < weight.shape[0]:
                raise ProbeError(f"Index de neurone requis entre 0 et {weight.shape[0] - 1}")
            if layer not in self.snapshots:
                self.snapshots[layer] = weight.detach().clone()
            before = self.weight_stats(layer)
            with torch.no_grad():
                if action == "scale":
                    if factor is None:
                        raise ProbeError("factor requis pour l'action scale")
                    weight[index] *= factor
                elif action == "zero":
                    weight[index] = 0.0
                else:
                    raise ProbeError(f"Action inconnue: {action}")
            self.modified = True
            return {
                "layer": layer,
                "neuron": index,
                "action": action,
                "before": before,
                "after": self.weight_stats(layer),
            }
        if not self.snapshots:
            raise ProbeError("Aucune modification à restaurer")
        with torch.no_grad():
            for snapshot_layer, snapshot in self.snapshots.items():
                self.model.get_submodule(snapshot_layer).weight.copy_(snapshot)
        restored = list(self.snapshots)
        self.snapshots = {}
        self.modified = False
        return {"action": "restore", "restored_layers": restored}

    def generate(self, prompt: str, max_new_tokens: int) -> dict:
        self._require_loaded()
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        tokens = self.tokenizer.convert_ids_to_tokens(output.sequences[0][-max_new_tokens:])
        logprobs = []
        for score in output.scores:
            probabilities = torch.softmax(score[0], dim=-1)
            token_id = output.sequences[0][-max_new_tokens + len(logprobs)]
            logprobs.append(round(float(torch.log(probabilities[token_id])), 4))
        return {
            "prompt": prompt,
            "tokens": tokens,
            "logprobs": logprobs,
            "perplexity": round(float(np.exp(-np.mean(logprobs))), 4) if logprobs else None,
        }


_probe = None


def get_probe() -> MockBackend | TorchBackend:
    global _probe
    if _probe is None:
        mode = os.environ.get("LLM_NEUROSCOPE_BACKEND", "auto")
        if mode == "mock":
            _probe = MockBackend()
        elif mode == "torch":
            _probe = TorchBackend()
        else:
            try:
                import torch
                import transformers

                _probe = TorchBackend()
            except ImportError:
                _probe = MockBackend()
    return _probe


def reset_probe():
    global _probe
    _probe = None
