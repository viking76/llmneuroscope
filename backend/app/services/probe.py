from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEAD_THRESHOLD = 1e-4
HISTOGRAM_BINS = 16
TOP_NEURONS = 5
EXPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "weight-edits"


class ProbeError(RuntimeError):
    pass


def resolve_export_path(filename: str | None) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"edit-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.safetensors"
    candidate = (EXPORT_DIR / filename).resolve()
    if EXPORT_DIR.resolve() not in candidate.parents:
        raise ProbeError("Chemin de fichier non autorisé: exportez dans data/weight-edits/")
    return candidate


def edit_metadata(model_name: str, edit_log: list) -> dict:
    return {
        "model_name": model_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "edit_log": json.dumps(edit_log),
    }


def attention_entropy_from_patterns(pattern: np.ndarray) -> np.ndarray:
    probabilities = np.clip(pattern, 1e-12, 1.0)
    return -(probabilities * np.log2(probabilities)).sum(axis=-1)


def summarize_attention(patterns: list[np.ndarray]) -> dict:
    entropies = [attention_entropy_from_patterns(p[0]) for p in patterns]
    n_layers, n_heads = len(entropies), entropies[0].shape[0]
    sequence_length = patterns[0].shape[-1]
    max_entropy = float(np.log2(max(sequence_length, 2)))
    head_matrix = [
        [round(float(head_entropy.mean()), 4) for head_entropy in layer_entropy]
        for layer_entropy in entropies
    ]
    layer_entropy = [round(float(np.mean(layer)), 4) for layer in head_matrix]
    flat = [
        {"layer": layer, "head": head, "entropy": head_matrix[layer][head],
         "ratio": round(head_matrix[layer][head] / max_entropy, 4)}
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    top_heads = sorted(flat, key=lambda entry: entry["entropy"], reverse=True)[:TOP_NEURONS]
    focused_heads = sorted(flat, key=lambda entry: entry["entropy"])[:TOP_NEURONS]
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "sequence_length": sequence_length,
        "max_entropy": round(max_entropy, 4),
        "layer_entropy": layer_entropy,
        "head_matrix": head_matrix,
        "diffuse_heads": [entry for entry in top_heads if entry["ratio"] > 0.9],
        "top_heads": top_heads,
        "focused_heads": focused_heads,
    }


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
        self.edit_log = []
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
            "n_edits": len(self.edit_log),
            "edit_log": self.edit_log[-10:],
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
            self.edit_log.append(
                {"layer": layer, "neuron": index, "action": action, "factor": factor}
            )
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
        self.edit_log.append({"action": "restore", "restored_layers": restored})
        return {"action": "restore", "restored_layers": restored}

    def export_weights(self, path: Path) -> dict:
        from safetensors.numpy import save_file

        if not self.snapshots:
            raise ProbeError("Aucune modification en cours à exporter")
        modified_layers = sorted(self.snapshots)
        save_file(
            {layer: self.layers[layer] for layer in modified_layers},
            str(path),
            metadata=edit_metadata(self.model_name, self.edit_log),
        )
        return {"path": str(path), "layers": modified_layers, "n_edits": len(self.edit_log)}

    def apply_weights(self, path: Path) -> dict:
        from safetensors.numpy import load_file

        tensors = load_file(str(path))
        applied, skipped = [], []
        for name, tensor in tensors.items():
            if name in self.layers and self.layers[name].shape == tensor.shape:
                if name not in self.snapshots:
                    self.snapshots[name] = self.layers[name].copy()
                self.layers[name] = tensor
                self.modified = True
                applied.append(name)
            else:
                skipped.append(name)
        if applied:
            self.edit_log.append({"action": "apply", "layers": applied, "from": str(path)})
        return {"applied": applied, "skipped": skipped}

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

    def attention_entropy(self, texts: list[str]) -> dict:
        patterns_per_text = []
        for text in texts[:16]:
            length = max(len(text.split()), 4)
            rng = np.random.default_rng(
                int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            )
            logits = rng.normal(0.0, 1.5, size=(4, length, length))
            mask = np.triu(np.ones((length, length), dtype=bool), k=1)
            logits[:, mask] = -np.inf
            logits -= logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            patterns_per_text.append(probabilities)
        averaged = [
            np.stack([patterns[layer] for patterns in patterns_per_text]).mean(axis=0)[None]
            for layer in range(4)
        ]
        return {
            "n_texts": len(texts[:16]),
            **summarize_attention(averaged),
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
        self.edit_log = []
        self.bridge = None

    def load(self, model_name: str, device: str) -> dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self.torch
        resolved = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None or self.model_name != model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
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
                "n_edits": 0,
                "edit_log": [],
            }
        return {
            "backend": self.backend,
            "loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "n_layers": len(self.layer_names()),
            "modified": self.modified,
            "snapshot_available": bool(self.snapshots),
            "n_edits": len(self.edit_log),
            "edit_log": self.edit_log[-10:],
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
            self.edit_log.append(
                {"layer": layer, "neuron": index, "action": action, "factor": factor}
            )
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
        self.edit_log.append({"action": "restore", "restored_layers": restored})
        return {"action": "restore", "restored_layers": restored}

    def export_weights(self, path: Path) -> dict:
        from safetensors.torch import save_file

        if not self.snapshots:
            raise ProbeError("Aucune modification en cours à exporter")
        modified_layers = sorted(self.snapshots)
        tensors = {
            layer: self.model.get_submodule(layer).weight.detach().cpu().contiguous()
            for layer in modified_layers
        }
        save_file(tensors, str(path), metadata=edit_metadata(self.model_name, self.edit_log))
        return {"path": str(path), "layers": modified_layers, "n_edits": len(self.edit_log)}

    def apply_weights(self, path: Path) -> dict:
        from safetensors.torch import load_file

        self._require_loaded()
        tensors = load_file(str(path), device=self.device)
        applied, skipped = [], []
        with self.torch.no_grad():
            for name, tensor in tensors.items():
                try:
                    weight = self.model.get_submodule(name).weight
                except AttributeError:
                    skipped.append(name)
                    continue
                if weight.shape != tensor.shape:
                    skipped.append(name)
                    continue
                if name not in self.snapshots:
                    self.snapshots[name] = weight.detach().clone()
                weight.copy_(tensor.to(weight.dtype))
                self.modified = True
                applied.append(name)
        if applied:
            self.edit_log.append({"action": "apply", "layers": applied, "from": str(path)})
        return {"applied": applied, "skipped": skipped}

    def _get_bridge(self):
        self._require_loaded()
        if self.bridge is None or self.bridge.cfg.model_name != self.model_name:
            from transformer_lens import TransformerBridge

            self.bridge = TransformerBridge.boot_transformers(
                self.model_name, device=self.device, dtype=self.torch.float32
            )
            self.bridge.enable_compatibility_mode()
        return self.bridge

    def attention_entropy(self, texts: list[str]) -> dict:
        bridge = self._get_bridge()
        patterns_per_text = []
        for text in texts[:16]:
            _, cache = bridge.run_with_cache(
                text, names_filter=lambda name: name.endswith("hook_pattern")
            )
            patterns = [
                value.float().cpu().numpy()
                for key, value in sorted(cache.items())
                if key.endswith("hook_pattern")
            ]
            if not patterns:
                raise ProbeError(
                    "Aucun pattern d'attention capturé — modèle incompatible avec TransformerLens"
                )
            patterns_per_text.append(patterns)
        n_layers = len(patterns_per_text[0])
        averaged = [
            np.stack([patterns[layer] for patterns in patterns_per_text]).mean(axis=0)
            for layer in range(n_layers)
        ]
        return {
            "n_texts": len(texts[:16]),
            "engine": "transformer-lens",
            **summarize_attention(averaged),
        }

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
