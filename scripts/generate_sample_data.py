"""Génère un dataset d'exemple JSONL pour tester LLN Neuroscope.

Usage: python scripts/generate_sample_data.py [output.jsonl]

Produit deux modèles:
- stable-v1: comportement sain
- degraded-v2: effondrement de logprobs après la position 20, boucles de
  répétition, pics de latence et dérive entre runs.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

VOCAB = ["le", "chat", "dort", "sur", "le", "tapis", "rouge", "et", "respire", "calmement"]
LOOP_TOKEN = "le"


def stable_record(model, prompt_id, run, rng):
    n = rng.randint(6, 14)
    tokens = [rng.choice(VOCAB) for _ in range(n)]
    logprobs = [round(rng.uniform(-0.2, -1.5), 3) for _ in range(n)]
    return {
        "model": model,
        "prompt_id": f"p-{prompt_id:03d}",
        "run": run,
        "tokens": tokens,
        "logprobs": logprobs,
        "latency_ms": round(rng.gauss(250, 40)),
        "timestamp": "2026-11-14T20:00:00Z",
    }


def degraded_record(model, prompt_id, run, rng):
    healthy = stable_record(model, prompt_id, run, rng)
    if run >= 2:
        n = len(healthy["tokens"])
        healthy["tokens"] = healthy["tokens"][:8] + [LOOP_TOKEN] * (n - 8) if n > 8 else [LOOP_TOKEN] * n
        healthy["logprobs"] = [
            round(lp, 3) if i < 8 else round(rng.uniform(-6.0, -12.0), 3)
            for i, lp in enumerate(healthy["logprobs"])
        ]
    if rng.random() < 0.1:
        healthy["latency_ms"] = round(rng.gauss(900, 60))
    return healthy


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_runs.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    records = []
    for prompt_id in range(1, 41):
        for run in (1, 2, 3):
            records.append(stable_record("stable-v1", prompt_id, run, rng))
            records.append(degraded_record("degraded-v2", prompt_id, run, rng))
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Écrit {len(records)} enregistrements dans {output}")


if __name__ == "__main__":
    main()
