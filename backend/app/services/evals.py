from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.metrics import repetition_ratio

BENCHMARK_TASKS = ["lambada_openai", "hellaswag", "arc_easy", "winogrande", "piqa"]
LANGFUSE_EXPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "langfuse-exports"


class EvalError(RuntimeError):
    pass


def run_benchmark(model_name: str, tasks: list[str], limit: int, device: str) -> dict:
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},dtype=float32",
        tasks=tasks,
        limit=limit,
        device=device,
    )
    entries = {}
    for task, metrics in results["results"].items():
        entries[task] = {
            key.split(",")[0]: round(float(value), 4)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    return {
        "engine": "lm-evaluation-harness",
        "model_name": model_name,
        "limit": limit,
        "tasks": entries,
    }


def mock_benchmark(model_name: str, tasks: list[str], limit: int) -> dict:
    entries = {}
    for task in tasks:
        seed = int(hashlib.sha256(f"{model_name}:{task}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        entries[task] = {
            "acc": round(float(rng.uniform(0.2, 0.6)), 4),
            "acc_stderr": round(float(rng.uniform(0.05, 0.2)), 4),
        }
    return {"engine": "mock", "model_name": model_name, "limit": limit, "tasks": entries}


def quality_report(texts: list[str], logprobs_per_text: list[list[float]]) -> dict:
    per_text = []
    for text, logprobs in zip(texts, logprobs_per_text):
        tokens = text.split()
        values = np.asarray(logprobs, dtype=float)
        perplexity = float(np.exp(-values.mean())) if values.size else None
        distinct = len(set(tokens)) / len(tokens) if tokens else 0.0
        per_text.append(
            {
                "text": text[:80],
                "n_tokens": len(tokens),
                "mean_logprob": round(float(values.mean()), 4) if values.size else None,
                "perplexity": round(perplexity, 4) if perplexity is not None else None,
                "logprob_std": round(float(values.std()), 4) if values.size else None,
                "repetition_ratio": round(repetition_ratio(tokens), 4),
                "distinct_ratio": round(float(distinct), 4),
            }
        )
    scored = [entry for entry in per_text if entry["perplexity"] is not None]
    aggregate = {
        "perplexity": round(float(np.mean([entry["perplexity"] for entry in scored])), 4) if scored else None,
        "repetition_ratio": round(float(np.mean([entry["repetition_ratio"] for entry in per_text])), 4),
        "distinct_ratio": round(float(np.mean([entry["distinct_ratio"] for entry in per_text])), 4),
    }
    return {"n_texts": len(texts), "texts": per_text, "aggregate": aggregate, "judge": None}


def deepeval_judge_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_local_judge_evals(texts: list[str]) -> dict:
    from deepeval.test_case import LLMTestCase

    from app.services.judge import JUDGE_MODEL, LocalCoherenceMetric, LocalInstructJudge

    metric = LocalCoherenceMetric(LocalInstructJudge())
    results = []
    for text in texts[:16]:
        try:
            metric.measure(LLMTestCase(input="", actual_output=text))
            results.append(
                {
                    "text": text[:60],
                    "score": metric.score,
                    "reason": (metric.reason or "")[:200],
                }
            )
        except Exception as exc:
            results.append({"text": text[:60], "error": str(exc)[:200]})
    return {"judge": f"deepeval/local:{JUDGE_MODEL}", "results": results}


def run_judge_evals(texts: list[str]) -> dict:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    metric = GEval(
        name="coherence",
        criteria="Rate the coherence and fluency of the text from 1 to 5.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gpt-4o-mini",
    )
    results = []
    for text in texts[:16]:
        try:
            metric.measure(LLMTestCase(input="", actual_output=text))
            results.append(
                {
                    "text": text[:60],
                    "score": metric.score,
                    "reason": (metric.reason or "")[:200],
                }
            )
        except Exception as exc:
            results.append({"text": text[:60], "error": str(exc)[:200]})
    return {"judge": "deepeval/gpt-4o-mini", "results": results}


def build_langfuse_traces(model: str, records: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    batches = []
    for record in records[:200]:
        logprobs = record.get("logprobs") or []
        trace_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{model}:{record.get('prompt_id')}:{record.get('run')}",
            )
        )
        score = round(float(np.mean(logprobs)), 4) if logprobs else None
        body = {
            "id": trace_id,
            "name": "llm-neuroscope-generation",
            "timestamp": now,
            "input": {"prompt_id": record.get("prompt_id"), "run": record.get("run")},
            "output": {"tokens": record.get("tokens"), "text": record.get("output")},
            "metadata": {
                "model": model,
                "latency_ms": record.get("latency_ms"),
                "n_tokens": len(record.get("tokens") or []),
            },
        }
        if score is not None:
            body["scores"] = [{"name": "mean_logprob", "value": score}]
        batches.append({"id": str(uuid.uuid4()), "type": "trace", "timestamp": now, "body": body})
    return batches


def export_langfuse(model: str, records: list[dict]) -> dict:
    batches = build_langfuse_traces(model, records)
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    if public_key and secret_key and host:
        from langfuse import Langfuse

        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        for batch in batches:
            body = batch["body"]
            client.trace(
                name=body["name"],
                input=body["input"],
                output=body["output"],
                metadata=body["metadata"],
                scores=body.get("scores"),
            )
        client.flush()
        return {"mode": "push", "n_traces": len(batches), "host": host}
    LANGFUSE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = LANGFUSE_EXPORT_DIR / f"traces-{stamp}.jsonl"
    path.write_text("\n".join(json.dumps(batch) for batch in batches), encoding="utf-8")
    return {
        "mode": "local",
        "n_traces": len(batches),
        "path": f"data/langfuse-exports/{path.name}",
        "hint": "Définissez LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY et LANGFUSE_HOST pour pousser directement vers un serveur Langfuse.",
    }
