from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import evals as evals_service
from app.services.evals import (
    BENCHMARK_TASKS,
    EvalError,
    deepeval_judge_available,
    export_langfuse,
    mock_benchmark,
    quality_report,
    run_benchmark,
    run_judge_evals,
)
from app.services.probe import get_probe
from app.services.storage import store

router = APIRouter(prefix="/api/evals", tags=["evals"])


class BenchmarkRequest(BaseModel):
    model_name: str | None = None
    tasks: list[str] = Field(default=["lambada_openai"], min_length=1, max_length=5)
    limit: int = Field(default=10, ge=1, le=100)


class QualityRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    judge: str = Field(default="none", pattern="^(none|local|cloud)$")


class LangfuseRequest(BaseModel):
    dataset_id: str
    model: str | None = None


def _resolve_model_name(requested: str | None) -> tuple[str, str]:
    probe = get_probe()
    status = probe.status()
    if requested:
        return requested, status["backend"]
    if status.get("loaded") and status.get("model_name"):
        return status["model_name"], status["backend"]
    raise HTTPException(
        status_code=400,
        detail="Aucun modèle chargé. Appelez POST /api/probe/load ou précisez model_name.",
    )


@router.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    unknown = [task for task in request.tasks if task not in BENCHMARK_TASKS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Tâches inconnues: {unknown}. Tâches disponibles: {BENCHMARK_TASKS}",
        )
    try:
        model_name, backend = _resolve_model_name(request.model_name)
        if backend == "mock":
            return mock_benchmark(model_name, request.tasks, request.limit)
        probe = get_probe()
        return run_benchmark(model_name, request.tasks, request.limit, probe.device or "cpu")
    except HTTPException:
        raise
    except EvalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Benchmark impossible ({exc}). Installez lm-eval (pip install lm-eval).",
        ) from exc


@router.post("/quality")
async def quality(request: QualityRequest):
    probe = get_probe()
    if not probe.status()["loaded"]:
        raise HTTPException(status_code=400, detail="Aucun modèle chargé. Appelez d'abord POST /api/probe/load.")
    try:
        logprobs_per_text = [probe.text_logprobs(text) for text in request.texts[:64]]
        report = quality_report(request.texts[:64], logprobs_per_text)
    except EvalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.judge == "cloud":
        if not deepeval_judge_available():
            report["judge_note"] = (
                "Juge cloud indisponible: définissez OPENAI_API_KEY pour activer G-Eval (gpt-4o-mini)."
            )
        else:
            try:
                report["judge"] = run_judge_evals(request.texts)
            except Exception as exc:
                report["judge"] = {"judge": "deepeval/gpt-4o-mini", "error": str(exc)[:200]}
    elif request.judge == "local":
        try:
            report["judge"] = evals_service.run_local_judge_evals(request.texts)
        except Exception as exc:
            report["judge"] = {"judge": "deepeval/local", "error": str(exc)[:200]}
    return report


@router.post("/langfuse")
async def langfuse_export(request: LangfuseRequest):
    records = store.records(request.dataset_id)
    if records is None:
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    model = request.model
    if model is None:
        models = sorted({record["model"] for record in records})
        if len(models) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Le dataset contient {len(models)} modèles; précisez model.",
            )
        model = models[0]
    try:
        return export_langfuse(model, records)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export Langfuse impossible ({exc}).") from exc
