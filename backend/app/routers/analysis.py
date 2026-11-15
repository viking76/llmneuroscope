from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from app.services.comparison import compare_models
from app.services.metrics import analyze_model, detect_anomalies, detect_blocking_points, heatmap
from app.services.reporting import build_report, export_csv, export_pdf
from app.services.storage import store

router = APIRouter(prefix="/api/datasets/{dataset_id}", tags=["analysis"])


def _load_records(dataset_id: str) -> list[dict]:
    records = store.records(dataset_id)
    if records is None:
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    return records


def _models(records: list[dict]) -> list[str]:
    return sorted({r["model"] for r in records})


def _resolve_model(records: list[dict], model: str | None) -> str:
    available = _models(records)
    if model is None:
        return available[0]
    if model not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model}' introuvable. Modèles disponibles: {available}",
        )
    return model


@router.get("/analysis")
async def get_analysis(dataset_id: str, model: str | None = Query(default=None)):
    records = _load_records(dataset_id)
    selected = _resolve_model(records, model)
    return analyze_model(records, selected)


@router.get("/heatmap")
async def get_heatmap(dataset_id: str, model: str | None = Query(default=None)):
    records = _load_records(dataset_id)
    selected = _resolve_model(records, model)
    return heatmap([r for r in records if r["model"] == selected])


@router.get("/blocking-points")
async def get_blocking_points(dataset_id: str, model: str | None = Query(default=None)):
    records = _load_records(dataset_id)
    selected = _resolve_model(records, model)
    return {"model": selected, "points": detect_blocking_points(records, selected)}


@router.get("/anomalies")
async def get_anomalies(dataset_id: str, model: str | None = Query(default=None)):
    records = _load_records(dataset_id)
    selected = _resolve_model(records, model)
    return {"model": selected, "anomalies": detect_anomalies(records, selected)}


@router.get("/report.csv")
async def get_report_csv(dataset_id: str, model: str | None = Query(default=None)):
    try:
        report = build_report(dataset_id, model)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"rapport-{report['model']}.csv"
    return Response(
        content=export_csv(report),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/report.pdf")
async def get_report_pdf(dataset_id: str, model: str | None = Query(default=None)):
    try:
        report = build_report(dataset_id, model)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"rapport-{report['model']}.pdf"
    return Response(
        content=export_pdf(report),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/compare")
async def get_comparison(
    dataset_id: str,
    model_a: str | None = Query(default=None),
    model_b: str | None = Query(default=None),
):
    records = _load_records(dataset_id)
    available = _models(records)
    if len(available) < 2:
        raise HTTPException(
            status_code=400,
            detail="La comparaison nécessite au moins deux modèles dans le dataset.",
        )
    resolved_a = _resolve_model(records, model_a)
    resolved_b = _resolve_model(records, model_b or next(m for m in available if m != resolved_a))
    if resolved_a == resolved_b:
        raise HTTPException(status_code=400, detail="Les deux modèles doivent être différents.")
    analysis_a = analyze_model(records, resolved_a)
    analysis_b = analyze_model(records, resolved_b)
    analysis_a["_records"] = [r for r in records if r["model"] == resolved_a]
    analysis_b["_records"] = [r for r in records if r["model"] == resolved_b]
    return compare_models(analysis_a, analysis_b)
