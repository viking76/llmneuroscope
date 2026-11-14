from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.probe import ProbeError, get_probe, resolve_export_path

router = APIRouter(prefix="/api/probe", tags=["probe"])


class LoadRequest(BaseModel):
    model_name: str = Field(default="gpt2")
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")


class AnalyzeTextsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class ModifyWeightRequest(BaseModel):
    layer: str
    neuron: int | None = None
    action: str = Field(pattern="^(scale|zero|restore)$")
    factor: float | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=16, ge=1, le=256)


class ExportRequest(BaseModel):
    filename: str | None = None


class ApplyRequest(BaseModel):
    filename: str


@router.get("/status")
async def status():
    return get_probe().status()


@router.post("/load")
async def load(request: LoadRequest):
    probe = get_probe()
    try:
        if hasattr(probe, "load"):
            return probe.load(request.model_name, request.device)
        probe.model_name = request.model_name
        return probe.status()
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Chargement du modèle impossible ({exc}). "
            "Installez torch et transformers (backend/requirements-probe.txt) "
            "et vérifiez le nom du modèle open-weights.",
        ) from exc


@router.post("/analyze-texts")
async def analyze_texts(request: AnalyzeTextsRequest):
    try:
        return get_probe().activation_stats(request.texts)
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/layers")
async def layers():
    probe = get_probe()
    if not probe.status()["loaded"]:
        raise HTTPException(status_code=400, detail="Aucun modèle chargé.")
    return {"layers": probe.layer_names()}


@router.get("/weights/{layer:path}")
async def weights(layer: str):
    try:
        return get_probe().weight_stats(layer)
    except ProbeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/weights/modify")
async def modify_weights(request: ModifyWeightRequest):
    try:
        return get_probe().modify(request.layer, request.neuron, request.action, request.factor)
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        return get_probe().generate(request.prompt, request.max_new_tokens)
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/weights/export")
async def export_weights(request: ExportRequest):
    try:
        path = resolve_export_path(request.filename)
        result = get_probe().export_weights(path)
        result["path"] = str(path.relative_to(path.parents[2]))
        return result
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/weights/apply")
async def apply_weights(request: ApplyRequest):
    try:
        path = resolve_export_path(request.filename)
        if not path.exists():
            raise ProbeError(f"Fichier introuvable: {path}")
        return get_probe().apply_weights(path)
    except ProbeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
