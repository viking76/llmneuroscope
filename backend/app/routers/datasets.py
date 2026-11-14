from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ingestion import IngestionError, parse_content
from app.services.storage import store

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    try:
        records = parse_content(file.filename or "upload.jsonl", content)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    summary = store.save(file.filename or "upload", records)
    return summary


@router.get("")
async def list_datasets():
    return store.list()


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    entry = store.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    summary = store._summary(entry)
    summary["sample_records"] = entry["records"][:5]
    return summary


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if not store.delete(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset introuvable.")
    return {"deleted": dataset_id}
