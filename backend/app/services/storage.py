from __future__ import annotations

import uuid
from datetime import datetime, timezone


class DatasetStore:
    def __init__(self):
        self._datasets: dict[str, dict] = {}

    def save(self, name: str, records: list[dict]) -> dict:
        dataset_id = uuid.uuid4().hex[:12]
        entry = {
            "id": dataset_id,
            "name": name,
            "records": records,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._datasets[dataset_id] = entry
        return self._summary(entry)

    def get(self, dataset_id: str) -> dict | None:
        return self._datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        return self._datasets.pop(dataset_id, None) is not None

    def list(self) -> list[dict]:
        return [self._summary(entry) for entry in self._datasets.values()]

    def records(self, dataset_id: str) -> list[dict] | None:
        entry = self._datasets.get(dataset_id)
        return entry["records"] if entry else None

    @staticmethod
    def _summary(entry: dict) -> dict:
        records = entry["records"]
        models = sorted({r["model"] for r in records})
        runs = sorted({r["run"] for r in records})
        return {
            "id": entry["id"],
            "name": entry["name"],
            "uploaded_at": entry["uploaded_at"],
            "n_records": len(records),
            "models": models,
            "n_runs": len(runs),
        }


store = DatasetStore()
