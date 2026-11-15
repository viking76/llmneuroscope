import json

import pytest

from app.services.reporting import build_report, export_csv, export_pdf
from app.services.storage import store


def _rows():
    rows = []
    for run in (1, 2, 3):
        for i in range(4):
            rows.append(
                {
                    "model": "m1",
                    "prompt_id": f"p{i}",
                    "run": run,
                    "tokens": ["a", "b", "c", "d"][: 2 + (i % 3)],
                    "logprobs": [-0.1 - 0.1 * i - 0.05 * run] * (2 + (i % 3)),
                    "latency_ms": 100.0 + 10 * i,
                }
            )
    return rows


@pytest.fixture()
def dataset_id():
    return store.save("rapport-tests.jsonl", _rows())["id"]


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_build_report_defaults_to_first_model(dataset_id):
    report = build_report(dataset_id, None)
    assert report["model"] == "m1"
    assert report["analysis"]["n_records"] == 12
    assert isinstance(report["blocking_points"], list)
    assert isinstance(report["anomalies"], list)


def test_build_report_unknown_dataset():
    with pytest.raises(LookupError):
        build_report("inconnu", None)


def test_build_report_unknown_model(dataset_id):
    with pytest.raises(LookupError):
        build_report(dataset_id, "autre-modele")


def test_export_csv_sections(dataset_id):
    csv_text = export_csv(build_report(dataset_id, "m1"))
    assert csv_text.startswith("section,cle,valeur")
    assert "resume" in csv_text
    assert "Perplexit" in csv_text
    assert "point_bloquant,position,type,valeur,z_score,severite" in csv_text
    assert "anomalie,type,severite,prompt_id,run,valeur" in csv_text


def test_export_pdf_header(dataset_id):
    pdf_bytes = export_pdf(build_report(dataset_id, "m1"))
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_api_report_csv(client, dataset_id):
    response = client.get(f"/api/datasets/{dataset_id}/report.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "m1" in response.text


def test_api_report_pdf(client, dataset_id):
    response = client.get(f"/api/datasets/{dataset_id}/report.pdf?model=m1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_api_report_unknown_model_returns_404(client, dataset_id):
    response = client.get(f"/api/datasets/{dataset_id}/report.csv?model=autre")
    assert response.status_code == 404


def test_api_report_unknown_dataset_returns_404(client):
    response = client.get("/api/datasets/inconnu/report.pdf")
    assert response.status_code == 404
