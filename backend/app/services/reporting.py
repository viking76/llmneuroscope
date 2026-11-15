from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services.metrics import analyze_model, detect_anomalies, detect_blocking_points
from app.services.storage import store

METRIC_LABELS = {
    "token_entropy": "Entropie de tokens (bits)",
    "vocabulary_size": "Taille du vocabulaire",
    "perplexity": "Perplexité",
}


def build_report(dataset_id: str, model: str | None) -> dict:
    records = store.records(dataset_id)
    if records is None:
        raise LookupError("Dataset introuvable.")
    models = sorted({record["model"] for record in records})
    if model is None:
        model = models[0]
    if model not in models:
        raise LookupError(f"Modèle inconnu: {model}")
    return {
        "model": model,
        "analysis": analyze_model(records, model),
        "blocking_points": detect_blocking_points(records, model),
        "anomalies": detect_anomalies(records, model),
    }


def _summary_rows(report: dict) -> list[tuple[str, str]]:
    analysis = report["analysis"]
    rows = [
        ("Modèle", analysis["model"]),
        ("Records", str(analysis["n_records"])),
        ("Runs", str(analysis["n_runs"])),
    ]
    for key, label in METRIC_LABELS.items():
        value = analysis.get(key)
        rows.append((label, f"{value:.4f}" if isinstance(value, float) else str(value)))
    for key in ("logprob", "length", "latency_ms"):
        stats = analysis.get(key)
        if stats:
            rows.append((key, f"moyenne {stats['mean']:.4f} · σ {stats['std']:.4f}"))
    repetition = analysis.get("repetition") or {}
    rows.append(("Répétition (ratio moyen)", f"{repetition.get('mean_ratio', 0):.4f}"))
    drift = analysis.get("drift") or {}
    rows.append(("Dérive max (JS)", f"{drift.get('max_js', 0):.4f}"))
    return rows


def export_csv(report: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["section", "cle", "valeur"])
    for key, value in _summary_rows(report):
        writer.writerow(["resume", key, value])
    writer.writerow([])
    writer.writerow(["point_bloquant", "position", "type", "valeur", "z_score", "severite"])
    for point in report["blocking_points"]:
        writer.writerow(
            ["point_bloquant", point["position"], point["type"], point["value"], point["z_score"], point["severity"]]
        )
    writer.writerow([])
    writer.writerow(["anomalie", "type", "severite", "prompt_id", "run", "valeur"])
    for anomaly in report["anomalies"]:
        writer.writerow(
            [
                "anomalie",
                anomaly["type"],
                anomaly["severity"],
                anomaly.get("prompt_id", ""),
                anomaly.get("run", ""),
                anomaly.get("value", ""),
            ]
        )
    return buffer.getvalue()


def export_pdf(report: dict) -> bytes:
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"], spaceBefore=12)
    muted_style = ParagraphStyle("muted", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    story = [
        Paragraph("LLM Neuroscope — Rapport d'analyse", styles["Title"]),
        Paragraph(
            f"Modèle {report['model']} · généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')}",
            muted_style,
        ),
        Spacer(1, 10),
    ]

    summary_data = [[key, value] for key, value in _summary_rows(report)]
    summary_table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6fa")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story += [Paragraph("Résumé des métriques", heading_style), summary_table]

    blocking_data = [["Position", "Type", "Valeur", "Z-score", "Gravité"]] + [
        [str(point["position"]), point["type"], str(point["value"]), str(point["z_score"]), point["severity"]]
        for point in report["blocking_points"][:20]
    ]
    blocking_table = Table(blocking_data, colWidths=[2.2 * cm, 5.4 * cm, 2.6 * cm, 2.6 * cm, 3.2 * cm])
    blocking_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story += [
        Paragraph(f"Points bloquants ({len(report['blocking_points'])})", heading_style),
        blocking_table,
    ]

    anomaly_data = [["Type", "Gravité", "Prompt", "Run", "Valeur"]] + [
        [
            anomaly["type"],
            anomaly["severity"],
            str(anomaly.get("prompt_id", "")),
            str(anomaly.get("run", "")),
            str(anomaly.get("value", "")),
        ]
        for anomaly in report["anomalies"][:20]
    ]
    anomaly_table = Table(anomaly_data, colWidths=[4.6 * cm, 2.2 * cm, 2.6 * cm, 1.6 * cm, 5 * cm])
    anomaly_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story += [
        Paragraph(f"Anomalies ({len(report['anomalies'])} — 20 premières)", heading_style),
        anomaly_table,
    ]

    document.build(story)
    return buffer.getvalue()
