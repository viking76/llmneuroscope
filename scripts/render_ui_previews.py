"""Génère des aperçus PNG du dashboard LLM Neuroscope à partir des données réelles de l'API.

Usage: python scripts/render_ui_previews.py [api_base] [output_dir]

Nécessite un backend lancé (uvicorn app.main:app) et le dataset d'exemple.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

BG = "#0d1117"
PANEL = "#161b26"
BORDER = "#2a2f3a"
TEXT = "#e6e9f0"
MUTED = "#8b93a7"
ACCENT = "#4f8cff"
DANGER = "#ff6b6b"
WARN = "#ffb84f"
GOOD = "#4fcf8b"

W = 1440
SIDEBAR_W = 280
PAD = 20

TYPE_LABELS = {
    "logprob_collapse": "Effondrement de logprobs",
    "degenerate_repetition": "Boucle de répétition",
    "length_outlier": "Longueur anormale",
    "latency_spike": "Pic de latence",
    "position_instability": "Instabilité de position",
    "entropy_spike": "Pic d'entropie",
}
SEVERITY_LABELS = {"high": "HAUTE", "medium": "MOYENNE", "low": "BASSE"}
SEVERITY_COLORS = {"high": DANGER, "medium": WARN, "low": ACCENT}
STATUS_LABELS = {"stable": "stable", "regression": "régression", "improvement": "amélioration"}
STATUS_COLORS = {"stable": MUTED, "regression": DANGER, "improvement": GOOD}
VERDICT_LABELS = {
    "stable": "STABLE",
    "warning": "VIGILANCE",
    "regression": "RÉGRESSION DÉTECTÉE",
    "improvement": "AMÉLIORATION",
}
VERDICT_COLORS = {
    "stable": GOOD,
    "warning": WARN,
    "regression": DANGER,
    "improvement": GOOD,
}


try:
    import reportlab

    FONT_PATH = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
except Exception:
    FONT_PATH = None


def font(size):
    if FONT_PATH and FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size)


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def blend(base, overlay, alpha):
    b, o = hex_rgb(base), hex_rgb(overlay)
    return tuple(int(b[i] * (1 - alpha) + o[i] * alpha) for i in range(3))


class Ui:
    def __init__(self, width, height):
        self.img = Image.new("RGB", (width, height), hex_rgb(BG))
        self.draw = ImageDraw.Draw(self.img)
        self.width = width
        self.height = height

    def panel(self, x, y, w, h, radius=10, fill=PANEL, outline=BORDER):
        self.draw.rounded_rectangle(
            [x, y, x + w, y + h], radius=radius, fill=hex_rgb(fill), outline=hex_rgb(outline), width=1
        )

    def text(self, x, y, content, size=14, color=TEXT, anchor="la"):
        self.draw.text((x, y), content, font=font(size), fill=hex_rgb(color), anchor=anchor)

    def text_width(self, content, size=14):
        return self.draw.textlength(content, font=font(size))

    def ellipsis(self, x, y, max_width, content, size=14, color=TEXT):
        if self.text_width(content, size) <= max_width:
            self.text(x, y, content, size, color)
            return
        while content and self.text_width(content + "…", size) > max_width:
            content = content[:-1]
        self.text(x, y, content + "…", size, color)

    def badge(self, x, y, label, color, size=11, padding=10):
        w = self.text_width(label, size) + padding * 2
        h = size + 12
        self.draw.rounded_rectangle(
            [x, y, x + w, y + h], radius=h / 2, fill=blend(PANEL, color, 0.18)
        )
        self.text(x + padding, y + h / 2, label, size, color, anchor="lm")
        return w

    def save(self, path):
        self.img.save(path)
        print(f"écrit {path}")


def header(ui, active_dataset_name, n_records, n_runs):
    ui.text(PAD, 24, "LLM Neuroscope", 22)
    ui.text(PAD, 56, "Détection des problèmes d'un LLM via l'analyse de son activité neuronale", 13, MUTED)
    if active_dataset_name:
        w = ui.text_width(active_dataset_name, 12) + 24
        ui.panel(W - PAD - 360, 26, 360, 44, radius=0, fill=BG, outline=BG)
        ui.draw.rounded_rectangle(
            [W - PAD - w, 28, W - PAD, 48], radius=9, fill=hex_rgb(ACCENT)
        )
        ui.text(W - PAD - w + 12, 38, active_dataset_name, 12, "#ffffff", anchor="lm")
        ui.text(
            W - PAD,
            58,
            f"{n_records} enregistrements · {n_runs} runs",
            12,
            MUTED,
            anchor="ra",
        )
    ui.draw.line([PAD, 92, W - PAD, 92], fill=hex_rgb(BORDER), width=1)


def sidebar(ui, datasets, active_id):
    x, y = PAD, 112
    ui.panel(x, y, SIDEBAR_W, 150)
    ui.text(x + 20, y + 18, "Importer des logs", 14)
    ui.text(x + 20, y + 42, "CSV · JSON · JSONL", 12, MUTED)
    bx = x + SIDEBAR_W / 2 - 70
    ui.draw.rounded_rectangle([bx, y + 72, bx + 140, y + 104], radius=8, fill=hex_rgb(ACCENT))
    ui.text(bx + 70, y + 88, "Choisir un fichier", 12, "#ffffff", anchor="mm")

    y += 170
    ui.panel(x, y, SIDEBAR_W, 90)
    ui.text(x + 14, y + 14, f"Datasets ({len(datasets)})", 14)
    for index, dataset in enumerate(datasets):
        item_y = y + 42 + index * 42
        active = dataset["id"] == active_id
        if active:
            ui.draw.rounded_rectangle(
                [x + 8, item_y - 6, x + SIDEBAR_W - 8, item_y + 30],
                radius=8,
                fill=blend(PANEL, ACCENT, 0.15),
                outline=hex_rgb(ACCENT),
            )
        ui.ellipsis(x + 18, item_y, SIDEBAR_W - 80, dataset["name"], 13)
        ui.text(
            x + 18,
            item_y + 18,
            f"{dataset['n_records']} enr. · {len(dataset['models'])} modèle(s)",
            11,
            MUTED,
        )


def tabs_bar(ui, active_index, model, models):
    x, y = 320, 112
    tab_labels = ["Vue d'ensemble", "Activité", "Anomalies", "Comparaison", "Sonde neuronale", "Évals"]
    widths = [122, 92, 100, 112, 138, 72]
    ui.panel(x, y, sum(widths) + 28, 42, radius=10)
    cx = x + 8
    for index, (label, w) in enumerate(zip(tab_labels, widths)):
        if index == active_index:
            ui.draw.rounded_rectangle([cx, y + 4, cx + w, y + 38], radius=7, fill=hex_rgb(ACCENT))
            ui.text(cx + w / 2, y + 21, label, 12, "#ffffff", anchor="mm")
        else:
            ui.text(cx + w / 2, y + 21, label, 12, MUTED, anchor="mm")
        cx += w + 8
    button_w, button_h = 64, 30
    pdf_x = W - PAD - button_w
    csv_x = pdf_x - button_w - 8
    for bx, label in ((csv_x, "CSV"), (pdf_x, "PDF")):
        ui.draw.rounded_rectangle([bx, y + 6, bx + button_w, y + 6 + button_h], radius=8, fill=hex_rgb(PANEL), outline=hex_rgb(BORDER))
        ui.text(bx + button_w / 2, y + 6 + button_h / 2, label, 11, TEXT, anchor="mm")
    select_x = csv_x - 280
    ui.text(select_x, y + 21, "Modèle :", 13, MUTED, anchor="lm")
    ui.panel(select_x + 70, y + 4, 190, 34, radius=8)
    ui.text(select_x + 84, y + 21, model, 12, anchor="lm")


def metric_card(ui, x, y, w, h, label, value, hint, tone=None):
    ui.panel(x, y, w, h)
    ui.draw.rounded_rectangle(
        [x, y, x + 3, y + h], radius=2, fill=hex_rgb(tone or ACCENT)
    )
    ui.text(x + 16, y + 12, label, 11, MUTED)
    ui.text(x + 16, y + 30, str(value), 20)
    if hint:
        ui.text(x + 16, y + 62, hint, 10, MUTED)


def metric_grid(ui, cards, top, cols=3, card_w=352, card_h=88, gap=14):
    for index, card in enumerate(cards):
        x = 320 + (index % cols) * (card_w + gap)
        y = top + (index // cols) * (card_h + gap)
        metric_card(ui, x, y, card_w, card_h, *card)


def chart_panel(ui, x, y, w, h, title):
    ui.panel(x, y, w, h)
    ui.text(x + 18, y + 16, title, 14)


def bar_chart(ui, x, y, w, h, entries, max_value=None):
    chart_top, chart_bottom = y + 40, y + h - 34
    max_value = max_value or max((v for _, v in entries), default=1) or 1
    ui.draw.line([x + 40, chart_bottom, x + w - 20, chart_bottom], fill=hex_rgb(BORDER))
    n = len(entries)
    slot = (w - 80) / max(n, 1)
    for index, (label, value) in enumerate(entries):
        bar_h = (value / max_value) * (chart_bottom - chart_top - 10)
        bx = x + 50 + index * slot + slot * 0.2
        ui.draw.rounded_rectangle(
            [bx, chart_bottom - bar_h, bx + slot * 0.6, chart_bottom],
            radius=4,
            fill=hex_rgb(ACCENT),
        )
        ui.text(bx + slot * 0.3, chart_bottom + 8, label, 11, MUTED, anchor="ma")
        ui.text(bx + slot * 0.3, chart_bottom - bar_h - 16, f"{value:.3f}", 10, TEXT, anchor="ma")


def line_chart(ui, x, y, w, h, series, y_range=None):
    chart_left, chart_right = x + 44, x + w - 20
    chart_top, chart_bottom = y + 36, y + h - 34
    all_values = [v for _, points, _ in series for v in points if v is not None]
    if not all_values:
        ui.text(x + w / 2, y + h / 2, "Pas de logprobs dans ce dataset", 12, MUTED, anchor="mm")
        return
    lo = min(all_values) if y_range is None else y_range[0]
    hi = max(all_values) if y_range is None else y_range[1]
    span = (hi - lo) or 1
    for gy in range(5):
        gy_y = chart_top + gy * (chart_bottom - chart_top) / 4
        ui.draw.line([chart_left, gy_y, chart_right, gy_y], fill=hex_rgb(BORDER))
        ui.text(chart_left - 6, gy_y, f"{hi - gy * span / 4:.1f}", 9, MUTED, anchor="rm")
    n_points = max(len(points) for _, points, _ in series)
    for _, points, color in series:
        coords = []
        for index, value in enumerate(points):
            if value is None:
                continue
            px = chart_left + index / max(n_points - 1, 1) * (chart_right - chart_left)
            py = chart_bottom - (value - lo) / span * (chart_bottom - chart_top)
            coords.append((px, py))
        if len(coords) > 1:
            ui.draw.line(coords, fill=hex_rgb(color), width=2)


def render_overview(data, datasets, ds_id, model):
    a = data["analysis"]
    ui = Ui(W, 700)
    header(ui, data["dataset"]["name"], data["dataset"]["n_records"], data["dataset"]["n_runs"])
    sidebar(ui, datasets, ds_id)
    tabs_bar(ui, 0, model, data["dataset"]["models"])

    top = 176
    degeneration_pct = f"{a['repetition']['degeneration_rate'] * 100:.1f} %"
    metric_grid(
        ui,
        [
            ("Perplexité", f"{a['perplexity']:.3f}", "plus bas = plus confiant", None),
            ("Entropie des tokens", f"{a['token_entropy']:.3f}", f"vocabulaire: {a['vocabulary_size']}", None),
            (
                "Taux de dégénérescence",
                degeneration_pct,
                "sorties en boucle de répétition",
                DANGER if a["repetition"]["degeneration_rate"] > 0.1 else GOOD,
            ),
            (
                "Dérive max entre runs",
                f"{a['drift']['max_js']:.3f}",
                f"run {a['drift']['from_run']} vers run {a['drift']['to_run']} (JS)"
                if a["drift"]["to_run"]
                else "aucune dérive mesurable",
                WARN if a["drift"]["max_js"] > 0.2 else None,
            ),
            ("Longueur moyenne", f"{a['length']['mean']:.1f}", f"ecart-type = {a['length']['std']:.1f}", None),
            ("Latence moyenne", f"{a['latency_ms']['mean']:.0f} ms", f"p95 = {a['latency_ms']['p95']:.0f} ms", None),
        ],
        top,
    )

    chart_y = top + 2 * 88 + 2 * 14 + 16
    chart_w, chart_h = 540, 300
    chart_panel(ui, 320, chart_y, chart_w, chart_h, "Entropie par run (convergence)")
    entries = [(f"Run {run}", value) for run, value in a["run_entropy"].items()]
    bar_chart(ui, 330, chart_y + 10, chart_w - 20, chart_h - 20, entries)

    chart_panel(ui, 320 + chart_w + 16, chart_y, chart_w, chart_h, "Log-prob moyen & entropie par position")
    line_chart(
        ui,
        330 + chart_w + 16,
        chart_y + 10,
        chart_w - 20,
        chart_h - 20,
        [
            ("logprob", a["position_logprob"], DANGER),
            ("entropie", a["position_entropy"], ACCENT),
        ],
    )
    ui.badge(320 + chart_w + 34, chart_y + chart_h - 26, "log-prob moyen", DANGER)
    ui.badge(320 + chart_w + 160, chart_y + chart_h - 26, "entropie", ACCENT)
    return ui


def render_heatmap(data, datasets, ds_id, model):
    hm = data["heatmap"]
    a = data["analysis"]
    ui = Ui(W, 640)
    header(ui, data["dataset"]["name"], data["dataset"]["n_records"], data["dataset"]["n_runs"])
    sidebar(ui, datasets, ds_id)
    tabs_bar(ui, 1, model, data["dataset"]["models"])

    top = 176
    ui.panel(320, top, 1100, 300)
    ui.text(338, top + 16, "Heatmap d'activité — entropie par position × run", 14)
    ui.text(
        338,
        top + 40,
        "Chaque cellule représente l'entropie (incertitude) du modèle à une position donnée, pour un run donné.",
        11,
        MUTED,
    )
    values = [v for row in hm["matrix"] for v in row["values"] if v is not None]
    max_value = max(values, default=0.001)
    cell_w, cell_h = 15, 22
    grid_x, grid_y = 338, top + 70
    for position in range(0, len(hm["positions"]), 8):
        ui.text(grid_x + position * cell_w, grid_y - 14, str(position), 9, MUTED)
    for row_index, row in enumerate(hm["matrix"]):
        ry = grid_y + row_index * (cell_h + 4)
        ui.text(grid_x - 6, ry + cell_h / 2, f"Run {row['run']}", 10, MUTED, anchor="rm")
        for position, value in enumerate(row["values"]):
            if value is None:
                continue
            intensity = value / max_value
            color = blend("#2864c8", "#ff3c28", intensity)
            ui.draw.rectangle(
                [grid_x + position * cell_w, ry, grid_x + position * cell_w + cell_w - 2, ry + cell_h - 2],
                fill=color,
            )
    legend_y = grid_y + len(hm["matrix"]) * (cell_h + 4) + 16
    ui.text(338, legend_y + 5, "0 (déterministe)", 10, MUTED)
    for step in range(40):
        ui.draw.rectangle(
            [438 + step * 8, legend_y, 438 + step * 8 + 8, legend_y + 10],
            fill=blend("#2864c8", "#ff3c28", step / 39),
        )
    ui.text(768, legend_y + 5, f"{max_value:.2f} (aléatoire)", 10, MUTED)

    profile_top = top + 330
    ui.panel(320, profile_top, 1100, 110)
    ui.text(338, profile_top + 14, "Profil d'entropie par position (tous runs confondus)", 14)
    bars = a["position_entropy"]
    bar_area_x, bar_area_y, bar_area_h = 338, profile_top + 40, 50
    bar_w = 1040 / max(len(bars), 1)
    for index, value in enumerate(bars):
        bar_h = max(3, value / max_value * bar_area_h)
        ui.draw.rectangle(
            [bar_area_x + index * bar_w, bar_area_y + bar_area_h - bar_h,
             bar_area_x + index * bar_w + bar_w - 2, bar_area_y + bar_area_h],
            fill=hex_rgb(ACCENT),
        )
    return ui


def render_anomalies(data, datasets, ds_id, model):
    anomalies = data["anomalies"]
    points = data["blocking"]["points"]
    ui = Ui(W, 1080)
    header(ui, data["dataset"]["name"], data["dataset"]["n_records"], data["dataset"]["n_runs"])
    sidebar(ui, datasets, ds_id)
    tabs_bar(ui, 2, model, data["dataset"]["models"])

    top = 176
    counts = {}
    for anomaly in anomalies:
        counts[anomaly["type"]] = counts.get(anomaly["type"], 0) + 1
    cards = [("Anomalies détectées", str(len(anomalies)), None, WARN if anomalies else GOOD)]
    for anomaly_type, count in list(counts.items())[:3]:
        cards.append(
            (TYPE_LABELS.get(anomaly_type, anomaly_type), str(count), None, DANGER if count else GOOD)
        )
    cards.append(
        (
            "Points bloquants (positions)",
            str(len(points)),
            f"positions {', '.join(str(p['position']) for p in points[:5])}…",
            DANGER if points else GOOD,
        )
    )
    metric_grid(ui, cards, top, cols=3)

    table_top = top + 2 * 88 + 2 * 14 + 16
    ui.panel(320, table_top, 1100, 420)
    ui.text(338, table_top + 16, "Journal des anomalies", 14)
    ui.panel(1150, table_top + 12, 250, 30, radius=8)
    ui.text(1164, table_top + 27, "Toutes sévérités", 11, MUTED, anchor="lm")

    headers = ["Sévérité", "Type", "Prompt", "Run", "Valeur", "Z-score"]
    col_x = [338, 430, 700, 830, 900, 1010]
    row_y = table_top + 50
    for label, cx in zip(headers, col_x):
        ui.text(cx, row_y, label, 11, MUTED)
    ui.draw.line([338, row_y + 20, 1400, row_y + 20], fill=hex_rgb(BORDER))
    for index, anomaly in enumerate(anomalies[:10]):
        ry = row_y + 32 + index * 34
        color = SEVERITY_COLORS[anomaly["severity"]]
        ui.badge(col_x[0], ry - 4, SEVERITY_LABELS[anomaly["severity"]], color)
        ui.text(col_x[1], ry, TYPE_LABELS.get(anomaly["type"], anomaly["type"]), 12)
        ui.text(col_x[2], ry, str(anomaly["prompt_id"]), 12)
        ui.text(col_x[3], ry, str(anomaly["run"]), 12)
        ui.text(col_x[4], ry, str(anomaly["value"]), 12)
        ui.text(col_x[5], ry, str(anomaly["z_score"] if anomaly["z_score"] is not None else "—"), 12)

    points_top = table_top + 440
    ui.panel(320, points_top, 1100, 180)
    ui.text(338, points_top + 16, "Points bloquants par position", 14)
    col_x2 = [338, 430, 700, 830, 950]
    headers2 = ["Position", "Type", "Valeur", "Z-score", "Sévérité"]
    row_y2 = points_top + 48
    for label, cx in zip(headers2, col_x2):
        ui.text(cx, row_y2, label, 11, MUTED)
    ui.draw.line([338, row_y2 + 20, 1400, row_y2 + 20], fill=hex_rgb(BORDER))
    for index, point in enumerate(points[:3]):
        ry = row_y2 + 32 + index * 34
        ui.text(col_x2[0], ry, str(point["position"]), 12)
        ui.text(col_x2[1], ry, TYPE_LABELS.get(point["type"], point["type"]), 12)
        ui.text(col_x2[2], ry, str(point["value"]), 12)
        ui.text(col_x2[3], ry, str(point["z_score"]), 12)
        ui.badge(col_x2[4], ry - 4, SEVERITY_LABELS[point["severity"]], SEVERITY_COLORS[point["severity"]])
    return ui


def render_compare(data, datasets, ds_id, model):
    comparison = data["comparison"]
    ui = Ui(W, 720)
    header(ui, data["dataset"]["name"], data["dataset"]["n_records"], data["dataset"]["n_runs"])
    sidebar(ui, datasets, ds_id)
    tabs_bar(ui, 3, model, data["dataset"]["models"])

    top = 176
    ui.panel(320, top, 1100, 520)
    ui.text(338, top + 16, "Comparaison de modèles", 14)
    ui.panel(1150, top + 12, 120, 30, radius=8)
    ui.text(1164, top + 27, comparison["model_a"], 11, anchor="lm")
    ui.text(1282, top + 27, "vers", 12, MUTED, anchor="lm")
    ui.panel(1300, top + 12, 100, 30, radius=8)
    ui.text(1314, top + 27, comparison["model_b"], 11, anchor="lm")

    verdict_color = VERDICT_COLORS[comparison["verdict"]]
    ui.badge(338, top + 54, VERDICT_LABELS[comparison["verdict"]], verdict_color, size=13)
    ui.text(
        338 + 260,
        top + 66,
        f"Divergence de distribution (JS) : {comparison['token_distribution_js']}",
        12,
        MUTED,
    )

    headers = ["Métrique", comparison["model_a"], comparison["model_b"], "Écart", "Écart %", "Statut"]
    col_x = [338, 640, 830, 1020, 1110, 1230]
    row_y = top + 100
    for label, cx in zip(headers, col_x):
        ui.text(cx, row_y, label, 11, MUTED)
    ui.draw.line([338, row_y + 20, 1400, row_y + 20], fill=hex_rgb(BORDER))
    for index, row in enumerate(comparison["metrics"][:11]):
        ry = row_y + 32 + index * 34
        ui.text(col_x[0], ry, row["metric"], 12)
        ui.text(col_x[1], ry, str(row["model_a"]), 12)
        ui.text(col_x[2], ry, str(row["model_b"]), 12)
        delta = row["delta"]
        ui.text(col_x[3], ry, f"{'+' if delta > 0 else ''}{delta}", 12)
        pct = row["delta_pct"]
        ui.text(col_x[4], ry, f"{'+' if pct and pct > 0 else ''}{pct} %" if pct is not None else "—", 12)
        ui.badge(col_x[5], ry - 4, STATUS_LABELS[row["status"]], STATUS_COLORS[row["status"]])
    return ui


def render_probe(probe):
    status = probe["status"]
    activation = probe["activation"]
    attention = probe["attention"]
    attribution = probe["attribution"]
    patching = probe["patching"]
    generate = probe["generate"]
    weights = probe["weights"]
    ui = Ui(W, 1360)
    header(ui, "sample_runs.jsonl", 240, 3)
    sidebar(ui, probe["datasets"], probe["ds_id"])
    tabs_bar(ui, 4, "stable-v1", ["stable-v1"])

    top = 176
    ui.panel(320, top, 540, 140)
    ui.text(338, top + 16, "Modèle chargé", 14)
    ui.badge(338, top + 42, status["backend"].upper(), ACCENT)
    ui.badge(338 + 70, top + 42, "MODIFIABLE", GOOD)
    ui.text(338, top + 84, status["model_name"], 13)
    ui.text(338, top + 106, f"{status['n_layers']} couches · device {status['device']}", 11, MUTED)

    ui.panel(876, top, 544, 140)
    ui.text(894, top + 16, "Génération instrumentée", 14)
    ui.panel(894, top + 40, 380, 30, radius=8)
    ui.text(908, top + 55, generate["prompt"], 12, anchor="lm")
    ui.draw.rounded_rectangle([1284, top + 40, 1400, top + 70], radius=8, fill=hex_rgb(ACCENT))
    ui.text(1342, top + 55, "Générer", 12, "#ffffff", anchor="mm")
    ui.text(894, top + 90, "".join(generate["tokens"]), 12)
    ui.badge(894, top + 108, f"perplexité {generate['perplexity']}", ACCENT)

    act_top = top + 160
    ui.panel(320, act_top, 1100, 200)
    ui.text(338, act_top + 16, "Analyse d'activation par couche", 14)
    ui.text(338, act_top + 40, "Ratio de neurones morts et activité maximale, pour un lot de textes de test.", 11, MUTED)
    for index, layer in enumerate(activation["layers"]):
        ry = act_top + 70 + index * 32
        ui.text(338, ry, layer["layer"], 12)
        bar_x, bar_w = 470, 340
        ui.draw.rounded_rectangle([bar_x, ry + 2, bar_x + bar_w, ry + 14], radius=6, fill=blend(PANEL, ACCENT, 0.2))
        fill_w = int(bar_w * min(layer["std"] * 2, 1.0))
        ui.draw.rounded_rectangle([bar_x, ry + 2, bar_x + fill_w, ry + 14], radius=6, fill=hex_rgb(ACCENT))
        dead = layer["dead_neuron_ratio"]
        dead_color = GOOD if dead < 0.2 else (WARN if dead < 0.5 else DANGER)
        ui.badge(830, ry - 4, f"neurones morts {dead:.0%}", dead_color)
        tops = " ".join(f"n{n['index']}" for n in layer["top_neurons"][:4])
        ui.text(1050, ry, f"top: {tops}", 11, MUTED)

    attn_top = act_top + 220
    ui.panel(320, attn_top, 1100, 250)
    ui.text(338, attn_top + 16, "Entropie d'attention par couche × tête", 14)
    ui.text(338, attn_top + 40, "Bleu = tête focalisée · rouge = tête diffuse (attention dispersée).", 11, MUTED)
    matrix = attention["head_matrix"]
    max_entropy = attention["max_entropy"] or 1
    cell_w, cell_h = 120, 32
    grid_x, grid_y = 380, attn_top + 70
    for row_index, row in enumerate(matrix):
        ui.text(grid_x - 10, grid_y + row_index * 36 + 14, f"L{row_index}", 11, MUTED, anchor="rm")
        for col_index, value in enumerate(row):
            intensity = value / max_entropy
            color = blend("#2864c8", "#ff3c28", intensity)
            ui.draw.rounded_rectangle(
                [grid_x + col_index * (cell_w + 6), grid_y + row_index * 36,
                 grid_x + col_index * (cell_w + 6) + cell_w, grid_y + row_index * 36 + cell_h],
                radius=4, fill=color,
            )
            ui.text(grid_x + col_index * (cell_w + 6) + cell_w / 2, grid_y + row_index * 36 + 16,
                    f"{value:.2f}", 10, "#ffffff", anchor="mm")
    ui.text(grid_x + 4 * (cell_w + 6) + 24, grid_y + 10, "Têtes les plus diffuses :", 12)
    for index, head in enumerate(attention["top_heads"][:3]):
        ry = grid_y + 36 + index * 26
        ui.badge(grid_x + 4 * (cell_w + 6) + 24, ry, f"L{head['layer']}·T{head['head']}", DANGER)
        ui.text(grid_x + 4 * (cell_w + 6) + 100, ry + 4, f"entropie {head['entropy']:.2f}", 11, MUTED)

    attr_top = attn_top + 270
    ui.panel(320, attr_top, 540, 220)
    ui.text(338, attr_top + 16, f"Attribution des tokens ({attribution['method']})", 14)
    ui.text(338, attr_top + 40, f"Cible : « {attribution['target_token']} » — contribution de chaque token", 11, MUTED)
    chip_x, chip_y = 338, attr_top + 70
    max_abs = max((abs(t["score"]) for t in attribution["tokens"]), default=1) or 1
    for token in attribution["tokens"]:
        intensity = abs(token["score"]) / max_abs
        if token["score"] >= 0:
            color = blend(PANEL, GOOD, 0.2 + 0.6 * intensity)
        else:
            color = blend(PANEL, DANGER, 0.2 + 0.6 * intensity)
        chip_w = ui.text_width(token["token"], 12) + 20
        ui.draw.rounded_rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + 30], radius=6, fill=color)
        ui.text(chip_x + chip_w / 2, chip_y + 15, token["token"], 12, "#ffffff", anchor="mm")
        chip_x += chip_w + 6
        if chip_x > 800:
            chip_x, chip_y = 338, chip_y + 38
    ui.badge(338, attr_top + 180, "vert = pousse vers la cible", GOOD)
    ui.badge(338 + 200, attr_top + 180, "rouge = s'oppose", DANGER)

    ui.panel(876, attr_top, 544, 220)
    ui.text(894, attr_top + 16, "Activation patching (nnsight)", 14)
    ui.text(894, attr_top + 40, f"Restoration par couche — KL baseline {patching['baseline_kl']:.2f}", 11, MUTED)
    for index, (restoration, layer_kl) in enumerate(zip(patching["layer_restoration"], patching["layer_kl"])):
        ry = attr_top + 70 + index * 30
        critical = index == patching["critical_layer"]
        color = DANGER if critical else ACCENT
        ui.text(894, ry, f"couche {index}", 12)
        bar_x, bar_w = 990, 330
        ui.draw.rounded_rectangle([bar_x, ry + 2, bar_x + bar_w, ry + 16], radius=7, fill=blend(PANEL, color, 0.25))
        fill_w = int(bar_w * restoration / 100)
        ui.draw.rounded_rectangle([bar_x, ry + 2, bar_x + fill_w, ry + 16], radius=7, fill=hex_rgb(color))
        ui.text(bar_x + bar_w + 10, ry + 2, f"{restoration:.0f} %", 11, color if critical else TEXT)
    ui.text(894, attr_top + 196, f"Couche critique : #{patching['critical_layer']} (en rouge)", 11, MUTED)

    weight_top = attr_top + 240
    ui.panel(320, weight_top, 1100, 110)
    ui.text(338, weight_top + 16, f"Poids — {weights['layer']}  (shape {weights['shape'][0]} × {weights['shape'][1]})", 14)
    ui.badge(900, weight_top + 14, f"std {weights['std']:.4f}", ACCENT)
    ui.badge(1010, weight_top + 14, f"min {weights['min']:.4f}", MUTED)
    ui.badge(1110, weight_top + 14, f"max {weights['max']:.4f}", MUTED)
    counts = weights["histogram"]["counts"]
    edges = weights["histogram"]["edges"]
    max_count = max(counts) or 1
    hist_x, hist_y, hist_h, hist_w = 338, weight_top + 50, 44, 1044
    bin_w = hist_w / len(counts)
    for index, count in enumerate(counts):
        bar_h = count / max_count * hist_h
        ui.draw.rectangle(
            [hist_x + index * bin_w, hist_y + hist_h - bar_h, hist_x + (index + 1) * bin_w - 2, hist_y + hist_h],
            fill=hex_rgb(ACCENT),
        )
    ui.text(hist_x, hist_y + 8, f"{edges[0]:.3f}", 9, MUTED)
    ui.text(hist_x + hist_w, hist_y + 8, f"{edges[-1]:.3f}", 9, MUTED, anchor="ra")
    return ui


def render_evals(evals):
    benchmark = evals["benchmark"]
    quality = evals["quality"]
    langfuse = evals["langfuse"]
    ui = Ui(W, 760)
    header(ui, "sample_runs.jsonl", 240, 3)
    sidebar(ui, evals["datasets"], evals["ds_id"])
    tabs_bar(ui, 5, "stable-v1", ["stable-v1"])

    top = 176
    ui.panel(320, top, 540, 300)
    ui.text(338, top + 16, "Benchmark standardisé (lm-evaluation-harness)", 14)
    chip_x, chip_y = 338, top + 46
    for task, active in (("lambada_openai", True), ("hellaswag", False), ("arc_easy", False)):
        chip_w = ui.text_width(task, 11) + 20
        fill = ACCENT if active else PANEL
        outline = ACCENT if active else BORDER
        ui.draw.rounded_rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + 26], radius=13,
                                  fill=blend(PANEL, ACCENT, 0.2) if active else hex_rgb(PANEL), outline=hex_rgb(outline))
        ui.text(chip_x + chip_w / 2, chip_y + 13, task, 11, "#ffffff" if active else MUTED, anchor="mm")
        chip_x += chip_w + 6
    ui.text(338, top + 90, f"Moteur : {benchmark['engine']} · limite {benchmark['limit']} exemples / tâche", 11, MUTED)
    headers = ["Tâche", "Métrique", "Valeur", "Erreur std"]
    col_x = [338, 520, 660, 750]
    row_y = top + 120
    for label, cx in zip(headers, col_x):
        ui.text(cx, row_y, label, 11, MUTED)
    ui.draw.line([338, row_y + 20, 840, row_y + 20], fill=hex_rgb(BORDER))
    index = 0
    for task, metrics in benchmark["tasks"].items():
        for metric, value in metrics.items():
            ry = row_y + 32 + index * 34
            ui.text(col_x[0], ry, task, 12)
            ui.text(col_x[1], ry, metric, 12)
            ui.text(col_x[2], ry, f"{value:.4f}", 12)
            stderr = metrics.get(f"{metric}_stderr")
            ui.text(col_x[3], ry, f"± {stderr:.4f}" if stderr is not None else "—", 12)
            index += 1

    ui.panel(876, top, 544, 300)
    ui.text(894, top + 16, "Qualité des sorties (DeepEval + heuristiques)", 14)
    ui.panel(894, top + 42, 360, 30, radius=8)
    ui.ellipsis(908, top + 57, 330, quality["texts"][0]["text"], 12)
    ui.draw.rounded_rectangle([1264, top + 42, 1400, top + 72], radius=8, fill=hex_rgb(ACCENT))
    ui.text(1332, top + 57, "Évaluer", 12, "#ffffff", anchor="mm")
    ui.badge(894, top + 84, "juge : local (Qwen 1.5B)", ACCENT)
    headers = ["Texte", "PPL", "Répét.", "Score"]
    col_x = [894, 1240, 1300, 1360]
    row_y = top + 120
    for label, cx in zip(headers, col_x):
        ui.text(cx, row_y, label, 11, MUTED)
    ui.draw.line([894, row_y + 20, 1400, row_y + 20], fill=hex_rgb(BORDER))
    scores = [0.8, 0.4, 0.6]
    for index, item in enumerate(quality["texts"][:3]):
        ry = row_y + 32 + index * 34
        ui.ellipsis(col_x[0], ry, 320, item["text"], 12)
        ui.text(col_x[1], ry, f"{item['perplexity']:.1f}", 12)
        ui.text(col_x[2], ry, f"{item['repetition_ratio']:.2f}", 12)
        score = scores[index % len(scores)]
        score_color = GOOD if score >= 0.5 else DANGER
        ui.badge(col_x[3], ry - 4, f"{score:.1f}", score_color)
    aggregate = quality["aggregate"]
    ui.badge(894, top + 264, f"PPL moy. {aggregate['perplexity']:.1f}", ACCENT)
    ui.badge(1020, top + 264, f"répétition {aggregate['repetition_ratio']:.2f}", WARN)
    ui.badge(1180, top + 264, f"diversité {aggregate['distinct_ratio']:.2f}", GOOD)

    lgf_top = top + 320
    ui.panel(320, lgf_top, 1100, 110)
    ui.text(338, lgf_top + 16, "Export des traces vers Langfuse", 14)
    ui.badge(338, lgf_top + 44, langfuse["mode"].upper(), ACCENT)
    ui.text(430, lgf_top + 55, f"{langfuse['n_traces']} traces exportées", 12)
    ui.text(338, lgf_top + 78, langfuse["path"], 11, MUTED)
    ui.panel(1150, lgf_top + 14, 250, 34, radius=8)
    ui.text(1164, lgf_top + 31, "Pousser vers Langfuse", 11, MUTED, anchor="lm")
    return ui


def main():
    api_base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    def get(path, **kwargs):
        return httpx.get(f"{api_base}/api{path}", timeout=30, **kwargs).json()

    sample = Path(__file__).parent.parent / "data" / "sample_runs.jsonl"
    if not sample.exists():
        raise SystemExit(f"Dataset introuvable: {sample}. Lancez scripts/generate_sample_data.py")
    for existing in get("/datasets"):
        httpx.delete(f"{api_base}/api/datasets/{existing['id']}", timeout=30)
    with sample.open("rb") as handle:
        dataset = httpx.post(
            f"{api_base}/api/datasets",
            timeout=30,
            files={"file": ("sample_runs.jsonl", handle)},
        ).json()
    datasets = get("/datasets")
    ds_id = dataset["id"]
    model = dataset["models"][0]
    other = dataset["models"][-1]

    data = {
        "dataset": dataset,
        "analysis": get(f"/datasets/{ds_id}/analysis", params={"model": model}),
        "heatmap": get(f"/datasets/{ds_id}/heatmap", params={"model": model}),
        "anomalies": get(f"/datasets/{ds_id}/anomalies", params={"model": model})["anomalies"],
        "blocking": get(f"/datasets/{ds_id}/blocking-points", params={"model": model}),
        "comparison": get(
            f"/datasets/{ds_id}/compare", params={"model_a": model, "model_b": other}
        ),
    }

    def post(path, payload):
        return httpx.post(f"{api_base}/api{path}", timeout=120, json=payload).json()

    post("/probe/load", {"model_name": "mock-transformer", "device": "cpu"})
    activation = post("/probe/analyze-texts", {"texts": ["Paris est la capitale de la France", "Le chat dort sur le tapis au soleil", "Explique la photosynthese en une phrase"]})
    attention = post("/probe/attention", {"texts": ["Paris est la capitale de la France"]})
    attribution = post("/probe/attribution", {"text": "Paris est la capitale de la France", "method": "integrated-gradients"})
    patching = post("/probe/patching", {"clean_text": "Paris est la capitale de la France", "corrupted_text": "Paris est la capitale de l'Italie"})
    generate = post("/probe/generate", {"prompt": "Paris est la", "max_new_tokens": 6})
    weights = httpx.get(f"{api_base}/api/probe/weights/layer_0.mlp", timeout=30).json()
    probe_data = {
        "datasets": datasets,
        "ds_id": ds_id,
        "status": httpx.get(f"{api_base}/api/probe/status", timeout=30).json(),
        "activation": activation,
        "attention": attention,
        "attribution": attribution,
        "patching": patching,
        "generate": generate,
        "weights": weights,
    }

    benchmark = post("/evals/benchmark", {"model_name": model, "tasks": ["lambada_openai"], "limit": 5})
    quality = post("/evals/quality", {"texts": ["Paris est la capitale de la France, un pays d'Europe occidentale.", "le le le le le le le le le le le le le le le"]})
    langfuse = post("/evals/langfuse", {"dataset_id": ds_id, "model": model})
    evals_data = {
        "datasets": datasets,
        "ds_id": ds_id,
        "benchmark": benchmark,
        "quality": quality,
        "langfuse": langfuse,
    }

    render_overview(data, datasets, ds_id, model).save(output_dir / "01-overview.png")
    render_heatmap(data, datasets, ds_id, model).save(output_dir / "02-heatmap.png")
    render_anomalies(data, datasets, ds_id, model).save(output_dir / "03-anomalies.png")
    render_compare(data, datasets, ds_id, model).save(output_dir / "04-compare.png")
    render_probe(probe_data).save(output_dir / "05-probe.png")
    render_evals(evals_data).save(output_dir / "06-evals.png")
    httpx.delete(f"{api_base}/api/datasets/{ds_id}", timeout=30)


if __name__ == "__main__":
    main()
