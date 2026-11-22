<div align="center">

# 🔬 LLM Neuroscope

**Détecter les problèmes d'un LLM en observant son activité neuronale — et intervenir dessus.**

Convergence · Dérive · Anomalies · Points bloquants · Poids & activations

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/Tests-31%2F31-success)
![License](https://img.shields.io/badge/Licence-à%20définir-orange)

</div>

---

## 📸 Aperçu

| Vue d'ensemble | Activité neuronale | Anomalies & points bloquants |
|:---:|:---:|:---:|
| ![Vue d'ensemble](docs/screenshots/01-overview.png) | ![Heatmap](docs/screenshots/02-heatmap.png) | ![Anomalies](docs/screenshots/03-anomalies.png) |
| **Comparaison de modèles** | **Sonde neuronale** | **Évals & benchmarks** |
| ![Comparaison](docs/screenshots/04-compare.png) | ![Sonde neuronale](docs/screenshots/05-probe.png) | ![Évals](docs/screenshots/06-evals.png) |

## ✨ Fonctionnalités

### 📊 Analyse statistique de logs (CSV / JSON / JSONL)

| Capacité | Détail |
|---|---|
| **Ingestion universelle** | Détection automatique des colonnes et alias (`model`, `tokens`, `logprobs`, `run`…) |
| **Convergence** | Entropie par run, dérive inter-runs (divergence de Jensen-Shannon) |
| **Santé du modèle** | Perplexité, entropie des tokens, vocabulaire, longueur, latence (p95) |
| **Dégénérescence** | Détection des boucles de répétition par n-grammes |
| **Points bloquants** | Effondrement de logprobs, instabilité de position, pics d'entropie — par position |
| **Anomalies** | Z-scores + sévérité (haute / moyenne / basse) sur logprobs, longueurs, latences |
| **Comparaison de versions** | Deltas métrique par métrique + verdict automatique (stable / vigilance / régression / amélioration) |

### 🧠 Sonde neuronale (modèles open-weights locaux)

| Capacité | Endpoint |
|---|---|
| Charger un modèle local (HF ou GGUF `repo/fichier.gguf`) | `POST /api/probe/load` |
| Tester avec des textes → activité par couche (neurones morts, top neurones) | `POST /api/probe/analyze-texts` |
| Inspecter les poids (stats, histogramme, poids morts) | `GET /api/probe/weights/{layer}` |
| Modifier un neurone (×facteur, mise à zéro) — **réversible** | `POST /api/probe/weights/modify` |
| Exporter / réappliquer les modifications | `POST /api/probe/weights/export` · `POST /api/probe/weights/apply` |
| **Entropie d'attention** par couche × tête (TransformerLens) | `POST /api/probe/attention` |
| **Attribution des tokens** — Integrated Gradients / GradientSHAP (Captum) | `POST /api/probe/attribution` |
| **Activation patching** — restauration couche par couche (nnsight) | `POST /api/probe/patching` |
| **Benchmarks standardisés** — lambada, hellaswag, arc_easy… (lm-eval) | `POST /api/evals/benchmark` |
| **Évals de qualité** — perplexité, répétition, diversité (+ juge G-Eval optionnel) | `POST /api/evals/quality` |
| **Export Langfuse** — traces d'inférence (push ou JSONL ingestion) | `POST /api/evals/langfuse` |
| Génération instrumentée (tokens + logprobs + perplexité) | `POST /api/probe/generate` |
| **Export du rapport** CSV / PDF (résumé, points bloquants, anomalies) | `GET /api/datasets/{id}/report.csv` · `GET /api/datasets/{id}/report.pdf` |

> ⚠️ **Important** : les poids et activations ne sont accessibles que pour un modèle
> open-weights exécuté localement (GPT-2, Pythia, TinyLlama, Mistral-7B…). Les APIs
> fermées (Mistral API, OpenAI…) n'exposent ni poids ni activations — pour elles,
> seule l'analyse de logs s'applique.

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Logs CSV / JSON / JSONL] --> B[API FastAPI]
    C[Textes de test] --> D[Sonde neuronale]
    D --> E[Modèle open-weights local\ntransformers + torch]
    D <--> F[safetensors\nexport / apply des poids]
    B --> G[Moteur de métriques\nentropie · JS · z-scores · n-grammes]
    G --> H[Dashboard React\nRecharts]
    D --> H
```

```
backend/              API FastAPI
├── app/services/     ingestion · metrics · comparison · probe (mock + torch)
├── app/routers/      datasets · analysis · probe
└── tests/            31 tests pytest
frontend/             Dashboard React + Vite + Recharts (5 onglets)
scripts/              generate_sample_data.py · render_ui_previews.py
data/                 sample_runs.jsonl · weight-edits/ (exports safetensors)
docs/                 agent-api.md (pilotage par IA) · screenshots/
```

## 🚀 Démarrage rapide

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (autre terminal)
cd frontend
npm install
npm run dev        # → http://localhost:3000

# Dataset de démonstration (2 modèles : un sain, un dégradé)
python scripts/generate_sample_data.py
```

Pour la sonde neuronale avec un vrai modèle :

```bash
pip install -r backend/requirements-probe.txt   # torch + transformers + safetensors
# puis onglet « Neurones & poids » → charger gpt2, EleutherAI/pythia-70m…
```

## 🤖 Pilotage par une autre IA

LLM Neuroscope est conçu pour être **piloté par un agent** : toutes les fonctions
sont des endpoints JSON documentés dans **[docs/agent-api.md](docs/agent-api.md)**
(session type : charger → tester avec textes → lire les poids → modifier →
mesurer → restaurer, avec bonnes pratiques).

## 🧪 Tests

```bash
cd backend && python -m pytest tests -q    # 31 tests
cd frontend && npm run build               # build Vite
python scripts/render_ui_previews.py       # aperçus PNG depuis l'API réelle
```

Validé en conditions réelles : `hf-internal-testing/tiny-random-gpt2` (~5 Mo) et
`EleutherAI/pythia-70m` (entraîné) — modification d'un neurone → effet mesuré sur
la génération → restauration bit-à-bit.

## 🧩 Écosystème & intégrations candidates

Veille sur l'écosystème open source — bibliothèques repérées pour enrichir l'outil :

| Bibliothèque | Ce qu'elle apporterait | Priorité |
|---|---|---|
| [safetensors](https://github.com/huggingface/safetensors) (Hugging Face) | ✅ **Intégré** — export/import des poids modifiés, format sûr et standard | fait |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | ✅ **Intégré** — entropie d'attention par couche × tête via `TransformerBridge` | fait |
| [nnsight](https://github.com/nnsight/nnsight) | ✅ **Intégré** — activation patching couche par couche via `TransformersModel` | fait |
| [Captum](https://github.com/pytorch/captum) (PyTorch) | ✅ **Intégré** — attribution de tokens : Integrated Gradients + GradientSHAP | fait |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | ✅ **Intégré** — benchmarks standardisés (lambada_openai, hellaswag, arc_easy, winogrande, piqa) | fait |
| [DeepEval](https://github.com/confident-ai/deepeval) | ✅ **Intégré** — évals de qualité : métriques internes (perplexité, répétition, diversité) + juge LLM au choix : **local** (Qwen2.5-1.5B-Instruct via `BaseMetric`, téléchargé au 1er usage) ou cloud G-Eval (`OPENAI_API_KEY`) | fait |
| [Langfuse](https://github.com/langfuse/langfuse) | ✅ **Intégré** — export des traces d'inférence : push SDK ou JSONL compatible ingestion | fait |

## 🗺️ Feuille de route

- [x] Entropie d'attention par couche × tête (via TransformerLens)
- [x] Activation patching couche par couche (via nnsight)
- [x] Attribution de tokens — Integrated Gradients / GradientSHAP (via Captum)
- [x] Benchmarks standardisés (via lm-evaluation-harness)
- [x] Évals de qualité avec juge optionnel (via DeepEval)
- [x] Export des traces vers Langfuse
- [x] Export PDF/CSV des rapports d'analyse
- [ ] Seuils d'alerte configurables par modèle
- [x] Support GGUF (chargement via transformers, déquantization à la volée)
- [ ] Mode multi-modèles (>2) dans la comparaison

---

<div align="center">
<sub>Construit avec FastAPI · React · Recharts · safetensors — les experts : 1 architecte ML, 1 dev backend, 1 dev frontend 🤖</sub>
</div>
