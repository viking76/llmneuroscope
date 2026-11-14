# 🔬 LLM Neuroscope

Outil d'analyse statistique pour détecter les problèmes d'un LLM via l'étude de son
« activité neuronale » : convergence entre runs, dérive, anomalies et points bloquants,
avec un dashboard graphique complet.

## Fonctionnalités

- **Import de données locales** : CSV, JSON, JSONL (logs d'inférence, traces, benchmarks)
  avec détection automatique des colonnes (`model`, `tokens`, `logprobs`, `latency_ms`, `run`, `prompt_id`…).
- **Métriques d'activité neuronale** :
  - entropie des tokens (globale et par position), taille de vocabulaire
  - perplexité et statistiques de log-probs (moyenne, σ, p05, p95)
  - taux de dégénérescence (boucles de répétition), ratio de répétition
  - dérive entre runs (divergence de Jensen-Shannon)
- **Détection de points bloquants** : effondrement de logprobs par position,
  instabilité de position (variance anormale entre enregistrements), pics d'entropie.
- **Détection d'anomalies** : boucles de répétition, effondrements de logprobs,
  longueurs anormales, pics de latence — avec sévérité et z-scores.
- **Comparaison de modèles** : tableau de deltas entre deux versions, verdict
  automatique (stable / vigilance / régression / amélioration).
- **Dashboard React** : cartes de métriques, courbes, heatmap position × run,
  journal d'anomalies filtrable, comparaison côte à côte.

## Structure

```
backend/          API FastAPI (ingestion, métriques, anomalies, comparaison)
frontend/         Dashboard React + Vite + Recharts
scripts/          Générateur de dataset d'exemple
data/             Datasets d'exemple générés
```

## Démarrage rapide

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Documentation interactive : http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard : http://localhost:3000 (proxy automatique vers le backend sur le port 8000).

### Dataset d'exemple

```bash
python scripts/generate_sample_data.py
```

Génère `data/sample_runs.jsonl` : 240 enregistrements, 2 modèles (`stable-v1` sain,
`degraded-v2` avec effondrement de logprobs après la position 8, boucles de répétition
et pics de latence) sur 3 runs. Importez ce fichier dans le dashboard pour voir les
détections en action.

### Tests

```bash
cd backend
python -m pytest tests -q
```

## Format des données attendues

Chaque enregistrement représente une génération du modèle :

| Champ         | Alias acceptés                                   | Requis |
| ------------- | ------------------------------------------------ | ------ |
| `model`       | `model_name`, `model_id`, `model_version`        | sinon `unknown` |
| `tokens`      | `token_ids`, `output_tokens`, `generated_tokens` | sinon dérivé de `output` |
| `logprobs`    | `token_logprobs`, `log_probs`                    | optionnel |
| `run`         | `run_id`, `seed`, `iteration`, `trial`           | sinon 1 |
| `prompt_id`   | `sample_id`, `example_id`, `case_id`             | sinon index |
| `latency_ms`  | `latency`, `duration_ms`, `time_ms`              | optionnel |
| `output`      | `response`, `completion`, `text`, `answer`       | optionnel |
| `timestamp`   | `created_at`, `datetime`, `date`                 | optionnel |

Exemple JSONL :

```json
{"model": "mistral-small", "prompt_id": "p-001", "run": 1, "tokens": ["le", "chat"], "logprobs": [-0.31, -0.74], "latency_ms": 220}
```
