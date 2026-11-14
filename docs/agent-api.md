# Guide de l'agent — piloter LLM Neuroscope par API

Ce document s'adresse à une IA (agent, script, LLM tooling) qui souhaite tester un
modèle de langue avec des textes, observer son activité neuronale, inspecter les
poids de ses neurones et les modifier si nécessaire.

## Prérequis

- Backend lancé : `uvicorn app.main:app --port 8000` (base : `http://localhost:8000`)
- **Lecture/modification des poids** : nécessite un modèle **open-weights exécuté
  localement** (`pip install -r backend/requirements-probe.txt`, puis `POST /api/probe/load`).
  Les modèles fermés accessibles par API (Mistral API, OpenAI…) n'exposent ni
  activations ni poids : seule l'analyse de logs (`/api/datasets`) fonctionne pour eux.
- Sans torch/transformers, la sonde fonctionne en mode `mock` (simulation
  déterministe) : utile pour tester l'intégration, pas pour de vrais poids.

## Session type d'un agent

### 1. Vérifier la sonde

```http
GET /api/probe/status
```

```json
{"backend": "torch", "loaded": false}
```

### 2. Charger un modèle open-weights

```http
POST /api/probe/load
{"model_name": "gpt2", "device": "auto"}
```

Réponse : statut + nombre de couches. Modèles conseillés pour commencer :
`gpt2` (124M), `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`.

### 3. Tester avec des textes (activité neuronale)

```http
POST /api/probe/analyze-texts
{"texts": ["Le chat dort.", "Explique la photosynthèse."]}
```

Pour chaque couche : moyenne, écart-type, **ratio de neurones morts** (jamais
activés sur le lot) et **top neurones** par activation maximale. Un ratio de
neurones morts élevé ou une activation quasi nulle sur toutes les couches sont
des signaux de blocage.

### 4. Inspecter les poids d'une couche

```http
GET /api/probe/layers
GET /api/probe/weights/transformer.h.0.mlp.c_fc
```

Statistiques : forme, moyenne, σ, norme, ratio de poids morts, histogramme
(16 classes) et aperçu des 8 premières valeurs. Une couche avec beaucoup de
poids morts ou une norme effondrée est candidate à intervention.

### 5. Modifier un neurone si nécessaire

```http
POST /api/probe/weights/modify
{"layer": "transformer.h.0.mlp.c_fc", "neuron": 12, "action": "scale", "factor": 0.5}
```

Actions : `scale` (multiplie la ligne du neurone par `factor`), `zero`
(désactive le neurone), `restore` (restaure les valeurs d'origine de toutes les
couches modifiées). Un instantané est pris automatiquement avant la première
modification — **toute modification est réversible** tant que le process tourne.
Les modifications ne sont pas persistées sur disque.

Réponse : statistiques avant/après pour vérifier l'effet.

### 6. Vérifier l'effet après modification

```http
POST /api/probe/analyze-texts
POST /api/probe/generate {"prompt": "...", "max_new_tokens": 32}
```

`analyze-texts` sur les mêmes textes montre si l'activité a changé ; `generate`
renvoie les tokens générés avec leur logprob et la perplexité : comparez
avant/après modification. Si le comportement se dégrade :

```http
POST /api/probe/weights/modify
{"layer": "...", "action": "restore"}
```

## Bonnes pratiques pour l'agent

1. Toujours `restore` après expérimentation, sauf décision explicite de garder.
2. Modifier **un seul neurone à la fois** et mesurer entre chaque changement.
3. Préférer `scale` avec un facteur proche de 1 (0.5 – 2.0) avant `zero`.
4. Utiliser un lot de textes variés dans `analyze-texts` pour des statistiques stables.
5. Pour une analyse de convergence entre versions, exporter des logs (JSONL avec
   `model`, `tokens`, `logprobs`, `run`) et utiliser `POST /api/datasets` +
   `GET /api/datasets/{id}/compare`.
