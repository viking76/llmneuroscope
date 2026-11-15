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

### 5. Analyser l'attention (par couche × tête)

```http
POST /api/probe/attention
{"texts": ["Le chat dort sur le tapis.", "Explique la photosynthèse."]}
```

Réponse :

```json
{
  "engine": "transformer-lens",
  "n_layers": 6, "n_heads": 8, "sequence_length": 9,
  "max_entropy": 3.1699,
  "layer_entropy": [1.4024, 1.4612, 1.3556, 1.2167, 0.7592, 0.7975],
  "head_matrix": [[1.4, 1.38, ...], ...],
  "diffuse_heads": [],
  "top_heads": [...],
  "focused_heads": [{"layer": 2, "head": 1, "entropy": 0.5, "ratio": 0.1258}]
}
```

- `entropy` : entropie de Shannon des probabilités d'attention d'une tête, en
  bits. `ratio = entropy / log2(sequence_length)` : 1.0 = attention uniforme
  (diffuse), proche de 0 = attention concentrée sur peu de tokens.
- `layer_entropy` : profil par couche. Une décroissance en fin de réseau
  (focalisation) est normale ; un profil plat ou une entropie quasi maximale
  partout peut signaler un modèle peu spécialisé.
- `diffuse_heads` : têtes avec `ratio > 0.9` (quasi uniformes).
- `focused_heads` : les têtes les plus concentrées, candidates pour
  l'interprétation (têtes de position, d'induction…).

Moteur : **TransformerLens** (`TransformerBridge`) côté torch ; simulation
déterministe côté mock. Cette analyse est **en lecture seule et indépendante** :
elle utilise sa propre instance du modèle (bridge), donc elle ne reflète pas les
modifications de poids faites via `/weights/modify`.

### 6. Attribuer la prédiction aux tokens (Captum)

```http
POST /api/probe/attribution
{"text": "The capital of France is", "method": "integrated-gradients"}
```

`method` : `integrated-gradients` (par défaut) ou `gradient-shap`. La cible est
le token prédit en position suivante (greedy) ; chaque token du prompt reçoit
un `score` (positif = pousse vers la prédiction, négatif = la freine) et une
`share` (part de |score| total).

Interprétation : si la prédiction repose sur un seul token (`share` > 0.8),
elle est fragile ; un prompt bien formé répartit l'attribution sur plusieurs
tokens porteurs de sens.

Moteur : **Captum** (gradients sur les embeddings via `inputs_embeds`).

### 7. Activation patching (nnsight)

```http
POST /api/probe/patching
{"clean_text": "The capital of France is Paris", "corrupted_text": "The capital of Italy is Paris"}
```

Pour chaque couche du décodeur : remplace l'activation du **dernier token** du
texte corrompu par celle du texte propre, puis mesure la divergence KL entre la
prédiction patchée et la prédiction propre.

Réponse : `baseline_kl` (corrompu vs propre, sans patch), `layer_kl` (par
couche), `layer_restoration` (% de KL baseline éliminée par le patch) et
`critical_layer` (couche qui restaure le moins).

Interprétation : une restauration qui monte avec la profondeur est normale
(l'information se condense vers la sortie). Une couche au milieu qui restaure
moins que ses voisines est un point bloquant potentiel. Le patch est
**positionnel** (dernier token) : corruptez un token antérieur au point mesuré,
sinon les deux textes produisent des sorties identiques et l'analyse est rejetée.

Moteur : **nnsight** (`TransformersModel`, instance séparée en lecture seule —
ne reflète pas les modifications de poids). Requiert que le texte corrompu
produise des logits différents du texte propre.

### 8. Modifier un neurone si nécessaire

```http
POST /api/probe/weights/modify
{"layer": "transformer.h.0.mlp.c_fc", "neuron": 12, "action": "scale", "factor": 0.5}
```

Actions : `scale` (multiplie la ligne du neurone par `factor`), `zero`
(désactive le neurone), `restore` (restaure les valeurs d'origine de toutes les
couches modifiées). Un instantané est pris automatiquement avant la première
modification — **toute modification est réversible** tant que le process tourne.
Réponse : statistiques avant/après pour vérifier l'effet.

### Persister les modifications (safetensors)

```http
POST /api/probe/weights/export {"filename": "mon-correctif.safetensors"}
POST /api/probe/weights/apply  {"filename": "mon-correctif.safetensors"}
```

`export` écrit les couches actuellement modifiées dans
`data/weight-edits/<filename>` (format safetensors + journal JSON des éditions) ;
`apply` recharge un fichier exporté dans le modèle courant. Les chemins sont
confinés à `data/weight-edits/` (toute traversée de répertoire est rejetée).

### 9. Vérifier l'effet après modification

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
6. Combiner `analyze-texts` (activations), `attention` (couches × têtes),
   `attribution` (tokens) et `patching` (couches critiques) avant toute
   modification : un blocage se manifeste rarement sur une seule de ces vues.
7. Pour `patching`, corrompre un token **antérieur** à la position mesurée
   (sujet, relation) — pas le dernier token.
8. Benchmarks (`POST /api/evals/benchmark`) : commencer par `lambada_openai`
   avec `limit` 5-10 avant de lancer une tâche coûteuse (`hellaswag`).
9. Traces : `POST /api/evals/langfuse` exporte les générations d'un dataset
   (JSONL local par défaut, push direct si `LANGFUSE_PUBLIC_KEY` /
   `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` sont définis).
10. Qualité avec juge (`POST /api/evals/quality`, `judge` = `local` | `cloud`) :
   le juge local (Qwen2.5-1.5B-Instruct, téléchargé au premier usage) note la
   cohérence de 0 à 1 avec une raison ; `cloud` utilise G-Eval via
   `OPENAI_API_KEY`. Un score < 0.5 sur un texte produit par le modèle testé
   est un signal de dégénération.
