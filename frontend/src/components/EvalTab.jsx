import { useCallback, useEffect, useState } from "react";

const TASKS = ["lambada_openai", "hellaswag", "arc_easy", "winogrande", "piqa"];

async function evalRequest(path, options = {}) {
  const response = await fetch(`/api/evals${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Erreur ${response.status}`);
  return body;
}

async function probeStatus() {
  const response = await fetch("/api/probe/status");
  return response.json();
}

export default function EvalTab() {
  const [status, setStatus] = useState(null);
  const [modelInput, setModelInput] = useState("");
  const [selectedTasks, setSelectedTasks] = useState(["lambada_openai"]);
  const [limit, setLimit] = useState(10);
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [texts, setTexts] = useState("Le chat dort sur le tapis.\nLe chat chat chat chat chat.");
  const [useJudge, setUseJudge] = useState(false);
  const [qualityResult, setQualityResult] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [langfuseResult, setLangfuseResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    probeStatus().then(setStatus).catch(() => {});
    fetch("/api/datasets")
      .then((r) => r.json())
      .then((list) => {
        setDatasets(list);
        if (list.length) setDatasetId(list[0].id);
      })
      .catch(() => {});
  }, []);

  const run = useCallback(async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const toggleTask = (task) =>
    setSelectedTasks((current) =>
      current.includes(task) ? current.filter((t) => t !== task) : [...current, task]
    );

  const runBenchmark = () =>
    run(async () => {
      setBenchmarkResult(
        await evalRequest("/benchmark", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: modelInput || null,
            tasks: selectedTasks,
            limit: Number(limit),
          }),
        })
      );
    });

  const runQuality = () =>
    run(async () => {
      const list = texts.split("\n").map((t) => t.trim()).filter(Boolean);
      setQualityResult(
        await evalRequest("/quality", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: list, use_judge: useJudge }),
        })
      );
    });

  const exportLangfuse = () =>
    run(async () => {
      setLangfuseResult(
        await evalRequest("/langfuse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: datasetId }),
        })
      );
    });

  return (
    <div className="tab-body">
      {error && <div className="error-banner">{error}</div>}

      <div className="chart-card">
        <h3>Benchmarks (lm-evaluation-harness)</h3>
        <p className="muted small">
          Scores standardisés sur les tâches EleutherAI (échantillon réduit pour un
          test rapide). Le modèle est chargé par le harness indépendamment de la sonde.
        </p>
        <div className="compare-selects">
          <input
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
            placeholder={status?.model_name || "EleutherAI/pythia-70m"}
            style={{ width: 220 }}
          />
          <label className="muted small">
            Limite
            <input
              type="number"
              min={1}
              max={100}
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              style={{ width: 70 }}
            />
          </label>
          <button
            className="btn primary small"
            onClick={runBenchmark}
            disabled={busy || selectedTasks.length === 0}
          >
            Lancer le benchmark
          </button>
        </div>
        <div className="layer-list" style={{ marginTop: 10 }}>
          {TASKS.map((task) => (
            <button
              key={task}
              className={`layer-chip ${selectedTasks.includes(task) ? "active" : ""}`}
              onClick={() => toggleTask(task)}
            >
              {task}
            </button>
          ))}
        </div>
        {benchmarkResult && (
          <table className="data-table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Tâche</th>
                {Object.keys(Object.values(benchmarkResult.tasks)[0] || {}).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(benchmarkResult.tasks).map(([task, metrics]) => (
                <tr key={task}>
                  <td>{task}</td>
                  {Object.entries(metrics).map(([key, value]) => (
                    <td key={key}>
                      {typeof value === "number" ? value.toFixed(4) : String(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {benchmarkResult && (
          <p className="muted small" style={{ marginTop: 6 }}>
            Moteur {benchmarkResult.engine} · modèle {benchmarkResult.model_name} ·
            limite {benchmarkResult.limit} exemples par tâche
          </p>
        )}
      </div>

      <div className="chart-card">
        <h3>Évals de qualité</h3>
        <p className="muted small">
          Métriques internes par texte : perplexité sous le modèle chargé, ratio de
          répétition (n-grammes), diversité lexicale. Un texte répétitif ou à
          perplexité explosive signale une dégénération.
        </p>
        <div className="compare-selects">
          <label className="muted small">
            <input
              type="checkbox"
              checked={useJudge}
              onChange={(e) => setUseJudge(e.target.checked)}
            />{" "}
            Juge DeepEval G-Eval (nécessite OPENAI_API_KEY)
          </label>
        </div>
        <textarea
          value={texts}
          onChange={(e) => setTexts(e.target.value)}
          rows={4}
          style={{ width: "100%", marginTop: 8 }}
        />
        <button className="btn primary small" onClick={runQuality} disabled={busy}>
          Évaluer la qualité
        </button>
        {qualityResult && (
          <div style={{ marginTop: 12 }}>
            <p className="muted small">
              Agrégat : perplexité {qualityResult.aggregate.perplexity ?? "—"} ·
              répétition {(qualityResult.aggregate.repetition_ratio * 100).toFixed(1)} % ·
              diversité {(qualityResult.aggregate.distinct_ratio * 100).toFixed(1)} %
            </p>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Texte</th>
                  <th>Perplexité</th>
                  <th>Répétition</th>
                  <th>Diversité</th>
                </tr>
              </thead>
              <tbody>
                {qualityResult.texts.map((entry, index) => (
                  <tr key={index}>
                    <td>{entry.text}</td>
                    <td>{entry.perplexity ?? "—"}</td>
                    <td>{(entry.repetition_ratio * 100).toFixed(0)} %</td>
                    <td>{(entry.distinct_ratio * 100).toFixed(0)} %</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {qualityResult.judge && (
              <p className="muted small">
                Juge {qualityResult.judge.judge} :{" "}
                {qualityResult.judge.results
                  .map((r) => (r.score != null ? `score ${r.score}` : r.error))
                  .join(" · ")}
              </p>
            )}
            {qualityResult.judge_note && (
              <p className="muted small">⚠️ {qualityResult.judge_note}</p>
            )}
          </div>
        )}
      </div>

      <div className="chart-card">
        <h3>Export Langfuse</h3>
        <p className="muted small">
          Exporte chaque génération du dataset comme trace Langfuse (score
          mean_logprob inclus). Sans clés configurées, un fichier JSONL compatible
          ingestion est écrit dans <code>data/langfuse-exports/</code> ; avec
          LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST, envoi direct au serveur.
        </p>
        <div className="compare-selects">
          <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            {datasets.length === 0 && <option value="">Aucun dataset</option>}
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.n_records} records)
              </option>
            ))}
          </select>
          <button
            className="btn primary small"
            onClick={exportLangfuse}
            disabled={busy || !datasetId}
          >
            Exporter les traces
          </button>
        </div>
        {langfuseResult && (
          <p className="muted small" style={{ marginTop: 10 }}>
            {langfuseResult.mode === "push"
              ? `✅ ${langfuseResult.n_traces} traces poussées vers ${langfuseResult.host}`
              : `📁 ${langfuseResult.n_traces} traces écrites dans ${langfuseResult.path}`}
          </p>
        )}
      </div>
    </div>
  );
}
