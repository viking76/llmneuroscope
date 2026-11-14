import { useCallback, useEffect, useState } from "react";
import {
  deleteDataset,
  fetchAnalysis,
  fetchAnomalies,
  fetchBlockingPoints,
  fetchDatasets,
  fetchHeatmap,
  uploadDataset,
} from "./api.js";
import UploadPanel from "./components/UploadPanel.jsx";
import DatasetList from "./components/DatasetList.jsx";
import OverviewTab from "./components/OverviewTab.jsx";
import HeatmapTab from "./components/HeatmapTab.jsx";
import AnomaliesTab from "./components/AnomaliesTab.jsx";
import ComparisonTab from "./components/ComparisonTab.jsx";

const TABS = [
  { id: "overview", label: "Vue d'ensemble" },
  { id: "heatmap", label: "Activité neuronale" },
  { id: "anomalies", label: "Anomalies & points bloquants" },
  { id: "compare", label: "Comparaison de modèles" },
];

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [model, setModel] = useState(null);
  const [tab, setTab] = useState("overview");
  const [analysis, setAnalysis] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [blockingPoints, setBlockingPoints] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refreshDatasets = useCallback(async () => {
    try {
      const list = await fetchDatasets();
      setDatasets(list);
      return list;
    } catch (err) {
      setError(`API injoignable : ${err.message}. Lancez le backend (uvicorn) puis rechargez.`);
      return [];
    }
  }, []);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  const loadAnalysis = useCallback(async (datasetId, modelName) => {
    setLoading(true);
    setError(null);
    try {
      const [a, hm, bp, an] = await Promise.all([
        fetchAnalysis(datasetId, modelName),
        fetchHeatmap(datasetId, modelName),
        fetchBlockingPoints(datasetId, modelName),
        fetchAnomalies(datasetId, modelName),
      ]);
      setAnalysis(a);
      setHeatmapData(hm);
      setBlockingPoints(bp.points);
      setAnomalies(an.anomalies);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const selectDataset = useCallback(
    async (dataset) => {
      setActiveId(dataset.id);
      setModel(dataset.models[0]);
      await loadAnalysis(dataset.id, dataset.models[0]);
    },
    [loadAnalysis]
  );

  const changeModel = useCallback(
    async (modelName) => {
      setModel(modelName);
      if (activeId) await loadAnalysis(activeId, modelName);
    },
    [activeId, loadAnalysis]
  );

  const handleUpload = useCallback(
    async (file) => {
      setError(null);
      try {
        const summary = await uploadDataset(file);
        await refreshDatasets();
        await selectDataset(summary);
      } catch (err) {
        setError(err.message);
      }
    },
    [refreshDatasets, selectDataset]
  );

  const handleDelete = useCallback(
    async (id) => {
      await deleteDataset(id);
      if (activeId === id) {
        setActiveId(null);
        setAnalysis(null);
      }
      await refreshDatasets();
    },
    [activeId, refreshDatasets]
  );

  const activeDataset = datasets.find((d) => d.id === activeId) || null;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>🔬 LLM Neuroscope</h1>
          <p className="subtitle">
            Détection des problèmes d'un LLM via l'analyse de son activité neuronale
          </p>
        </div>
        {activeDataset && (
          <div className="dataset-info">
            <span className="badge">{activeDataset.name}</span>
            <span className="muted">
              {activeDataset.n_records} enregistrements · {activeDataset.n_runs} runs
            </span>
          </div>
        )}
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="layout">
        <aside className="sidebar">
          <UploadPanel onUpload={handleUpload} />
          <DatasetList
            datasets={datasets}
            activeId={activeId}
            onSelect={selectDataset}
            onDelete={handleDelete}
          />
        </aside>

        <main className="content">
          {!activeDataset && (
            <div className="empty-state">
              <h2>Aucun dataset sélectionné</h2>
              <p>
                Importez un fichier CSV, JSON ou JSONL contenant vos logs d'inférence
                (modèle, tokens, logprobs, latence), ou générez un dataset d'exemple avec{" "}
                <code>python scripts/generate_sample_data.py</code>.
              </p>
            </div>
          )}

          {activeDataset && (
            <>
              <div className="toolbar">
                <div className="tabs">
                  {TABS.map((t) => (
                    <button
                      key={t.id}
                      className={`tab ${tab === t.id ? "active" : ""}`}
                      onClick={() => setTab(t.id)}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <label className="model-select">
                  Modèle :
                  <select value={model || ""} onChange={(e) => changeModel(e.target.value)}>
                    {activeDataset.models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {loading && <div className="loading">Analyse en cours…</div>}

              {!loading && analysis && (
                <>
                  {tab === "overview" && <OverviewTab analysis={analysis} />}
                  {tab === "heatmap" && <HeatmapTab heatmap={heatmapData} analysis={analysis} />}
                  {tab === "anomalies" && (
                    <AnomaliesTab anomalies={anomalies} blockingPoints={blockingPoints} />
                  )}
                  {tab === "compare" && (
                    <ComparisonTab
                      dataset={activeDataset}
                      currentModel={model}
                      fetchComparison={fetchComparison}
                    />
                  )}
                </>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
