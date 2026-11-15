import { useCallback, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "";

async function probeRequest(path, options = {}) {
  const response = await fetch(`/api/probe${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Erreur ${response.status}`);
  return body;
}

function Histogram({ histogram }) {
  const max = Math.max(...histogram.counts, 1);
  return (
    <div className="histogram">
      {histogram.counts.map((count, index) => (
        <div
          key={index}
          className="histogram-bar"
          style={{ height: `${Math.max(2, (count / max) * 100)}%` }}
          title={`[${histogram.edges[index]?.toFixed(4)} ; ${histogram.edges[index + 1]?.toFixed(4)}] : ${count}`}
        />
      ))}
    </div>
  );
}

export default function ProbeTab() {
  const [status, setStatus] = useState(null);
  const [modelInput, setModelInput] = useState("gpt2");
  const [device, setDevice] = useState("auto");
  const [layers, setLayers] = useState([]);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [weightStats, setWeightStats] = useState(null);
  const [neuron, setNeuron] = useState(0);
  const [factor, setFactor] = useState("0.5");
  const [texts, setTexts] = useState("Le chat dort sur le tapis.\nExplique la photosynthèse.");
  const [activationResult, setActivationResult] = useState(null);
  const [generation, setGeneration] = useState(null);
  const [prompt, setPrompt] = useState("La convergence d'un modèle se mesure avec");
  const [maxTokens, setMaxTokens] = useState(12);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [applyFilename, setApplyFilename] = useState("");
  const [attentionResult, setAttentionResult] = useState(null);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await probeRequest("/status");
      setStatus(s);
      if (s.loaded) {
        const l = await probeRequest("/layers");
        setLayers(l.layers);
      } else {
        setLayers([]);
      }
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

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

  const loadModel = () =>
    run(async () => {
      await probeRequest("/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelInput, device }),
      });
      await refreshStatus();
    });

  const selectLayer = (layer) =>
    run(async () => {
      setSelectedLayer(layer);
      setWeightStats(await probeRequest(`/weights/${layer}`));
    });

  const modify = (action) =>
    run(async () => {
      if (!selectedLayer) return;
      await probeRequest("/weights/modify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          layer: selectedLayer,
          neuron: action === "restore" ? null : Number(neuron),
          action,
          factor: action === "scale" ? Number(factor) : null,
        }),
      });
      setWeightStats(await probeRequest(`/weights/${selectedLayer}`));
      await refreshStatus();
    });

  const exportEdits = () =>
    run(async () => {
      const result = await probeRequest("/weights/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      setExportResult(result);
      setApplyFilename(result.path.split("/").pop());
      await refreshStatus();
    });

  const applyEdits = () =>
    run(async () => {
      await probeRequest("/weights/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: applyFilename }),
      });
      if (selectedLayer) setWeightStats(await probeRequest(`/weights/${selectedLayer}`));
      await refreshStatus();
    });

  const analyzeTexts = () =>
    run(async () => {
      const list = texts.split("\n").map((t) => t.trim()).filter(Boolean);
      setActivationResult(await probeRequest("/analyze-texts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: list }),
      }));
    });

  const analyzeAttention = () =>
    run(async () => {
      const list = texts.split("\n").map((t) => t.trim()).filter(Boolean);
      setAttentionResult(await probeRequest("/attention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: list }),
      }));
    });

  const generateText = () =>
    run(async () => {
      setGeneration(await probeRequest("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: Number(maxTokens) }),
      }));
    });

  return (
    <div className="tab-body">
      {error && <div className="error-banner">{error}</div>}

      <div className="chart-card">
        <div className="table-toolbar">
          <h3>Sonde neuronale {status && <span className={`status-badge ${status.loaded ? "improvement" : "stable"}`}>{status.backend}</span>}</h3>
          <div className="compare-selects">
            <input
              value={modelInput}
              onChange={(e) => setModelInput(e.target.value)}
              placeholder="gpt2, EleutherAI/pythia-160m…"
              style={{ width: 220 }}
            />
            <select value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </select>
            <button className="btn primary small" onClick={loadModel} disabled={busy}>
              {busy ? "…" : "Charger"}
            </button>
          </div>
        </div>
        {status && (
          <p className="muted small">
            {status.loaded
              ? `Modèle : ${status.model_name} · ${status.n_layers} couches · device ${status.device}`
              : "Aucun modèle chargé. Un modèle open-weights local est requis pour lire et modifier les poids (les APIs fermées n'exposent pas leurs poids)."}
            {status.modified && " · ⚠️ poids modifiés (non persistés)"}
          </p>
        )}
      </div>

      {status?.loaded && (
        <>
          <div className="chart-card">
            <h3>Activité neuronale sur textes de test</h3>
            <textarea
              rows={3}
              value={texts}
              onChange={(e) => setTexts(e.target.value)}
              style={{ width: "100%" }}
            />
            <button className="btn primary small" onClick={analyzeTexts} disabled={busy}>
              Analyser l'activité
            </button>
            {activationResult && (
              <table className="data-table" style={{ marginTop: 12 }}>
                <thead>
                  <tr>
                    <th>Couche</th>
                    <th>Moyenne</th>
                    <th>σ</th>
                    <th>Neurones morts</th>
                    <th>Top neurones (activation max)</th>
                  </tr>
                </thead>
                <tbody>
                  {activationResult.layers.map((layer) => (
                    <tr key={layer.layer}>
                      <td>{layer.layer}</td>
                      <td>{layer.mean.toFixed(4)}</td>
                      <td>{layer.std.toFixed(4)}</td>
                      <td className={layer.dead_neurons > 0 ? "text-danger" : ""}>
                        {layer.dead_neurons} ({(layer.dead_neuron_ratio * 100).toFixed(1)} %)
                      </td>
                      <td>
                        {layer.top_neurons.map((n) => `#${n.index} (${n.max_activation.toFixed(3)})`).join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="chart-card">
            <h3>Attention (TransformerLens)</h3>
            <p className="muted small">
              Entropie d'attention par couche et par tête : une tête « diffuse »
              (ratio proche de 1) regarde tout le contexte, une tête « focalisée »
              (ratio proche de 0) suit un motif précis.
            </p>
            <button className="btn primary small" onClick={analyzeAttention} disabled={busy}>
              Analyser l'attention
            </button>
            {attentionResult && (
              <div style={{ marginTop: 12 }}>
                <p className="muted small">
                  {attentionResult.n_layers} couches × {attentionResult.n_heads} têtes ·
                  séquence {attentionResult.sequence_length} tokens · entropie max{" "}
                  {attentionResult.max_entropy} bits
                </p>
                <div className="attention-grid">
                  {attentionResult.head_matrix.map((row, layer) =>
                    row.map((value, head) => {
                      const ratio = value / attentionResult.max_entropy;
                      return (
                        <div
                          key={`${layer}-${head}`}
                          className="attention-cell"
                          style={{
                            background: `rgb(${Math.round(40 + ratio * 215)}, ${Math.round(60 + (1 - ratio) * 40)}, ${Math.round(120 + (1 - ratio) * 80)})`,
                          }}
                          title={`Couche ${layer} · tête ${head} · entropie ${value} bits (${(ratio * 100).toFixed(0)} %)`}
                        />
                      );
                    })
                  )}
                </div>
                <div className="heat-lists">
                  <div>
                    <p className="section-title small">Têtes les plus diffuses</p>
                    {attentionResult.top_heads.slice(0, 3).map((entry) => (
                      <p key={`${entry.layer}-${entry.head}`} className="muted small">
                        Couche {entry.layer} · tête {entry.head} — {entry.entropy} bits ({(entry.ratio * 100).toFixed(0)} %)
                      </p>
                    ))}
                  </div>
                  <div>
                    <p className="section-title small">Têtes les plus focalisées</p>
                    {attentionResult.focused_heads.slice(0, 3).map((entry) => (
                      <p key={`${entry.layer}-${entry.head}`} className="muted small">
                        Couche {entry.layer} · tête {entry.head} — {entry.entropy} bits ({(entry.ratio * 100).toFixed(0)} %)
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="chart-card">
            <h3>Poids des couches</h3>
            <div className="layer-list">
              {layers.map((layer) => (
                <button
                  key={layer}
                  className={`layer-chip ${layer === selectedLayer ? "active" : ""}`}
                  onClick={() => selectLayer(layer)}
                >
                  {layer}
                </button>
              ))}
            </div>
            {weightStats && (
              <div className="weight-details">
                <p className="muted small">
                  {weightStats.layer} · forme {weightStats.shape.join("×")} · moyenne{" "}
                  {weightStats.mean} · σ {weightStats.std} · norme {weightStats.norm} ·
                  poids morts {(weightStats.dead_ratio * 100).toFixed(2)} %
                </p>
                <Histogram histogram={weightStats.histogram} />
                <p className="muted small">
                  Aperçu ligne 0 : [{weightStats.preview.join(", ")}]
                </p>
                <div className="compare-selects" style={{ marginTop: 10 }}>
                  <label className="muted small">
                    Neurone #
                    <input
                      type="number"
                      min={0}
                      value={neuron}
                      onChange={(e) => setNeuron(e.target.value)}
                      style={{ width: 70 }}
                    />
                  </label>
                  <label className="muted small">
                    Facteur
                    <input
                      type="number"
                      step="0.1"
                      value={factor}
                      onChange={(e) => setFactor(e.target.value)}
                      style={{ width: 80 }}
                    />
                  </label>
                  <button className="btn small" onClick={() => modify("scale")} disabled={busy}>
                    Multiplier
                  </button>
                  <button className="btn small" onClick={() => modify("zero")} disabled={busy}>
                    Mettre à zéro
                  </button>
                  <button
                    className="btn danger small"
                    onClick={() => modify("restore")}
                    disabled={busy || !status.snapshot_available}
                  >
                    Restaurer l'original
                  </button>
                </div>
                <div className="compare-selects" style={{ marginTop: 8 }}>
                  <button
                    className="btn small"
                    onClick={exportEdits}
                    disabled={busy || !status.snapshot_available}
                  >
                    Exporter (safetensors)
                  </button>
                  <input
                    value={applyFilename}
                    onChange={(e) => setApplyFilename(e.target.value)}
                    placeholder="fichier .safetensors"
                    style={{ width: 200 }}
                  />
                  <button
                    className="btn small"
                    onClick={applyEdits}
                    disabled={busy || !applyFilename}
                  >
                    Appliquer
                  </button>
                </div>
                {exportResult && (
                  <p className="muted small">
                    Exporté : <code>{exportResult.path}</code> ({exportResult.layers.join(", ")})
                  </p>
                )}
                {status.edit_log?.length > 0 && (
                  <details style={{ marginTop: 8 }}>
                    <summary className="muted small">
                      Journal des modifications ({status.n_edits})
                    </summary>
                    <pre className="edit-log">
                      {status.edit_log.map((entry) => JSON.stringify(entry)).join("\n")}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>

          <div className="chart-card">
            <h3>Génération instrumentée</h3>
            <div className="compare-selects">
              <input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                style={{ flex: 1 }}
              />
              <input
                type="number"
                min={1}
                max={256}
                value={maxTokens}
                onChange={(e) => setMaxTokens(e.target.value)}
                style={{ width: 70 }}
              />
              <button className="btn primary small" onClick={generateText} disabled={busy}>
                Générer
              </button>
            </div>
            {generation && (
              <>
                <p className="muted small" style={{ marginTop: 10 }}>
                  Perplexité : {generation.perplexity}
                </p>
                <div className="token-strip">
                  {generation.tokens.map((token, index) => {
                    const logprob = generation.logprobs[index];
                    const tone = logprob > -1 ? "good" : logprob > -4 ? "warn" : "bad";
                    return (
                      <span
                        key={index}
                        className={`token-chip ${tone}`}
                        title={`logprob ${logprob}`}
                      >
                        {token}
                        <small>{logprob.toFixed(2)}</small>
                      </span>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
