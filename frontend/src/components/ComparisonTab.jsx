import { useCallback, useEffect, useState } from "react";

const VERDICT_LABELS = {
  stable: "✅ Stable",
  warning: "⚠️ Vigilance",
  regression: "🔴 Régression détectée",
  improvement: "🟢 Amélioration",
};

const STATUS_LABELS = {
  stable: "stable",
  regression: "régression",
  improvement: "amélioration",
};

export default function ComparisonTab({ dataset, currentModel, fetchComparison }) {
  const [modelA, setModelA] = useState(currentModel);
  const [modelB, setModelB] = useState(
    dataset.models.find((m) => m !== currentModel) || currentModel
  );
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const runComparison = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setResult(await fetchComparison(dataset.id, modelA, modelB));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [dataset.id, modelA, modelB, fetchComparison]);

  useEffect(() => {
    runComparison();
  }, [runComparison]);

  return (
    <div className="tab-body">
      <div className="chart-card">
        <div className="table-toolbar">
          <h3>Comparaison de modèles</h3>
          <div className="compare-selects">
            <select value={modelA} onChange={(e) => setModelA(e.target.value)}>
              {dataset.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <span className="muted">→</span>
            <select value={modelB} onChange={(e) => setModelB(e.target.value)}>
              {dataset.models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <button className="btn primary small" onClick={runComparison} disabled={loading}>
              {loading ? "…" : "Comparer"}
            </button>
          </div>
        </div>

        {error && <p className="error-banner">{error}</p>}

        {result && (
          <>
            <div className="verdict-box">
              <span className={`verdict-badge ${result.verdict}`}>
                {VERDICT_LABELS[result.verdict] || result.verdict}
              </span>
              <span className="muted">
                Divergence de distribution (JS) : {result.token_distribution_js ?? "—"}
              </span>
            </div>

            <table className="data-table">
              <thead>
                <tr>
                  <th>Métrique</th>
                  <th>{result.model_a}</th>
                  <th>{result.model_b}</th>
                  <th>Δ</th>
                  <th>Δ %</th>
                  <th>Statut</th>
                </tr>
              </thead>
              <tbody>
                {result.metrics.map((row) => (
                  <tr key={row.key} className={`status-${row.status}`}>
                    <td>{row.metric}</td>
                    <td>{row.model_a}</td>
                    <td>{row.model_b}</td>
                    <td>{row.delta > 0 ? `+${row.delta}` : row.delta}</td>
                    <td>{row.delta_pct !== null ? `${row.delta_pct > 0 ? "+" : ""}${row.delta_pct} %` : "—"}</td>
                    <td>
                      <span className={`status-badge ${row.status}`}>
                        {STATUS_LABELS[row.status]}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
