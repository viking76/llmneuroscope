import { useState } from "react";

const TYPE_LABELS = {
  logprob_collapse: "Effondrement de logprobs",
  degenerate_repetition: "Boucle de répétition",
  length_outlier: "Longueur anormale",
  latency_spike: "Pic de latence",
  position_instability: "Instabilité de position",
};

const SEVERITY_LABELS = { high: "Haute", medium: "Moyenne", low: "Basse" };

export default function AnomaliesTab({ anomalies, blockingPoints }) {
  const [severityFilter, setSeverityFilter] = useState("all");

  const counts = anomalies.reduce((acc, anomaly) => {
    acc[anomaly.type] = (acc[anomaly.type] || 0) + 1;
    return acc;
  }, {});

  const filtered =
    severityFilter === "all"
      ? anomalies
      : anomalies.filter((a) => a.severity === severityFilter);

  return (
    <div className="tab-body">
      <div className="metric-grid">
        <MetricCard label="Anomalies détectées" value={anomalies.length} tone={anomalies.length > 0 ? "warn" : "good"} />
        {Object.entries(counts).map(([type, count]) => (
          <MetricCard key={type} label={TYPE_LABELS[type] || type} value={count} tone={count > 0 ? "bad" : "good"} />
        ))}
        {blockingPoints.length > 0 && (
          <MetricCard
            label="Points bloquants (positions)"
            value={blockingPoints.length}
            tone="bad"
            hint={`positions ${blockingPoints.map((p) => p.position).slice(0, 6).join(", ")}${blockingPoints.length > 6 ? "…" : ""}`}
          />
        )}
      </div>

      <div className="chart-card">
        <div className="table-toolbar">
          <h3>Journal des anomalies</h3>
          <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
            <option value="all">Toutes sévérités</option>
            <option value="high">Haute</option>
            <option value="medium">Moyenne</option>
            <option value="low">Basse</option>
          </select>
        </div>
        {filtered.length === 0 ? (
          <p className="muted">Aucune anomalie avec ce filtre. 🎉</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Sévérité</th>
                <th>Type</th>
                <th>Prompt</th>
                <th>Run</th>
                <th>Valeur</th>
                <th>Z-score</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((anomaly, index) => (
                <tr key={index} className={`severity-${anomaly.severity}`}>
                  <td>
                    <span className={`severity-badge ${anomaly.severity}`}>
                      {SEVERITY_LABELS[anomaly.severity]}
                    </span>
                  </td>
                  <td>{TYPE_LABELS[anomaly.type] || anomaly.type}</td>
                  <td>{anomaly.prompt_id}</td>
                  <td>{anomaly.run}</td>
                  <td>{anomaly.value}</td>
                  <td>{anomaly.z_score ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {blockingPoints.length > 0 && (
        <div className="chart-card">
          <h3>Points bloquants par position</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Position</th>
                <th>Type</th>
                <th>Valeur</th>
                <th>Z-score</th>
                <th>Sévérité</th>
              </tr>
            </thead>
            <tbody>
              {blockingPoints.map((point, index) => (
                <tr key={index} className={`severity-${point.severity}`}>
                  <td>{point.position}</td>
                  <td>{TYPE_LABELS[point.type] || point.type}</td>
                  <td>{point.value}</td>
                  <td>{point.z_score}</td>
                  <td>
                    <span className={`severity-badge ${point.severity}`}>
                      {SEVERITY_LABELS[point.severity]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, hint, tone }) {
  return (
    <div className={`metric-card ${tone || ""}`}>
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {hint && <p className="metric-hint">{hint}</p>}
    </div>
  );
}
