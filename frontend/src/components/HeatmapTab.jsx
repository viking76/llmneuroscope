export default function HeatmapTab({ heatmap, analysis }) {
  if (!heatmap || !heatmap.matrix || heatmap.matrix.length === 0) {
    return (
      <div className="tab-body">
        <div className="empty-state">
          <p>Aucune donnée de tokens pour construire la heatmap d'activité.</p>
        </div>
      </div>
    );
  }

  const values = heatmap.matrix.flatMap((row) => row.values.filter((v) => v !== null));
  const max = Math.max(...values, 0.001);
  const blockingPositions = new Set((analysis.position_entropy || []).map((_, i) => i));

  const color = (value) => {
    if (value === null) return "transparent";
    const intensity = value / max;
    const r = Math.round(40 + intensity * 215);
    const g = Math.round(60 + (1 - intensity) * 40);
    const b = Math.round(120 + (1 - intensity) * 80);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="tab-body">
      <div className="chart-card">
        <h3>Heatmap d'activité — entropie par position × run</h3>
        <p className="muted small">
          Chaque cellule représente l'entropie (incertitude) du modèle à une position donnée,
          pour un run donné. Une ligne qui « chauffe » indique une zone d'instabilité.
        </p>
        <div className="heatmap-scroll">
          <div className="heatmap">
            <div className="heatmap-row heatmap-header">
              <div className="heatmap-label">Run</div>
              {heatmap.positions.map((p) => (
                <div key={p} className="heatmap-cell header" title={`Position ${p}`}>
                  {p % 8 === 0 ? p : ""}
                </div>
              ))}
            </div>
            {heatmap.matrix.map((row) => (
              <div key={row.run} className="heatmap-row">
                <div className="heatmap-label">{row.run}</div>
                {row.values.map((value, index) => (
                  <div
                    key={index}
                    className="heatmap-cell"
                    style={{ background: color(value) }}
                    title={`Run ${row.run} · position ${index} · entropie ${value ?? "—"}`}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="heatmap-legend">
          <span className="muted small">0 (déterministe)</span>
          <div className="legend-gradient" />
          <span className="muted small">{max.toFixed(2)} (aléatoire)</span>
        </div>
      </div>

      <div className="chart-card">
        <h3>Profil d'entropie par position (tous runs confondus)</h3>
        <div className="entropy-bars">
          {analysis.position_entropy?.map((value, index) => (
            <div
              key={index}
              className="entropy-bar"
              style={{ height: `${Math.max(4, (value / max) * 100)}%` }}
              title={`Position ${index} · entropie ${value}`}
            />
          ))}
        </div>
        <p className="muted small">
          {blockingPositions.size} positions analysées · max = {max.toFixed(3)} bits
        </p>
      </div>
    </div>
  );
}
