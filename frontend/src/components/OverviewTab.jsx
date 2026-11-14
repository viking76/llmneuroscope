import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function MetricCard({ label, value, hint, tone }) {
  return (
    <div className={`metric-card ${tone || ""}`}>
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {hint && <p className="metric-hint">{hint}</p>}
    </div>
  );
}

export default function OverviewTab({ analysis }) {
  const runEntropyData = Object.entries(analysis.run_entropy || {}).map(([run, entropy]) => ({
    run: `Run ${run}`,
    entropie: entropy,
  }));

  const positionData = (analysis.position_logprob || []).map((value, index) => ({
    position: index,
    "log-prob moyen": value,
    "entropie": analysis.position_entropy?.[index] ?? null,
  }));

  const degenerationPct = ((analysis.repetition?.degeneration_rate || 0) * 100).toFixed(1);

  return (
    <div className="tab-body">
      <div className="metric-grid">
        <MetricCard label="Perplexité" value={analysis.perplexity ?? "—"} hint="plus bas = plus confiant" />
        <MetricCard label="Entropie des tokens" value={analysis.token_entropy} hint={`vocabulaire: ${analysis.vocabulary_size}`} />
        <MetricCard
          label="Taux de dégénérescence"
          value={`${degenerationPct} %`}
          tone={analysis.repetition?.degeneration_rate > 0.1 ? "bad" : ""}
          hint="sorties en boucle de répétition"
        />
        <MetricCard
          label="Dérive max entre runs"
          value={analysis.drift?.max_js ?? 0}
          tone={analysis.drift?.max_js > 0.2 ? "warn" : ""}
          hint={
            analysis.drift?.to_run
              ? `run ${analysis.drift.from_run} → run ${analysis.drift.to_run} (JS)`
              : "aucune dérive mesurable"
          }
        />
        <MetricCard label="Longueur moyenne" value={analysis.length?.mean?.toFixed(1) ?? "—"} hint={`σ = ${analysis.length?.std?.toFixed(1) ?? "—"}`} />
        <MetricCard label="Latence moyenne" value={`${analysis.latency_ms?.mean?.toFixed(0) ?? "—"} ms`} hint={`p95 = ${analysis.latency_ms?.p95?.toFixed(0) ?? "—"} ms`} />
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <h3>Entropie par run (convergence)</h3>
          {runEntropyData.length > 1 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={runEntropyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                <XAxis dataKey="run" stroke="#8b93a7" />
                <YAxis stroke="#8b93a7" />
                <Tooltip contentStyle={{ background: "#1a1f2b", border: "1px solid #2a2f3a" }} />
                <Bar dataKey="entropie" fill="#4f8cff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">Un seul run : impossible de mesurer la convergence.</p>
          )}
        </div>

        <div className="chart-card">
          <h3>Log-prob moyen par position</h3>
          {positionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={positionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                <XAxis dataKey="position" stroke="#8b93a7" />
                <YAxis stroke="#8b93a7" />
                <Tooltip contentStyle={{ background: "#1a1f2b", border: "1px solid #2a2f3a" }} />
                <Line type="monotone" dataKey="log-prob moyen" stroke="#ff6b6b" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="entropie" stroke="#4f8cff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">Pas de logprobs dans ce dataset.</p>
          )}
        </div>
      </div>
    </div>
  );
}
