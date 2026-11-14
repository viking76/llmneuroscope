export default function DatasetList({ datasets, activeId, onSelect, onDelete }) {
  return (
    <div className="dataset-list">
      <p className="section-title">Datasets ({datasets.length})</p>
      {datasets.length === 0 && <p className="muted">Aucun dataset importé.</p>}
      {datasets.map((dataset) => (
        <div
          key={dataset.id}
          className={`dataset-item ${dataset.id === activeId ? "active" : ""}`}
          onClick={() => onSelect(dataset)}
        >
          <div className="dataset-item-main">
            <span className="dataset-name">{dataset.name}</span>
            <span className="muted small">
              {dataset.n_records} enr. · {dataset.models.length} modèle(s)
            </span>
          </div>
          <button
            className="btn danger small"
            title="Supprimer"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(dataset.id);
            }}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
