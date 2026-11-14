const API_BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Erreur ${response.status}`);
  }
  return response.json();
}

export const fetchDatasets = () => request("/datasets");

export const uploadDataset = async (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/datasets", { method: "POST", body: form });
};

export const deleteDataset = (id) => request(`/datasets/${id}`, { method: "DELETE" });

export const fetchAnalysis = (id, model) =>
  request(`/datasets/${id}/analysis${model ? `?model=${encodeURIComponent(model)}` : ""}`);

export const fetchHeatmap = (id, model) =>
  request(`/datasets/${id}/heatmap${model ? `?model=${encodeURIComponent(model)}` : ""}`);

export const fetchBlockingPoints = (id, model) =>
  request(`/datasets/${id}/blocking-points${model ? `?model=${encodeURIComponent(model)}` : ""}`);

export const fetchAnomalies = (id, model) =>
  request(`/datasets/${id}/anomalies${model ? `?model=${encodeURIComponent(model)}` : ""}`);

export const fetchComparison = (id, modelA, modelB) =>
  request(`/datasets/${id}/compare?model_a=${encodeURIComponent(modelA)}&model_b=${encodeURIComponent(modelB)}`);
