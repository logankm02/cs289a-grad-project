import axios from "axios";

// Use Vite env var `VITE_API_BASE` when provided.
// In Vite dev (`localhost:5173`) default to backend on 8000; otherwise use same-origin.
const envBase =
  typeof import.meta !== "undefined" &&
  import.meta.env &&
  import.meta.env.VITE_API_BASE
    ? import.meta.env.VITE_API_BASE
    : null;

const originBase = typeof window !== "undefined" ? window.location.origin : "";
const base =
  envBase ||
  (originBase.includes("5173") ? "http://localhost:8000" : originBase) ||
  "http://localhost:8000";
const api = axios.create({ baseURL: `${base}/api` });

export async function fetchClusters(opts = {}) {
  // user_email, min_cluster_size and max_threads can be provided via params
  // Accept an options object: { user_email, min_cluster_size, max_threads }
  const resp = await api.get("/clusters", { params: opts });
  return resp.data;
}

export async function runClusters(opts = {}) {
  // opts -> sent as query params (user_email, min_cluster_size, max_threads)
  const resp = await api.post("/clusters/run", null, { params: opts });
  return resp.data;
}

export async function getTask(taskId) {
  const resp = await api.get(`/tasks/${taskId}`);
  return resp.data;
}

export async function runIndex(opts = {}) {
  const resp = await api.post("/index/run", null, { params: opts });
  return resp.data;
}

export async function fetchEmbeddings(user_email) {
  const resp = await api.get("/embeddings", { params: { user_email } });
  return resp.data;
}

export async function fetchLastCluster() {
  const resp = await api.get("/clusters/last");
  return resp.data;
}

export default api;
