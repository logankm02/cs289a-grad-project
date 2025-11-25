import React, { useState } from 'react'
import { runClusters, fetchClusters, getTask, fetchEmbeddings, fetchLastCluster, runIndex } from './api'
import Plot from 'react-plotly.js'

const POLL_MS = 2000
const MAX_POLLS = 60 // ~2 minutes

export default function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [taskId, setTaskId] = useState(null)
  const [taskKind, setTaskKind] = useState(null)
  const [userEmail, setUserEmail] = useState('')
  const [maxThreads, setMaxThreads] = useState(200)
  const [pageSize, setPageSize] = useState(100)
  const [embedBatch, setEmbedBatch] = useState(64)
  const [resetIndex, setResetIndex] = useState(false)
  const [minClusterSize, setMinClusterSize] = useState(5)
  const [points, setPoints] = useState(null)

  const clusterView = extractClusters(result)

  const formatError = (err) => {
    if (err?.response?.data?.detail) return err.response.data.detail
    if (err?.response?.data?.error) return err.response.data.error
    if (err?.response?.data) return JSON.stringify(err.response.data)
    return err?.message || String(err)
  }

  const pollUntilFinished = async (id) => {
    let last
    for (let i = 0; i < MAX_POLLS; i++) {
      last = await getTask(id)
      setResult({ status: last.status, task_id: id, task: last })
      if (last.status === 'finished' || last.status === 'error') {
        return last
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_MS))
    }
    return { status: 'timeout', task_id: id, task: last }
  }

  async function onRunIndex() {
    if (!userEmail) {
      setResult({ error: 'Enter user email before indexing' })
      return
    }
    setLoading(true)
    setPoints(null)
    try {
      const res = await runIndex({
        user_email: userEmail,
        max_threads: maxThreads,
        page_size: pageSize,
        embed_batch: embedBatch,
        reset: resetIndex,
        log_level: 'INFO',
      })
      setTaskId(res.task_id)
      setTaskKind('index')
      setResult({ status: 'queued', task_id: res.task_id, kind: 'index' })
      const finalRes = await pollUntilFinished(res.task_id)
      setResult({ ...finalRes, task_id: res.task_id, kind: 'index' })
    } catch (err) {
      setResult({ error: formatError(err) })
    } finally {
      setLoading(false)
    }
  }

  async function onRunAsync() {
    if (!userEmail) {
      setResult({ error: 'Enter user email before running clustering' })
      return
    }
    setLoading(true)
    try {
      const res = await runClusters({ user_email: userEmail, min_cluster_size: minClusterSize })
      setTaskId(res.task_id)
      setTaskKind('cluster')
      setResult({ status: 'queued', task_id: res.task_id, kind: 'cluster' })
      const finalRes = await pollUntilFinished(res.task_id)
      setResult({ ...finalRes, task_id: res.task_id, kind: 'cluster' })
    } catch (err) {
      setResult({ error: formatError(err) })
    } finally {
      setLoading(false)
    }
  }

  async function onFetch() {
    if (!userEmail) {
      setResult({ error: 'Enter user email before running clustering' })
      return
    }
    setLoading(true)
    try {
      const res = await fetchClusters({ user_email: userEmail, min_cluster_size: minClusterSize })
      setResult(res)
    } catch (err) {
      setResult({ error: formatError(err) })
    } finally {
      setLoading(false)
    }
  }

  async function onPollTask() {
    if (!taskId) return
    setLoading(true)
    try {
      const res = await getTask(taskId)
      setResult({ status: res.status, task_id: taskId, task: res, kind: taskKind })
    } catch (err) {
      setResult({ error: formatError(err) })
    } finally {
      setLoading(false)
    }
  }

  async function onLoadEmbeddings() {
    if (!userEmail) {
      setResult({ error: 'Enter user email to fetch embeddings' })
      return
    }
    setLoading(true)
    try {
      const res = await fetchEmbeddings(userEmail)
      setPoints(res.points)
      setResult({ status: 'embeddings loaded', count: res.points.length })
    } catch (err) {
      setResult({ error: formatError(err) })
      setPoints(null)
    } finally {
      setLoading(false)
    }
  }

  async function onLoadLastCluster() {
    setLoading(true)
    try {
      const res = await fetchLastCluster()
      setResult(res)
    } catch (err) {
      if (err?.response?.status === 404) {
        setResult({ status: 'empty', message: 'No cluster result available yet. Run clustering first.' })
      } else {
        setResult({ error: formatError(err) })
      }
    } finally {
      setLoading(false)
    }
  }

  const scatter = points
    ? {
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        text: points.map((p) => p.metadata?.subject || p.thread_id),
        mode: 'markers',
        type: 'scattergl',
        marker: { size: 8 },
      }
    : null

  return (
    <div className="app">
      <header>
        <h1>Email Clustering UI</h1>
      </header>
      <main>
        <div className="controls" style={{ marginBottom: 12 }}>
          <div className="card instructions">
            <h3>How to use</h3>
            <ol>
              <li>Start the backend: <code>uvicorn api:app --port 8000</code> from the repo root.</li>
              <li>Start the frontend: <code>cd frontend && npm run dev</code> (uses <code>http://localhost:8000/api</code> by default).</li>
              <li>Enter the same email you indexed; if new, run “Run indexing (async)” first.</li>
              <li>After indexing finishes, run clustering (sync or async) to see grouped threads.</li>
              <li>Use “Load embeddings & show PCA” for the scatter plot, and browse clusters below.</li>
            </ol>
            <p className="hint">
              If you see “No emails have been indexed yet”, restart the backend with the same env vars used during indexing
              (e.g., same <code>OPENAI_API_KEY</code> or none).
            </p>
          </div>
        </div>

        <div className="controls">
          <div className="card">
            <h3>Data</h3>
            <label className="stacked">
              User email
              <input value={userEmail} onChange={(e) => setUserEmail(e.target.value)} placeholder="you@example.com" />
            </label>
            <div className="grid">
              <label className="stacked">
                Max threads
                <input type="number" min="1" value={maxThreads} onChange={(e) => setMaxThreads(Number(e.target.value || 0))} />
              </label>
              <label className="stacked">
                Page size
                <input type="number" min="1" value={pageSize} onChange={(e) => setPageSize(Number(e.target.value || 0))} />
              </label>
              <label className="stacked">
                Embed batch
                <input type="number" min="1" value={embedBatch} onChange={(e) => setEmbedBatch(Number(e.target.value || 0))} />
              </label>
            </div>
            <label className="checkbox">
              <input type="checkbox" checked={resetIndex} onChange={(e) => setResetIndex(e.target.checked)} />
              Reset existing embeddings before indexing
            </label>
            <button onClick={onRunIndex} disabled={loading}>Run indexing (async)</button>
          </div>

          <div className="card">
            <h3>Clustering</h3>
            <div className="grid">
              <label className="stacked">
                Min cluster size
                <input type="number" min="2" value={minClusterSize} onChange={(e) => setMinClusterSize(Number(e.target.value || 0))} />
              </label>
            </div>
            <button onClick={onFetch} disabled={loading}>Run sync clustering</button>
            <button onClick={onRunAsync} disabled={loading}>Run async clustering (auto-poll)</button>
            <button onClick={onPollTask} disabled={loading || !taskId}>Poll last task</button>
            <button onClick={onLoadLastCluster} disabled={loading}>Load last cluster result</button>
          </div>

          <div className="card">
            <h3>Visualization</h3>
            <button onClick={onLoadEmbeddings} disabled={loading}>Load embeddings & show PCA</button>
            <p className="hint">Scatter uses current user email. Index + cluster first, then refresh.</p>
          </div>
        </div>

        <section className="result">
          <h2>Result</h2>
          <div className="result-meta">
            {taskKind && <span className="pill">{taskKind}</span>}
            {taskId && <span className="pill muted">task {taskId}</span>}
          </div>
          <pre>{result ? JSON.stringify(result, null, 2) : 'No result yet'}</pre>
        </section>

        {scatter && (
          <section style={{ marginTop: 20 }}>
            <h2>Embedding PCA Scatter</h2>
            <Plot data={[scatter]} layout={{ width: 800, height: 600 }} />
          </section>
        )}

        {clusterView.clusters.length > 0 && (
          <section className="cluster-board">
            <h2>Clusters</h2>
            <div className="cluster-grid">
              {clusterView.clusters.map((c) => (
                <div className="cluster-card" key={c.cluster_id || c.label}>
                  <div className="cluster-header">
                    <div className="pill">Cluster {c.cluster_id ?? ''}</div>
                    <div className="pill muted">{(c.thread_count ?? c.threads?.length ?? 0)} threads</div>
                  </div>
                  <div className="cluster-label">{c.label || 'Unlabeled cluster'}</div>
                  <ul>
                    {(c.threads || []).map((t) => (
                      <li key={t.thread_id || `${c.cluster_id}-${t.subject}`}>
                        <div className="subject">{t.subject || '(no subject)'}</div>
                        <div className="meta-line">{t.senders || 'Unknown senders'}</div>
                        <div className="meta-line small">{t.last_date}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        )}

        {clusterView.noise.length > 0 && (
          <section className="cluster-board">
            <h2>Noise / Outliers</h2>
            <div className="cluster-card">
              <ul>
                {clusterView.noise.map((t, idx) => (
                  <li key={t.thread_id || idx}>
                    <div className="subject">{t.subject || '(no subject)'}</div>
                    <div className="meta-line">{t.senders || 'Unknown senders'}</div>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

function extractClusters(res) {
  const payload = res?.result?.results || res?.results?.results || res?.results || null
  const clustersObj = payload?.clusters || null
  const clusters = clustersObj ? Object.values(clustersObj).sort((a, b) => (a.cluster_id || 0) - (b.cluster_id || 0)) : []
  const noise = payload?.noise_threads || []
  return { clusters, noise }
}
