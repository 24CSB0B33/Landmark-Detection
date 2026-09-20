import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Engine definitions
const ENGINES = [
  {
    id: 'clip',
    label: 'CLIP Semantic (Recommended)',
    description: 'Zero-shot recognition — works for any world landmark',
  },
  {
    id: 'regional',
    label: 'Google Landmark DB',
    description: 'Asia, Europe & North America — 297,000+ categories',
  },
  {
    id: 'custom',
    label: 'Custom Model.keras',
    description: 'Original notebook VGG19 model (3,539 classes)',
  },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl]     = useState(null);
  const [fileInfo, setFileInfo]         = useState(null);
  const [isDragging, setIsDragging]     = useState(false);
  const [isLoading, setIsLoading]       = useState(false);
  const [error, setError]               = useState(null);
  const [results, setResults]           = useState(null);
  const [samples, setSamples]           = useState([]);
  const [serverStatus, setServerStatus] = useState('checking');
  const [engine, setEngine]             = useState('clip');
  const [modelMeta, setModelMeta]       = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch('/api/health')
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(d => { setServerStatus('online'); setModelMeta(d.models); })
      .catch(() => setServerStatus('offline'));

    fetch('/api/samples')
      .then(r => r.json())
      .then(d => { if (d.samples) setSamples(d.samples); })
      .catch(() => {});
  }, []);

  const fmtBytes = bytes => {
    if (!bytes) return '';
    if (bytes < 1024)    return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
  };

  const handleFile = file => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, WebP).');
      return;
    }
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setFileInfo({ name: file.name, size: fmtBytes(file.size) });
    setError(null);
    setResults(null);
  };

  const handleDrop = e => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleSelectSample = async sample => {
    try {
      setIsLoading(true); setError(null); setResults(null);
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], `${sample.name.replace(/\s+/g,'_')}.jpg`, { type: 'image/jpeg' });
      if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setFileInfo({ name: sample.name + '.jpg', size: fmtBytes(blob.size) });
    } catch (e) { setError('Could not load sample: ' + e.message); }
    finally { setIsLoading(false); }
  };

  const handleClear = () => {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null); setPreviewUrl(null); setFileInfo(null);
    setResults(null); setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDetect = async () => {
    if (!selectedFile) return;
    setIsLoading(true); setError(null);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('engine', engine);
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Detection failed.');
      setResults(data);
    } catch (e) {
      setError(e.message || 'Error communicating with server.');
    } finally {
      setIsLoading(false);
    }
  };

  const badgeLabel = () => {
    if (serverStatus !== 'online') return serverStatus === 'offline' ? 'Backend Offline' : 'Connecting...';
    if (engine === 'clip')     return `CLIP Engine · ${(modelMeta?.clip_landmarks_count || 140)} Landmarks`;
    if (engine === 'regional') return `Google DB · ${(modelMeta?.global_classes_count || 297509).toLocaleString()} Classes`;
    return `Custom Model · ${(modelMeta?.custom_classes_count || 3539).toLocaleString()} Classes`;
  };

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-badge-row">
          <div className="model-badge">
            <span className={`status-dot ${serverStatus === 'offline' ? 'offline' : ''}`} />
            {badgeLabel()}
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            CLIP · VGG19 · TFLite
          </span>
        </div>
        <h1 className="app-title">Landmark Detection</h1>
        <p className="app-subtitle">
          Upload any architectural or monument photo — the app identifies the landmark and shows
          percentage confidence probabilities.
        </p>

        {/* Engine Switcher */}
        <div className="engine-switcher">
          {ENGINES.map(e => (
            <button
              key={e.id}
              type="button"
              className={`engine-tab ${engine === e.id ? 'active' : ''}`}
              title={e.description}
              onClick={() => { setEngine(e.id); setResults(null); }}
            >
              {e.label}
            </button>
          ))}
        </div>
        <p className="engine-desc">
          {ENGINES.find(e => e.id === engine)?.description}
        </p>
      </header>

      {/* ── Main Grid ── */}
      <main className="main-grid">

        {/* LEFT — Image Input */}
        <section className="card">
          <div className="card-title">
            <span>Input Image</span>
            {previewUrl && (
              <button type="button" className="btn-secondary"
                style={{ padding: '4px 10px', fontSize: '0.78rem' }}
                onClick={handleClear}>
                Clear
              </button>
            )}
          </div>

          <input ref={fileInputRef} type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />

          {!previewUrl ? (
            <div className={`dropzone ${isDragging ? 'active' : ''}`}
              onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={e => { e.preventDefault(); setIsDragging(false); }}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}>
              <svg className="dropzone-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="dropzone-prompt">
                Drag &amp; drop your photo here, or <span className="dropzone-browse">browse</span>
              </p>
              <p className="dropzone-hint">JPEG, PNG, WebP — any world landmark</p>
            </div>
          ) : (
            <>
              <div className="preview-wrapper">
                <img src={previewUrl} alt="Selected landmark" className="preview-img" />
                <div className="preview-overlay">
                  <button type="button" className="overlay-btn"
                    onClick={() => fileInputRef.current?.click()}>
                    Change
                  </button>
                </div>
              </div>
              {fileInfo && (
                <div className="file-meta">
                  <span style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {fileInfo.name}
                  </span>
                  <span>{fileInfo.size}</span>
                </div>
              )}
            </>
          )}

          {/* Quick Samples */}
          {samples.length > 0 && (
            <div className="samples-section">
              <div className="samples-label">Quick test samples:</div>
              <div className="samples-row">
                {samples.map(s => (
                  <button key={s.id} type="button" className="sample-chip"
                    onClick={() => handleSelectSample(s)} disabled={isLoading}>
                    <img src={s.url} alt={s.name} className="sample-chip-thumb" />
                    <span>{s.name}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="error-banner">
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          <div className="action-row">
            <button type="button" className="btn-primary"
              disabled={!selectedFile || isLoading || serverStatus === 'offline'}
              onClick={handleDetect}>
              {isLoading ? (
                <><span className="spinner" /><span>Analyzing Landmark...</span></>
              ) : (
                <>
                  <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span>Detect Landmark</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* RIGHT — Results */}
        <section className="card">
          <div className="card-title">
            <span>Detection Results</span>
            {results && (
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {results.inference_time_ms} ms
              </span>
            )}
          </div>

          {!results && !isLoading ? (
            <div className="empty-results">
              <svg className="empty-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <div className="empty-title">No Predictions Yet</div>
              <p className="empty-desc">
                Upload an image and click <strong>Detect Landmark</strong> to see results.
              </p>
            </div>
          ) : isLoading ? (
            <div className="empty-results">
              <div className="spinner" style={{
                width: 32, height: 32,
                borderColor: 'var(--border-color)',
                borderTopColor: 'var(--accent-primary)',
                marginBottom: 16
              }} />
              <div className="empty-title">Running Analysis</div>
              <p className="empty-desc">Matching against global landmark database...</p>
            </div>
          ) : (
            <>
              {/* Best Match Card */}
              <div className="best-match-card">
                <div className="best-match-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="eyebrow-tag">Best Match</span>
                    {results.best_match.region && (
                      <span className="region-tag">{results.best_match.region}</span>
                    )}
                  </div>
                  <span className="inference-badge">
                    {results.model_info}
                  </span>
                </div>

                <div className="best-match-body">
                  <div className="best-match-info">
                    <h2 className="best-match-name">{results.best_match.name}</h2>
                    {/* Show city/country for CLIP results */}
                    {results.best_match.city && (
                      <div className="best-match-location">
                        📍 {results.best_match.city}{results.best_match.country ? `, ${results.best_match.country}` : ''}
                      </div>
                    )}
                    <div className="best-match-id">
                      {results.engine === 'clip'
                        ? `Landmark #${results.best_match.landmark_id + 1} of ${results.model_info.match(/\d+/)?.[0] || ''}`
                        : `Landmark Class #${results.best_match.landmark_id}`}
                    </div>
                  </div>

                  {/* Big Percentage */}
                  <div className="percentage-box">
                    <div className="percentage-number">
                      {results.best_match.confidence_formatted}
                    </div>
                    <div className="percentage-label">Confidence</div>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="progress-track">
                  <div className="progress-fill"
                    style={{ width: `${Math.min(100, Math.max(2, results.best_match.probability))}%` }} />
                </div>
              </div>

              {/* Candidate List */}
              {results.top_matches?.length > 1 && (
                <>
                  <div className="candidates-title">Candidate Predictions</div>
                  <div className="candidates-list">
                    {results.top_matches.map(item => (
                      <div key={item.rank} className="candidate-item">
                        <span className="candidate-rank">#{item.rank}</span>
                        <div className="candidate-details">
                          <div className="candidate-name" title={item.name}>
                            {item.name}
                            {item.region && <span className="region-tag">{item.region}</span>}
                          </div>
                          <div className="candidate-id">
                            {item.city ? `${item.city}${item.country ? ', ' + item.country : ''}` : `ID: ${item.landmark_id}`}
                          </div>
                        </div>
                        <div className="candidate-metric">
                          <div className="candidate-percent">{item.confidence_formatted}</div>
                          <div className="candidate-mini-bar">
                            <div className="candidate-mini-fill"
                              style={{ width: `${Math.min(100, Math.max(1, item.probability))}%` }} />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>Landmark Detection · CLIP Semantic Engine · React + Flask</span>
      </footer>
    </div>
  );
}
