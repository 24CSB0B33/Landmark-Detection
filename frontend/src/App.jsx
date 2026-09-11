import React, { useState, useEffect, useRef } from 'react';
import './App.css';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [samples, setSamples] = useState([]);
  const [serverStatus, setServerStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [engine, setEngine] = useState('global'); // 'global' | 'custom'
  const [modelMeta, setModelMeta] = useState(null);

  const fileInputRef = useRef(null);

  // Check backend server status on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error('Server returned error');
        return res.json();
      })
      .then((data) => {
        setServerStatus('online');
        setModelMeta(data.models);
      })
      .catch(() => {
        setServerStatus('offline');
      });

    // Load sample landmarks
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples) setSamples(data.samples);
      })
      .catch(() => {});
  }, []);

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, WebP).');
      return;
    }

    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objectUrl);
    setFileInfo({
      name: file.name,
      size: formatFileSize(file.size),
    });
    setError(null);
    setResults(null);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample) => {
    try {
      setIsLoading(true);
      setError(null);
      setResults(null);

      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], `${sample.name.toLowerCase().replace(/\s+/g, '_')}.jpg`, {
        type: 'image/jpeg',
      });

      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }

      const objectUrl = URL.createObjectURL(file);
      setSelectedFile(file);
      setPreviewUrl(objectUrl);
      setFileInfo({
        name: `${sample.name}.jpg`,
        size: formatFileSize(blob.size),
      });
      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      setError('Could not load sample image: ' + err.message);
    }
  };

  const handleClear = () => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileInfo(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDetect = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('engine', engine);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Landmark detection failed.');
      }

      setResults(data);
    } catch (err) {
      setError(err.message || 'Error communicating with server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-badge-row">
          <div className="model-badge">
            <span
              className={`status-dot ${serverStatus === 'offline' ? 'offline' : ''}`}
            />
            {serverStatus === 'online'
              ? engine === 'global'
                ? `Global Landmark Model (${(modelMeta?.global_classes_count || 250000).toLocaleString()} Classes)`
                : `Custom Notebook Model (${(modelMeta?.custom_classes_count || 3539).toLocaleString()} Classes)`
              : serverStatus === 'offline'
              ? 'Backend Offline'
              : 'Connecting to Model...'}
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Google Landmark Recognition V2
          </span>
        </div>
        <h1 className="app-title">Landmark Detection</h1>
        <p className="app-subtitle">
          Upload an architectural image or monument photo to identify the landmark and view match confidence probability.
        </p>

        {/* Engine Switcher */}
        <div className="engine-switcher">
          <button
            type="button"
            className={`engine-tab ${engine === 'global' ? 'active' : ''}`}
            onClick={() => {
              setEngine('global');
              setResults(null);
            }}
          >
            Global Landmarks (Recommended)
          </button>
          <button
            type="button"
            className={`engine-tab ${engine === 'custom' ? 'active' : ''}`}
            onClick={() => {
              setEngine('custom');
              setResults(null);
            }}
          >
            Original Model.keras (3,539 classes)
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-grid">
        {/* Left Column: Image Input */}
        <section className="card">
          <div className="card-title">
            <span>Input Image</span>
            {previewUrl && (
              <button
                type="button"
                className="btn-secondary"
                style={{ padding: '4px 10px', fontSize: '0.78rem' }}
                onClick={handleClear}
              >
                Clear
              </button>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png, image/jpeg, image/webp"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleFile(e.target.files[0]);
              }
            }}
          />

          {!previewUrl ? (
            <div
              className={`dropzone ${isDragging ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <svg
                className="dropzone-icon"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p className="dropzone-prompt">
                Drag and drop your photo here, or{' '}
                <span className="dropzone-browse">browse</span>
              </p>
              <p className="dropzone-hint">Supports JPEG, PNG, WebP up to 10MB</p>
            </div>
          ) : (
            <div>
              <div className="preview-wrapper">
                <img
                  src={previewUrl}
                  alt="Selected landmark preview"
                  className="preview-img"
                />
                <div className="preview-overlay">
                  <button
                    type="button"
                    className="overlay-btn"
                    onClick={() => fileInputRef.current && fileInputRef.current.click()}
                  >
                    Change
                  </button>
                </div>
              </div>
              {fileInfo && (
                <div className="file-meta">
                  <span style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {fileInfo.name}
                  </span>
                  <span>{fileInfo.size}</span>
                </div>
              )}
            </div>
          )}

          {/* Quick sample selector */}
          {samples.length > 0 && (
            <div className="samples-section">
              <div className="samples-label">Or test with a sample:</div>
              <div className="samples-row">
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    className="sample-chip"
                    onClick={() => handleSelectSample(sample)}
                    disabled={isLoading}
                  >
                    <img
                      src={sample.url}
                      alt={sample.name}
                      className="sample-chip-thumb"
                    />
                    <span>{sample.name}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div className="error-banner">
              <svg
                width="16"
                height="16"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Action Row */}
          <div className="action-row">
            <button
              type="button"
              className="btn-primary"
              disabled={!selectedFile || isLoading || serverStatus === 'offline'}
              onClick={handleDetect}
            >
              {isLoading ? (
                <>
                  <span className="spinner" />
                  <span>Analyzing Landmark...</span>
                </>
              ) : (
                <>
                  <svg
                    width="18"
                    height="18"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                  <span>Detect Landmark</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* Right Column: Prediction Results */}
        <section className="card">
          <div className="card-title">
            <span>Detection Results</span>
            {results && (
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {results.inference_time_ms} ms • {results.model_info}
              </span>
            )}
          </div>

          {!results && !isLoading ? (
            <div className="empty-results">
              <svg
                className="empty-icon"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <div className="empty-title">No Predictions Yet</div>
              <p className="empty-desc">
                Select or upload an image and click <strong>Detect Landmark</strong> to see match predictions and probability percentages.
              </p>
            </div>
          ) : isLoading ? (
            <div className="empty-results">
              <div
                className="spinner"
                style={{
                  width: '32px',
                  height: '32px',
                  borderColor: 'var(--border-color)',
                  borderTopColor: 'var(--accent-primary)',
                  marginBottom: '16px',
                }}
              />
              <div className="empty-title">Running Neural Network</div>
              <p className="empty-desc">
                Evaluating feature maps across {engine === 'global' ? '250,000+' : '3,539'} landmark categories...
              </p>
            </div>
          ) : (
            <div>
              {/* Best Match Spotlight Card */}
              <div className="best-match-card">
                <div className="best-match-header">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span className="eyebrow-tag">Best Match</span>
                    {results.best_match.region && (
                      <span className="region-tag">{results.best_match.region}</span>
                    )}
                  </div>
                  <span className="inference-badge">Rank #1</span>
                </div>

                <div className="best-match-body">
                  <div className="best-match-info">
                    <h2 className="best-match-name">
                      {results.best_match.name}
                    </h2>
                    <div className="best-match-id">
                      Landmark Class #{results.best_match.landmark_id}
                    </div>
                  </div>

                  {/* Percentage Probability Display */}
                  <div className="percentage-box">
                    <div className="percentage-number">
                      {results.best_match.confidence_formatted}
                    </div>
                    <div className="percentage-label">Confidence</div>
                  </div>
                </div>

                {/* Progress bar representing percentage */}
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(100, Math.max(2, results.best_match.probability))}%`,
                    }}
                  />
                </div>
              </div>

              {/* Top Candidates Breakdown */}
              {results.top_matches && results.top_matches.length > 1 && (
                <div>
                  <div className="candidates-title">Candidate Predictions</div>
                  <div className="candidates-list">
                    {results.top_matches.map((item) => (
                      <div key={item.rank} className="candidate-item">
                        <span className="candidate-rank">#{item.rank}</span>
                        <div className="candidate-details">
                          <div className="candidate-name" title={item.name}>
                            {item.name}
                            {item.region && (
                              <span className="region-tag">{item.region}</span>
                            )}
                          </div>
                          <div className="candidate-id">
                            ID: {item.landmark_id}
                          </div>
                        </div>
                        <div className="candidate-metric">
                          <div className="candidate-percent">
                            {item.confidence_formatted}
                          </div>
                          <div className="candidate-mini-bar">
                            <div
                              className="candidate-mini-fill"
                              style={{
                                width: `${Math.min(100, Math.max(1, item.probability))}%`,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span>Google Landmarks Recognition • Fast In-Memory Inference</span>
      </footer>
    </div>
  );
}
