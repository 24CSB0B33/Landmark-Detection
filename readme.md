# Landmark Detection Web Application

A clean, responsive, stutter-free web application for classifying architectural landmarks and monuments using a trained Deep Learning model (`Model.keras` / VGG19) across 3,539 categories.

## Features

- **Drag-and-Drop Image Upload**: Instant image selection with zero UI stutter, live image preview, and file metadata display.
- **Minimal, Neutral Design**: Modern slate aesthetic with high contrast and without exaggerated or garish neon colors.
- **Percentage Probability Display**: Large, prominent confidence percentage for the best match, along with a visual progress meter.
- **Top 5 Predictions Breakdown**: Ranked runner-up candidate predictions with individual confidence percentages.
- **Pre-warmed Model Pipeline**: Background warm-up inference at server startup prevents first-request latency.
- **Built-in Sample Landmarks**: 1-click test landmarks (Historic Cathedral, Monument Tower, Ancient Archway) for instant testing.

## Tech Stack

- **Frontend**: React, Vite, HTML5, Vanilla CSS with custom properties (`:root`).
- **Backend API**: Python, Flask, Flask-CORS, Pillow.
- **Model**: TensorFlow / Keras (VGG19 trained on Google Landmarks Dataset, 3,539 classes).

## Quick Start

### Option 1: One-Click Launch (Standalone)
Double-click `start_app.bat` or run:
```bash
& "C:\Users\chand\anaconda3\envs\Py_code\python.exe" server.py
```
Open your browser at **`http://127.0.0.1:5000`**.

### Option 2: Full Development Mode (with Vite HMR)
1. In Terminal 1, run the Flask backend:
   ```bash
   & "C:\Users\chand\anaconda3\envs\Py_code\python.exe" server.py
   ```
2. In Terminal 2, run the Vite frontend:
   ```bash
   cd frontend
   npm run dev
   ```
   Open your browser at **`http://127.0.0.1:5173`**.

## API Endpoints

- `GET /api/health` - Server status, classes count, and model readiness.
- `GET /api/samples` - Curated sample images for instant test detection.
- `POST /api/predict` - Accepts multipart form data with image file (`image`), returns top-5 predictions with names, IDs, probabilities, and inference latency.