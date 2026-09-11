import os
import io
import time
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")
DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# 1. Regional High-Accuracy Models (TFHub Google Landmark Recognition)
REGIONS = ["asia", "europe", "north_america"]
tflite_models = {}

print("[Server] Loading regional landmark models and label maps...")
total_classes = 0
for reg in REGIONS:
    m_path = os.path.join(MODELS_DIR, f"landmarks_classifier_{reg}_V1.tflite")
    lbl_path = os.path.join(MODELS_DIR, f"landmarks_classifier_{reg}_V1_label_map.csv")
    if os.path.exists(m_path) and os.path.exists(lbl_path):
        interp = tf.lite.Interpreter(model_path=m_path)
        interp.allocate_tensors()
        labels_df = pd.read_csv(lbl_path)
        # Create fast ID to name dictionary
        label_dict = dict(zip(labels_df["id"], labels_df["name"]))
        tflite_models[reg] = {
            "interpreter": interp,
            "labels": label_dict,
            "input_index": interp.get_input_details()[0]["index"],
            "output_index": interp.get_output_details()[0]["index"]
        }
        total_classes += len(label_dict)
        print(f"[Server] Loaded {reg.replace('_', ' ').title()} model ({len(label_dict):,} classes)")

# 2. Local Custom Model (from workspace notebook)
custom_model = None
custom_classes = None
custom_labels = {}
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "Model.keras")
CUSTOM_CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")
CUSTOM_LABELS_PATH = os.path.join(BASE_DIR, "landmark_labels.json")

if os.path.exists(CUSTOM_MODEL_PATH) and os.path.exists(CUSTOM_CLASSES_PATH):
    try:
        import keras
        print("[Server] Loading custom notebook Model.keras...")
        custom_model = keras.models.load_model(CUSTOM_MODEL_PATH)
        custom_classes = np.load(CUSTOM_CLASSES_PATH)
        if os.path.exists(CUSTOM_LABELS_PATH):
            with open(CUSTOM_LABELS_PATH, "r", encoding="utf-8") as f:
                custom_labels = json.load(f)
        print(f"[Server] Loaded custom Model.keras ({len(custom_classes)} classes)")
    except Exception as e:
        print(f"[Server] Note: Custom model could not be loaded: {e}")

# 3. Warm-up Inference (Eliminates first-request stutter)
print("[Server] Running warm-up inferences...")
dummy_321 = np.zeros((1, 321, 321, 3), dtype=np.uint8)
for reg, m_info in tflite_models.items():
    interp = m_info["interpreter"]
    interp.set_tensor(m_info["input_index"], dummy_321)
    interp.invoke()
    _ = interp.get_tensor(m_info["output_index"])

if custom_model is not None:
    dummy_224 = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = custom_model.predict(dummy_224, verbose=0)
print("[Server] All models warm and ready.")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "models": {
            "global_loaded": len(tflite_models) > 0,
            "global_classes_count": total_classes,
            "regions": list(tflite_models.keys()),
            "custom_loaded": custom_model is not None,
            "custom_classes_count": len(custom_classes) if custom_classes is not None else 0
        }
    })

def predict_global(pil_img):
    """Classifies using high-accuracy regional Google Landmark models."""
    img_321 = pil_img.resize((321, 321), Image.Resampling.BILINEAR)
    arr = np.expand_dims(np.array(img_321, dtype=np.uint8), axis=0)

    candidates = []
    for reg, m_info in tflite_models.items():
        interp = m_info["interpreter"]
        interp.set_tensor(m_info["input_index"], arr)
        interp.invoke()
        preds = interp.get_tensor(m_info["output_index"])[0]

        top_idxs = np.argsort(preds)[::-1][:5]
        labels = m_info["labels"]
        for idx in top_idxs:
            score = float(preds[idx])
            name = labels.get(int(idx), f"Landmark #{idx}")
            candidates.append({
                "region": reg.replace("_", " ").title(),
                "landmark_id": int(idx),
                "name": name,
                "score": score
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by landmark name
    seen_names = set()
    unique_candidates = []
    for c in candidates:
        clean_name = c["name"].strip().lower()
        if clean_name not in seen_names:
            seen_names.add(clean_name)
            prob_percent = round(c["score"] * 100.0, 2)
            unique_candidates.append({
                "rank": len(unique_candidates) + 1,
                "landmark_id": c["landmark_id"],
                "name": c["name"],
                "region": c["region"],
                "probability": prob_percent,
                "confidence_formatted": f"{prob_percent:.2f}%"
            })
        if len(unique_candidates) >= 5:
            break

    return unique_candidates

def predict_custom(pil_img):
    """Classifies using custom local notebook model."""
    if custom_model is None or custom_classes is None:
        raise ValueError("Custom Model.keras is not available.")

    img_224 = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.expand_dims(np.array(img_224, dtype=np.float32) / 255.0, axis=0)

    preds = custom_model.predict(arr, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:5]

    matches = []
    for rank, idx in enumerate(top_indices, start=1):
        class_id = int(custom_classes[idx])
        prob_percent = round(float(preds[idx] * 100.0), 2)
        info = custom_labels.get(str(class_id), {})
        name = info.get("name", f"Landmark #{class_id}")
        matches.append({
            "rank": rank,
            "landmark_id": class_id,
            "name": name,
            "region": "Custom",
            "probability": prob_percent,
            "confidence_formatted": f"{prob_percent:.2f}%"
        })
    return matches

@app.route("/api/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        engine = request.form.get("engine", "global")
        pil_image = None

        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "Empty file provided"}), 400
            pil_image = Image.open(file.stream)
        elif request.is_json and "image" in request.json:
            img_data = request.json["image"]
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]
            img_bytes = base64.b64decode(img_data)
            pil_image = Image.open(io.BytesIO(img_bytes))
            engine = request.json.get("engine", "global")
        else:
            return jsonify({"error": "No image found in request."}), 400

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        if engine == "custom" and custom_model is not None:
            top_matches = predict_custom(pil_image)
            model_info = "Custom Notebook Model (3,539 classes)"
        else:
            top_matches = predict_global(pil_image)
            model_info = f"Google Landmark Classifier ({total_classes:,} classes)"

        best = top_matches[0] if top_matches else None
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            "success": True,
            "model_info": model_info,
            "engine": engine,
            "best_match": best,
            "top_matches": top_matches,
            "inference_time_ms": elapsed_ms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sample_images/<path:filename>")
def serve_sample_image(filename):
    return send_from_directory(SAMPLES_DIR, filename)

@app.route("/api/samples", methods=["GET"])
def get_samples():
    sample_files = [
        {"id": 1, "name": "Historic Cathedral", "file": "historic_cathedral.jpg"},
        {"id": 2, "name": "Monument Tower", "file": "monument_tower.jpg"},
        {"id": 3, "name": "Ancient Archway", "file": "ancient_arch.jpg"}
    ]
    available = []
    for s in sample_files:
        if os.path.exists(os.path.join(SAMPLES_DIR, s["file"])):
            available.append({
                "id": s["id"],
                "name": s["name"],
                "url": f"/sample_images/{s['file']}"
            })
    return jsonify({"samples": available})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    if os.path.exists(os.path.join(DIST_DIR, "index.html")):
        return send_from_directory(DIST_DIR, "index.html")
    return jsonify({"message": "Landmark Detection API is online."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Server] Starting Landmark inference server on http://127.0.0.1:{port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
