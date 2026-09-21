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
import onnxruntime as ort
from tokenizers import Tokenizer

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")
DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
CLIP_DIR = os.path.join(MODELS_DIR, "clip")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CLIP Semantic Engine  (primary — most accurate)
# ─────────────────────────────────────────────────────────────
CLIP_MODEL_PATH = os.path.join(CLIP_DIR, "model_quantized.onnx")
CLIP_TOKENIZER_PATH = os.path.join(CLIP_DIR, "tokenizer.json")
CLIP_EMBEDDINGS_PATH = os.path.join(CLIP_DIR, "landmarks_embeddings.npy")
CLIP_DB_PATH = os.path.join(CLIP_DIR, "landmarks_db.json")

clip_session = None
clip_tokenizer = None
clip_text_embeds = None
clip_db = []

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

if all(os.path.exists(p) for p in [CLIP_MODEL_PATH, CLIP_TOKENIZER_PATH, CLIP_EMBEDDINGS_PATH, CLIP_DB_PATH]):
    print("[Server] Loading CLIP semantic landmark engine...")
    clip_session = ort.InferenceSession(CLIP_MODEL_PATH)
    clip_tokenizer = Tokenizer.from_file(CLIP_TOKENIZER_PATH)
    clip_tokenizer.enable_padding(length=77, pad_id=0, pad_token="<|endoftext|>")
    clip_tokenizer.enable_truncation(max_length=77)
    clip_text_embeds = np.load(CLIP_EMBEDDINGS_PATH)
    with open(CLIP_DB_PATH, "r", encoding="utf-8") as f:
        clip_db = json.load(f)
    print(f"[Server] CLIP engine loaded: {len(clip_db)} landmarks, embeddings shape {clip_text_embeds.shape}")
else:
    print("[Server] CLIP model files not found — run build_embeddings.py to generate them.")

# ─────────────────────────────────────────────────────────────
# 2. Regional TFLite Models (Google Landmark Recognition)
# ─────────────────────────────────────────────────────────────
REGIONS = ["asia", "europe", "north_america"]
tflite_models = {}
total_classes = 0

print("[Server] Loading regional landmark models and label maps...")
for reg in REGIONS:
    m_path = os.path.join(MODELS_DIR, f"landmarks_classifier_{reg}_V1.tflite")
    lbl_path = os.path.join(MODELS_DIR, f"landmarks_classifier_{reg}_V1_label_map.csv")
    if os.path.exists(m_path) and os.path.exists(lbl_path):
        interp = tf.lite.Interpreter(model_path=m_path)
        interp.allocate_tensors()
        labels_df = pd.read_csv(lbl_path)
        label_dict = dict(zip(labels_df["id"], labels_df["name"]))
        tflite_models[reg] = {
            "interpreter": interp,
            "labels": label_dict,
            "input_index": interp.get_input_details()[0]["index"],
            "output_index": interp.get_output_details()[0]["index"]
        }
        total_classes += len(label_dict)
        print(f"[Server] Loaded {reg.replace('_',' ').title()} model ({len(label_dict):,} classes)")

# ─────────────────────────────────────────────────────────────
# 3. Custom Notebook Model  (Model.keras)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# 4. Warm-up Inferences
# ─────────────────────────────────────────────────────────────
print("[Server] Running warm-up inferences...")
dummy_321 = np.zeros((1, 321, 321, 3), dtype=np.uint8)
for reg, m_info in tflite_models.items():
    m_info["interpreter"].set_tensor(m_info["input_index"], dummy_321)
    m_info["interpreter"].invoke()
    _ = m_info["interpreter"].get_tensor(m_info["output_index"])

if custom_model is not None:
    _ = custom_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

if clip_session is not None:
    dummy_ids = np.zeros((1, 77), dtype=np.int64)
    dummy_mask = np.ones((1, 77), dtype=np.int64)
    dummy_px = np.zeros((1, 3, 224, 224), dtype=np.float32)
    _ = clip_session.run(["image_embeds"], {
        "pixel_values": dummy_px, "input_ids": dummy_ids, "attention_mask": dummy_mask
    })
    print("[Server] CLIP engine warmed up.")

print("[Server] All models warm and ready.")


# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────

def preprocess_clip(pil_img: Image.Image) -> np.ndarray:
    """Resize and normalise an image to CLIP's (1, 3, 224, 224) tensor."""
    img = pil_img.resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD          # CLIP normalisation
    arr = np.transpose(arr, (2, 0, 1))          # HWC → CHW
    return np.expand_dims(arr, axis=0)           # (1, 3, 224, 224)


def predict_clip(pil_img: Image.Image):
    """Zero-shot landmark recognition using CLIP semantic embeddings."""
    pixel_values = preprocess_clip(pil_img)
    dummy_ids  = np.zeros((1, 77), dtype=np.int64)
    dummy_mask = np.ones((1, 77),  dtype=np.int64)

    out = clip_session.run(["image_embeds"], {
        "pixel_values": pixel_values,
        "input_ids": dummy_ids,
        "attention_mask": dummy_mask
    })
    img_emb = out[0]
    img_emb = img_emb / np.linalg.norm(img_emb, axis=-1, keepdims=True)

    # Cosine similarity × 100 temperature scale → softmax probabilities
    sims = (img_emb @ clip_text_embeds.T)[0] * 100.0
    probs = np.exp(sims - np.max(sims))
    probs = probs / np.sum(probs)

    top_indices = np.argsort(probs)[::-1][:5]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        lm = clip_db[idx]
        prob_pct = round(float(probs[idx]) * 100.0, 2)
        results.append({
            "rank": int(rank),
            "landmark_id": int(idx),
            "name": lm["name"],
            "city": lm.get("city", ""),
            "country": lm.get("country", ""),
            "region": lm.get("region", ""),
            "probability": float(prob_pct),
            "confidence_formatted": f"{prob_pct:.2f}%"
        })
    return results


def predict_global(pil_img: Image.Image):
    """Classifies using Google's regional TFLite landmark models."""
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
            name = labels.get(int(idx), f"Landmark #{idx}")
            candidates.append({
                "region": reg.replace("_", " ").title(),
                "landmark_id": int(idx),
                "name": name,
                "score": float(preds[idx])
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    seen, unique = set(), []
    for c in candidates:
        key = c["name"].strip().lower()
        if key not in seen:
            seen.add(key)
            pct = round(c["score"] * 100.0, 2)
            unique.append({
                "rank": len(unique) + 1,
                "landmark_id": c["landmark_id"],
                "name": c["name"],
                "region": c["region"],
                "probability": pct,
                "confidence_formatted": f"{pct:.2f}%"
            })
        if len(unique) >= 5:
            break
    return unique


def predict_custom(pil_img: Image.Image):
    """Classifies using the custom notebook VGG19 model (Model.keras)."""
    if custom_model is None or custom_classes is None:
        raise ValueError("Custom Model.keras is not available.")
    arr = np.expand_dims(
        np.array(pil_img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0,
        axis=0
    )
    preds = custom_model.predict(arr, verbose=0)[0]
    results = []
    for rank, idx in enumerate(np.argsort(preds)[::-1][:5], 1):
        cid = int(custom_classes[idx])
        pct = round(float(preds[idx]) * 100.0, 2)
        info = custom_labels.get(str(cid), {})
        results.append({
            "rank": rank,
            "landmark_id": cid,
            "name": info.get("name", f"Landmark #{cid}"),
            "region": "Custom",
            "probability": pct,
            "confidence_formatted": f"{pct:.2f}%"
        })
    return results


# ─────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "models": {
            "clip_loaded": clip_session is not None,
            "clip_landmarks_count": len(clip_db),
            "global_loaded": len(tflite_models) > 0,
            "global_classes_count": total_classes,
            "regions": list(tflite_models.keys()),
            "custom_loaded": custom_model is not None,
            "custom_classes_count": len(custom_classes) if custom_classes is not None else 0
        }
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # --- Parse engine choice & image ---
        engine = request.form.get("engine", "clip")
        pil_image = None

        if "image" in request.files:
            f = request.files["image"]
            if f.filename == "":
                return jsonify({"error": "Empty file provided"}), 400
            pil_image = Image.open(f.stream)
        elif request.is_json and "image" in request.json:
            img_data = request.json["image"]
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]
            pil_image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            engine = request.json.get("engine", "clip")
        else:
            return jsonify({"error": "No image found in request."}), 400

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # --- Route to correct engine ---
        if engine == "clip" and clip_session is not None:
            top_matches = predict_clip(pil_image)
            model_info = f"CLIP Semantic Engine ({len(clip_db)} Global Landmarks)"
        elif engine == "regional":
            top_matches = predict_global(pil_image)
            model_info = f"Google Landmark Classifier ({total_classes:,} classes)"
        elif engine == "custom" and custom_model is not None:
            top_matches = predict_custom(pil_image)
            model_info = "Custom Notebook Model (3,539 classes)"
        else:
            # Fallback: try CLIP first, then regional
            if clip_session is not None:
                top_matches = predict_clip(pil_image)
                model_info = f"CLIP Semantic Engine ({len(clip_db)} Global Landmarks)"
            else:
                top_matches = predict_global(pil_image)
                model_info = f"Google Landmark Classifier ({total_classes:,} classes)"

        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        return jsonify({
            "success": True,
            "model_info": model_info,
            "engine": engine,
            "best_match": top_matches[0] if top_matches else None,
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
        {"id": 2, "name": "Monument Tower",    "file": "monument_tower.jpg"},
        {"id": 3, "name": "Ancient Archway",   "file": "ancient_arch.jpg"}
    ]
    available = [
        {"id": s["id"], "name": s["name"], "url": f"/sample_images/{s['file']}"}
        for s in sample_files
        if os.path.exists(os.path.join(SAMPLES_DIR, s["file"]))
    ]
    return jsonify({"samples": available})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    if os.path.exists(os.path.join(DIST_DIR, "index.html")):
        return send_from_directory(DIST_DIR, "index.html")
    return jsonify({"message": "Landmark Detection API is online."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Server] Starting Landmark inference server on http://127.0.0.1:{port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
