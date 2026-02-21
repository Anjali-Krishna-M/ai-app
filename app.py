from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from feature_extraction import TOP_DOMAINS, extract_url_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

app = Flask(__name__)
_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_bundle():
    if _bundle is None:
        raise FileNotFoundError("Model file not found. Run `python train_model.py` first.")
    return _bundle


def explain_prediction(feature_map: dict, model_pipeline, top_n: int = 5):
    model = model_pipeline.named_steps["model"]
    contributions = {}

    if hasattr(model, "coef_"):
        coeffs = model.coef_[0]
        for idx, name in enumerate(feature_map.keys()):
            contributions[name] = abs(coeffs[idx] * feature_map[name])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for idx, name in enumerate(feature_map.keys()):
            contributions[name] = abs(importances[idx] * feature_map[name])
    else:
        for name, value in feature_map.items():
            contributions[name] = abs(value)

    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": name, "impact": float(score)} for name, score in ranked]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/model-metrics")
def model_metrics():
    bundle = load_bundle()
    return jsonify(
        {
            "best_model": bundle["best_model_name"],
            "metrics": bundle["model_metrics"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    bundle = load_bundle()
    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "")
    if not url.strip():
        return jsonify({"error": "URL is required"}), 400

    features = extract_url_features(url).to_dict()
    ordered = [features[c] for c in bundle["feature_columns"]]
    X = np.array([ordered], dtype=float)
    pipeline = bundle["model"]

    prob = float(pipeline.predict_proba(X)[0][1])
    pred = int(prob >= 0.5)

    return jsonify(
        {
            "url": url,
            "prediction": "Phishing" if pred == 1 else "Legitimate",
            "confidence": round(max(prob, 1 - prob) * 100, 2),
            "probabilities": {
                "legitimate": round((1 - prob) * 100, 2),
                "phishing": round(prob * 100, 2),
            },
            "typosquat_hint": any(d in url.lower() for d in TOP_DOMAINS) is False and features["typosquat_flag"] == 1,
            "top_features": explain_prediction(features, pipeline),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
