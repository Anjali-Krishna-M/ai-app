# MCA-Level Phishing URL Detection Web App

A complete phishing URL detection system using Flask + Scikit-learn + EDA visualizations.

## Features
- Real-world dataset ingestion:
  - Phishing URLs from an open PhishTank-style feed
  - Legitimate URLs from Tranco top domains
- Full preprocessing pipeline:
  - Lowercasing / normalization
  - Duplicate + missing value removal
  - Label encoding (`1=phishing`, `0=legitimate`)
- Feature engineering:
  - Lexical, structural, and typosquatting features
- Multi-model comparison:
  - Logistic Regression, Random Forest, Decision Tree, SVM, Gradient Boosting
- Visualizations:
  - Matplotlib + Seaborn (saved to `static/plots/`)
  - Chart.js (frontend interactive charts)
- Flask routes:
  - `/` home prediction page
  - `/predict` URL prediction API (POST)
  - `/dashboard` analytics dashboard

## Project structure
- `feature_extraction.py`
- `visualization.py`
- `train_model.py`
- `app.py`
- `templates/`
- `static/`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model + generate plots
```bash
python train_model.py
```

This creates:
- `models/phishing_detector.joblib`
- `models/model_metrics.json`
- plot images in `static/plots/`

## Run app
```bash
python app.py
```
Open `http://127.0.0.1:5000`.

## Reproducibility
- Fixed random seed = `42`
- Deterministic train/test split = `80/20`
