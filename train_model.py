from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature_extraction import extract_url_features, normalize_url
from visualization import (save_class_distribution, save_correlation_heatmap,
                           save_feature_distribution, save_model_comparison)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_PLOTS = BASE_DIR / "static" / "plots"
MODEL_DIR = BASE_DIR / "models"

PHISHING_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
LEGIT_URL = "https://raw.githubusercontent.com/sushil79g/tranco-list/main/top-1m.csv"


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


def _download_if_missing() -> tuple[Path, Path]:
    DATA_DIR.mkdir(exist_ok=True)
    phishing_path = DATA_DIR / "phishing_urls.txt"
    legit_path = DATA_DIR / "legit_urls.csv"

    if not phishing_path.exists():
        phishing_path.write_text(requests.get(PHISHING_URL, timeout=60).text, encoding="utf-8")
    if not legit_path.exists():
        legit_path.write_text(requests.get(LEGIT_URL, timeout=60).text, encoding="utf-8")
    return phishing_path, legit_path


def load_real_world_dataset(sample_size: int = 12000) -> pd.DataFrame:
    phishing_path, legit_path = _download_if_missing()

    phishing_df = pd.read_csv(phishing_path, names=["url"], comment="#", on_bad_lines="skip")
    legit_df = pd.read_csv(legit_path, names=["rank", "domain"], on_bad_lines="skip")

    legit_df["url"] = "https://" + legit_df["domain"].astype(str)

    phishing_df = phishing_df[["url"]].dropna().head(sample_size)
    legit_df = legit_df[["url"]].dropna().head(sample_size)

    phishing_df["label"] = 1
    legit_df["label"] = 0

    df = pd.concat([phishing_df, legit_df], ignore_index=True)
    df["url"] = df["url"].astype(str).map(normalize_url)
    df = df.drop_duplicates(subset=["url"]).dropna(subset=["url"])
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df["url"].map(lambda u: extract_url_features(u).to_dict()).apply(pd.Series)
    return pd.concat([df, features], axis=1)


def train() -> None:
    STATIC_PLOTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_real_world_dataset()
    featured_df = build_feature_frame(df)

    save_class_distribution(featured_df, STATIC_PLOTS)
    save_feature_distribution(featured_df, "url_length", "URL Length Distribution", "url_length_distribution.png", STATIC_PLOTS)
    save_feature_distribution(featured_df, "special_char_count", "Special Character Distribution", "special_char_distribution.png", STATIC_PLOTS)
    save_feature_distribution(featured_df, "domain_length", "Domain Length Comparison", "domain_length_distribution.png", STATIC_PLOTS)
    save_correlation_heatmap(featured_df.drop(columns=["url"]), STATIC_PLOTS)

    feature_cols = [c for c in featured_df.columns if c not in {"url", "label"}]
    X = featured_df[feature_cols]
    y = featured_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    records = []
    trained = {}
    for name, estimator in MODELS.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        }
        records.append(metrics)
        trained[name] = pipeline

    metrics_df = pd.DataFrame(records).sort_values("f1", ascending=False)
    save_model_comparison(metrics_df[["model", "accuracy", "precision", "recall", "f1"]], STATIC_PLOTS)

    best_name = metrics_df.iloc[0]["model"]
    best_model = trained[best_name]

    y_best_prob = best_model.predict_proba(X_test)[:, 1]
    y_best_pred = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_best_pred, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(STATIC_PLOTS / "best_confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_best_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{best_name} AUC={roc_auc_score(y_test, y_best_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve (Best Model)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(STATIC_PLOTS / "best_roc_curve.png")
    plt.close(fig)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_cols,
            "model_metrics": records,
            "best_model_name": best_name,
        },
        MODEL_DIR / "phishing_detector.joblib",
    )
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train()
