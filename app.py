from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import pytesseract
    from PIL import Image
except Exception:  # optional dependency at runtime
    pytesseract = None
    Image = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "adshield.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

SCAM_KEYWORDS = {
    "registration fee": "Job Scam",
    "guaranteed returns": "Crypto Scam",
    "double your money": "Investment Scam",
    "urgent": "Phishing",
    "click here": "Phishing",
    "lottery": "Lottery Scam",
    "winner": "Lottery Scam",
    "whatsapp": "Impersonation Scam",
    "limited offer": "Marketing Deception",
    "transfer": "Payment Scam",
}

TRAINING_DATA = [
    ("Join our team, salary and interview schedule attached", 0),
    ("Official sale announcement with verified store details", 0),
    ("University internship opportunity apply via portal", 0),
    ("Guaranteed returns in 24 hours invest now", 1),
    ("Pay registration fee to confirm your job", 1),
    ("You are lottery winner transfer charges to receive prize", 1),
    ("Urgent click here to verify your bank account", 1),
    ("Double your money with secret crypto bot", 1),
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")


# --- Model bootstrap ---
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
model = LogisticRegression(max_iter=1000)
X_train = vectorizer.fit_transform([text for text, _ in TRAINING_DATA])
model.fit(X_train, [label for _, label in TRAINING_DATA])


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_error: Exception | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ad_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            verdict TEXT NOT NULL,
            category TEXT NOT NULL,
            trigger_words TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    db.commit()


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def trigger_words(text: str) -> List[str]:
    cleaned = clean_text(text)
    return [kw for kw in SCAM_KEYWORDS if kw in cleaned]


def categorize(words: List[str]) -> str:
    if not words:
        return "Low Risk / Unclassified"
    return SCAM_KEYWORDS.get(words[0], "Suspicious")


def login_required() -> bool:
    return "user_id" in session


@app.route("/")
def home():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 6:
            flash("Username must be 3+ chars and password must be 6+ chars.")
            return redirect(url_for("signup"))
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (
                    username,
                    generate_password_hash(password, method="pbkdf2:sha256"),
                    datetime.utcnow().isoformat(),
                ),
            )
            db.commit()
            flash("Signup successful. Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            return redirect(url_for("signup"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        flash("Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def analyze_and_store(text: str, user_id: int):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    fake_prob = float(model.predict_proba(X)[0][1])
    words = trigger_words(text)
    category = categorize(words)
    verdict = "Fake" if fake_prob >= 0.5 or words else "Genuine"

    db = get_db()
    db.execute(
        """
        INSERT INTO scans (user_id, ad_text, cleaned_text, fake_probability, verdict, category, trigger_words, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            text,
            cleaned,
            fake_prob,
            verdict,
            category,
            ", ".join(words),
            datetime.utcnow().isoformat(),
        ),
    )
    db.commit()

    tips = [
        "Never pay upfront fees for jobs or prizes.",
        "Verify company identity from official websites.",
        "Avoid clicking urgent links from unknown sources.",
    ]

    return {
        "risk_score": round(fake_prob * 100, 2),
        "verdict": verdict,
        "category": category,
        "trigger_words": words,
        "tips": tips,
    }


@app.post("/api/scan")
def scan_text():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required."}), 400

    return jsonify(analyze_and_store(text, session["user_id"]))


@app.post("/api/ocr-scan")
def ocr_scan():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    if pytesseract is None or Image is None:
        return jsonify({"error": "OCR dependencies are unavailable on this server."}), 500

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Invalid image file."}), 400

    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    extracted_text = pytesseract.image_to_string(Image.open(save_path))
    if not extracted_text.strip():
        return jsonify({"error": "No readable text found in image."}), 400

    return jsonify(analyze_and_store(extracted_text, session["user_id"]))


@app.get("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session.get("username"))


@app.get("/api/stats")
def stats():
    if not login_required():
        return jsonify({"error": "Unauthorized"}), 401

    db = get_db()
    totals = db.execute(
        "SELECT COUNT(*) as total, SUM(CASE WHEN verdict='Fake' THEN 1 ELSE 0 END) as fake_total FROM scans"
    ).fetchone()
    user_history = db.execute(
        """
        SELECT verdict, category, fake_probability, trigger_words, created_at, ad_text
        FROM scans WHERE user_id = ?
        ORDER BY id DESC LIMIT 20
        """,
        (session["user_id"],),
    ).fetchall()

    return jsonify(
        {
            "total_scans": int(totals["total"] or 0),
            "fake_scans": int(totals["fake_total"] or 0),
            "genuine_scans": int((totals["total"] or 0) - (totals["fake_total"] or 0)),
            "history": [dict(row) for row in user_history],
        }
    )


with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
