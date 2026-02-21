from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_class_distribution(df, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="label", palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"])
    ax.set_title("Class Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png")
    plt.close(fig)


def save_feature_distribution(df, column: str, title: str, filename: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=column, hue="label", bins=30, kde=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def save_correlation_heatmap(df, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png")
    plt.close(fig)


def save_model_comparison(metrics_df, output_dir: Path) -> None:
    melted = metrics_df.melt(id_vars=["model"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png")
    plt.close(fig)
