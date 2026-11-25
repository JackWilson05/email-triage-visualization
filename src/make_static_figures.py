"""Create Python figure exports for the final report and website appendix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "figures"
PUBLIC_FIGURE_DIR = PROJECT / "docs" / "assets"
PUBLIC_DATA = PROJECT / "docs" / "data" / "website_data.json"

HAM = "#3D6F9E"
SPAM = "#C65A52"
AMBER = "#B8892D"
INK = "#243447"
SLATE = "#526779"
MUTED = "#6B7885"
WARNING_PRECISION_TARGET = 0.75
BLOCK_PRECISION_TARGET = 0.95


def base_style() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(PUBLIC_FIGURE_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not (DATA_DIR / "model_metrics.csv").exists():
        raise FileNotFoundError("Run train_and_evaluate_models.py before making figures.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base_style()

    metrics = pd.read_csv(DATA_DIR / "model_metrics.csv")
    predictions = pd.read_csv(DATA_DIR / "test_predictions_safe.csv")
    payload = json.loads(PUBLIC_DATA.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    display = metrics.sort_values("test_pr_auc")
    ax.barh(display["model"], display["test_pr_auc"], color=SLATE)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test average precision")
    ax.set_title("Six-model comparison")
    save(fig, "01_six_model_average_precision.png")

    long_features = predictions.melt(
        id_vars="label",
        value_vars=["url_count", "text_word_count", "received_hops", "attachment_count"],
        var_name="feature",
        value_name="value",
    )
    long_features["label_name"] = long_features["label"].map({0: "Legitimate", 1: "Spam"})
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    for axis, feature in zip(axes.flat, ["url_count", "text_word_count", "received_hops", "attachment_count"]):
        chart = long_features[long_features["feature"] == feature]
        sns.violinplot(data=chart, x="label_name", y="value", palette=[HAM, SPAM], ax=axis, inner="quartile", cut=0)
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("")
    fig.suptitle("Selected signal distributions in the held-out test set", y=1.02, weight="bold")
    save(fig, "02_selected_signal_distributions.png")

    best = payload["summary"]["best_model"]
    text_metric = metrics[(metrics["feature_set"] == "text") & (metrics["algorithm"] == "XGBoost")].iloc[0]
    context_metric = metrics[(metrics["feature_set"] == "context") & (metrics["algorithm"] == "XGBoost")].iloc[0]
    text_score = predictions[f"score_xgboost_text"]
    context_score = predictions[f"score_xgboost_context"]
    fig, ax = plt.subplots(figsize=(7, 5.8))
    for label, color, name in [(0, HAM, "Legitimate"), (1, SPAM, "Spam")]:
        subset = predictions["label"] == label
        ax.scatter(text_score[subset], context_score[subset], s=15, alpha=0.45, color=color, label=name)
    ax.set_xlabel("Text-only XGBoost score")
    ax.set_ylabel("Context-only XGBoost score")
    ax.set_title("Text and context can disagree on the same email")
    ax.legend(frameon=False)
    save(fig, "03_text_context_risk_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for key, curve in payload["pr_curves"].items():
        label = key.replace("_", " ").title()
        feature_set = "combined" if "combined" in key else "context" if "context" in key else "text"
        linestyle = {"text": "-", "context": "--", "combined": ":"}[feature_set]
        color = INK if "naive_bayes" in key else SLATE
        ax.plot([point["recall"] for point in curve], [point["precision"] for point in curve], color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.axhline(payload["summary"]["spam_rate"], color=MUTED, linestyle="--", linewidth=1, label="Spam prevalence")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-recall curves on the held-out test set")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "04_precision_recall_curves.png")

    selected = metrics[metrics["model"] == best].iloc[0]
    warning = selected["warning_threshold"]
    block = selected["block_threshold"]
    thresholds = sorted(set(predictions["best_score"].round(4)))
    rows = []
    for threshold in thresholds[:: max(1, len(thresholds) // 150)]:
        flagged = predictions["best_score"] >= threshold
        if flagged.sum() == 0:
            continue
        precision = (predictions.loc[flagged, "label"] == 1).mean()
        recall = ((predictions["label"] == 1) & flagged).sum() / (predictions["label"] == 1).sum()
        rows.append({"threshold": threshold, "precision": precision, "recall": recall})
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(curve["threshold"], curve["precision"], color=INK, linewidth=2.2, label="Observed test precision")
    ax.plot(curve["threshold"], curve["recall"], color=SLATE, linewidth=2.0, label="Recall")
    ax.axhline(BLOCK_PRECISION_TARGET, color=SLATE, linestyle="--", linewidth=1)
    ax.axhline(WARNING_PRECISION_TARGET, color=AMBER, linestyle="--", linewidth=1)
    ax.axvline(warning, color=AMBER, linestyle=":", linewidth=1.5, label="Warning threshold")
    ax.axvline(block, color=SLATE, linestyle=":", linewidth=1.5, label="Quarantine threshold")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlim(0, 1)
    ax.annotate(
        f"{BLOCK_PRECISION_TARGET:.0%} validation target\n{selected['block_precision_test']:.1%} on test",
        xy=(block, selected["block_precision_test"]),
        xytext=(15, -34),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK},
    )
    ax.set(
        xlabel="Best-model score threshold (symlog scale)",
        ylabel="Held-out performance",
        ylim=(0, 1),
        title="Validation triage cutoffs do not fully hold on test data",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "05_precision_based_triage_thresholds.png")

    print(f"Saved Python figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
