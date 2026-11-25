"""Train the six spam-detection models and export chart-ready public data.

Both algorithms receive the exact same text-only, context-only, and combined
feature matrices. The only difference between paired models is the classifier.
Thresholds are selected from validation-set precision, not from an assumed
meaning of the model score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError as error:  # Clear instruction rather than silently changing models.
    raise SystemExit(
        "xgboost is required for the final six-model comparison. Install it with "
        "`python -m pip install xgboost` and rerun this script."
    ) from error


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "processed"
FULL_DATA = DATA_DIR / "emails_merged_full.csv"
QUALITY_PATH = DATA_DIR / "data_quality.json"
PUBLIC_DATA_DIR = PROJECT / "docs" / "data"

TEXT_STYLE_COLUMNS = [
    "subject_char_count", "text_char_count", "text_word_count", "url_count",
    "uppercase_rate", "exclamation_rate", "dollar_rate", "digit_rate",
]
CONTEXT_NUMERIC_COLUMNS = [
    "sent_hour", "timezone_offset", "attachment_count", "recipient_count", "received_hops",
]
CONTEXT_BINARY_COLUMNS = [
    "is_html", "has_attachment", "has_reply_to", "has_list_unsubscribe", "is_multipart",
]
CONTEXT_CATEGORICAL_COLUMNS = ["sent_weekday", "sender_tld", "content_type"]
# Triage targets are selected on validation data and checked once on test data.
WARNING_PRECISION_TARGET = 0.75
BLOCK_PRECISION_TARGET = 0.95


@dataclass
class FeatureMatrices:
    text_train: object
    text_validation: object
    text_test: object
    context_train: object
    context_validation: object
    context_test: object

    def get(self, feature_set: str):
        if feature_set == "text":
            return self.text_train, self.text_validation, self.text_test
        if feature_set == "context":
            return self.context_train, self.context_validation, self.context_test
        if feature_set == "combined":
            return (
                hstack([self.text_train, self.context_train]).tocsr(),
                hstack([self.text_validation, self.context_validation]).tocsr(),
                hstack([self.text_test, self.context_test]).tocsr(),
            )
        raise ValueError(f"Unknown feature set: {feature_set}")


def log_then_bin_pipeline(n_bins: int = 6) -> Pipeline:
    """A nonnegative shared representation usable by both model families."""
    return Pipeline(
        steps=[
            ("impute_missing", SimpleImputer(strategy="median")),
            ("log_counts", FunctionTransformer(lambda values: np.log1p(np.clip(values, 0, None)))),
            (
                "quantile_bins",
                KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile", subsample=None),
            ),
        ]
    )


def make_text_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=15000,
                    sublinear_tf=True,
                ),
                "model_text",
            ),
            ("style_bins", log_then_bin_pipeline(), TEXT_STYLE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def make_context_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("context_numeric_bins", log_then_bin_pipeline(), CONTEXT_NUMERIC_COLUMNS),
            (
                "context_binary",
                Pipeline(
                    steps=[
                        ("impute_missing", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CONTEXT_BINARY_COLUMNS,
            ),
            (
                "context_categorical",
                Pipeline(
                    steps=[
                        ("impute_missing", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)),
                    ]
                ),
                CONTEXT_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def split_data(data: pd.DataFrame):
    """Create reproducible 60/20/20 stratified train, validation, and test sets."""
    train, remainder = train_test_split(
        data, test_size=0.40, stratify=data["label"], random_state=42
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=remainder["label"], random_state=42
    )
    return train.copy(), validation.copy(), test.copy()


def build_feature_matrices(train, validation, test) -> FeatureMatrices:
    text_transformer = make_text_transformer()
    context_transformer = make_context_transformer()
    return FeatureMatrices(
        text_train=text_transformer.fit_transform(train),
        text_validation=text_transformer.transform(validation),
        text_test=text_transformer.transform(test),
        context_train=context_transformer.fit_transform(train),
        context_validation=context_transformer.transform(validation),
        context_test=context_transformer.transform(test),
    )


def positive_weights(y_train: pd.Series) -> tuple[np.ndarray, float]:
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    positive_scale = negatives / positives
    weights = np.where(y_train.to_numpy() == 1, positive_scale, 1.0)
    return weights, positive_scale


def choose_precision_threshold(y_true, scores, target_precision: float, min_flagged: int = 20) -> float:
    """Choose the lowest validation score that reaches the requested precision."""
    candidates = np.unique(scores)
    candidates.sort()
    for threshold in candidates:
        flagged = scores >= threshold
        if int(flagged.sum()) < min_flagged:
            continue
        if precision_score(y_true, flagged, zero_division=0) >= target_precision:
            return float(threshold)
    return float(np.nextafter(scores.max(), np.inf))


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    predicted = scores >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "flagged_count": int(predicted.sum()),
        "flagged_share": float(predicted.mean()),
    }


def fit_and_score(algorithm, matrices, y_train, y_validation, y_test, weights, positive_scale):
    x_train, x_validation, x_test = matrices
    if algorithm == "Naive Bayes":
        model = MultinomialNB(alpha=0.5)
        model.fit(x_train, y_train, sample_weight=weights)
    elif algorithm == "XGBoost":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=positive_scale,
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        )
        # XGBoost applies the same positive-class ratio through scale_pos_weight.
        # Do not also pass row weights or the positive class would be weighted twice.
        model.fit(x_train, y_train)
    else:
        raise ValueError(algorithm)

    validation_scores = model.predict_proba(x_validation)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    return model, validation_scores, test_scores


def make_pr_curve(y_true, scores) -> list[dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # Downsample for a small public JSON while retaining endpoints.
    indices = np.unique(np.linspace(0, len(precision) - 1, 160, dtype=int))
    points = []
    for index in indices:
        points.append(
            {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "threshold": float(thresholds[index]) if index < len(thresholds) else 1.0,
            }
        )
    return points


def main() -> None:
    if not FULL_DATA.exists():
        raise FileNotFoundError("Run build_merged_dataset.py before training models.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FULL_DATA)
    train, validation, test = split_data(data)
    matrices = build_feature_matrices(train, validation, test)
    y_train = train["label"]
    y_validation = validation["label"]
    y_test = test["label"]
    weights, positive_scale = positive_weights(y_train)

    model_rows = []
    all_predictions = test[[
        "record_id", "label", "sent_hour", "sent_weekday", "is_html", "has_attachment",
        "attachment_count", "recipient_count", "received_hops", "url_count", "uppercase_rate",
        "text_word_count", "sender_tld", "content_type",
    ]].copy()
    pr_curves = {}
    trained = {}

    for algorithm in ("Naive Bayes", "XGBoost"):
        for feature_set in ("text", "context", "combined"):
            model_name = f"{algorithm} | {feature_set.title()}"
            model, val_scores, test_scores = fit_and_score(
                algorithm,
                matrices.get(feature_set),
                y_train,
                y_validation,
                y_test,
                weights,
                positive_scale,
            )
            warning_threshold = choose_precision_threshold(y_validation, val_scores, WARNING_PRECISION_TARGET)
            block_threshold = choose_precision_threshold(y_validation, val_scores, BLOCK_PRECISION_TARGET)
            warning_test = metrics_at_threshold(y_test, test_scores, warning_threshold)
            block_test = metrics_at_threshold(y_test, test_scores, block_threshold)
            warning_validation = metrics_at_threshold(y_validation, val_scores, warning_threshold)
            block_validation = metrics_at_threshold(y_validation, val_scores, block_threshold)
            model_rows.append(
                {
                    "model": model_name,
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "validation_pr_auc": float(average_precision_score(y_validation, val_scores)),
                    "test_pr_auc": float(average_precision_score(y_test, test_scores)),
                    "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
                    "warning_threshold": warning_threshold,
                    "block_threshold": block_threshold,
                    "warning_precision_target": WARNING_PRECISION_TARGET,
                    "block_precision_target": BLOCK_PRECISION_TARGET,
                    "warning_precision_validation": warning_validation["precision"],
                    "warning_recall_validation": warning_validation["recall"],
                    "warning_flagged_count_validation": warning_validation["flagged_count"],
                    "block_precision_validation": block_validation["precision"],
                    "block_recall_validation": block_validation["recall"],
                    "block_flagged_count_validation": block_validation["flagged_count"],
                    "warning_precision_test": warning_test["precision"],
                    "warning_recall_test": warning_test["recall"],
                    "warning_flagged_count_test": warning_test["flagged_count"],
                    "warning_flagged_share_test": warning_test["flagged_share"],
                    "block_precision_test": block_test["precision"],
                    "block_recall_test": block_test["recall"],
                    "block_flagged_count_test": block_test["flagged_count"],
                    "block_flagged_share_test": block_test["flagged_share"],
                }
            )
            key = f"{algorithm.lower().replace(' ', '_')}_{feature_set}"
            all_predictions[f"score_{key}"] = test_scores
            pr_curves[key] = make_pr_curve(y_test, test_scores)
            trained[key] = {"model": model, "validation_scores": val_scores, "test_scores": test_scores}

    metrics = pd.DataFrame(model_rows).sort_values("validation_pr_auc", ascending=False).reset_index(drop=True)
    best = metrics.iloc[0]
    best_key = f"{best['algorithm'].lower().replace(' ', '_')}_{best['feature_set']}"
    best_scores = trained[best_key]["test_scores"]
    all_predictions["best_model"] = best["model"]
    all_predictions["best_score"] = best_scores
    all_predictions["triage_band"] = np.select(
        [best_scores >= best["block_threshold"], best_scores >= best["warning_threshold"]],
        ["Quarantine", "Warn or review"],
        default="Deliver",
    )
    all_predictions["best_prediction"] = (best_scores >= best["warning_threshold"]).astype(int)
    all_predictions["error_type"] = np.select(
        [
            (all_predictions["label"] == 1) & (all_predictions["best_prediction"] == 0),
            (all_predictions["label"] == 0) & (all_predictions["best_prediction"] == 1),
        ],
        ["Missed spam", "Legitimate email flagged"],
        default="Correct",
    )

    metrics.to_csv(DATA_DIR / "model_metrics.csv", index=False)
    all_predictions.to_csv(DATA_DIR / "test_predictions_safe.csv", index=False)

    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))
    distributions = []
    selected_features = ["url_count", "text_word_count", "received_hops", "attachment_count"]
    for feature in selected_features:
        for label, group in test.groupby("label"):
            values = group[feature].dropna().astype(float)
            distributions.append(
                {
                    "feature": feature,
                    "label": int(label),
                    "count": int(len(values)),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "mean": float(values.mean()),
                }
            )

    public_payload = {
        "summary": {
            "joined_records": quality["joined_records"],
            "test_records": int(len(test)),
            "spam_rate": float(data["label"].mean()),
            "positive_scale_train": float(positive_scale),
            "best_model": best["model"],
            "warning_threshold": float(best["warning_threshold"]),
            "block_threshold": float(best["block_threshold"]),
            "warning_precision_target": WARNING_PRECISION_TARGET,
            "block_precision_target": BLOCK_PRECISION_TARGET,
        },
        "metrics": metrics.to_dict(orient="records"),
        "points": all_predictions.to_dict(orient="records"),
        "distributions": distributions,
        "pr_curves": pr_curves,
        "coverage": [
            {"field": field, "available": quality["joined_records"] - missing}
            for field, missing in quality["missingness"].items()
            if field in {"sent_hour", "sent_weekday", "sender_tld", "content_type", "received_hops"}
        ],
    }
    payload_json = json.dumps(public_payload, indent=2)
    (PUBLIC_DATA_DIR / "website_data.json").write_text(payload_json, encoding="utf-8")
    # A script copy works when index.html is opened directly, where browsers
    # otherwise block fetch() calls to a local JSON file.
    (PUBLIC_DATA_DIR / "website_data.js").write_text(
        "window.WEBSITE_DATA = " + payload_json + ";\n", encoding="utf-8"
    )

    print("Finished six-model evaluation.")
    print(f"Best validation model: {best['model']}")
    print(f"Public website data: {PUBLIC_DATA_DIR / 'website_data.json'}")


if __name__ == "__main__":
    main()
