"""
Cross-Validation Harness and Metric Evaluation
-----------------------------------------------
Group-aware cross-validation (GroupKFold by trip_id) to strictly prevent
data leakage across trips, computing both standard ML metrics and
domain-specific Map-Matching metrics.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, roc_auc_score, confusion_matrix
from typing import Any


def evaluate_observation_top1(df: pd.DataFrame, pred_probs: np.ndarray) -> float:
    """Computes Top-1 observation accuracy (% where top-ranked candidate is the true arc)."""
    sub = df[["trip_id", "observation_idx", "is_true_match"]].copy()
    sub["prob"] = pred_probs
    top_indices = sub.groupby(["trip_id", "observation_idx"])["prob"].idxmax()
    return float(sub.loc[top_indices, "is_true_match"].mean())


def evaluate_model_performance(model: Any, X_val: np.ndarray, y_val: np.ndarray, df_val: pd.DataFrame) -> dict:
    """Evaluates classifier on validation set and computes all Table 3.1 metrics."""
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        margins = model.decision_function(X_val)
        probs = 1.0 / (1.0 + np.exp(-margins))
    else:
        probs = model.predict(X_val).astype(float)
    infer_time_s = time.perf_counter() - t0

    preds = (probs >= 0.5).astype(int)

    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    precision = float(precision_score(y_val, preds, zero_division=0))
    recall = float(recall_score(y_val, preds, zero_division=0))

    try:
        loss = float(log_loss(y_val, np.clip(probs, 1e-7, 1.0 - 1e-7)))
    except Exception:
        loss = float("nan")

    try:
        auc = float(roc_auc_score(y_val, probs))
    except Exception:
        auc = float("nan")

    top1 = evaluate_observation_top1(df_val, probs)
    n_obs = df_val.groupby(["trip_id", "observation_idx"]).ngroups
    ms_per_obs = (infer_time_s / n_obs * 1000.0) if n_obs > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "top1_accuracy": top1,
        "precision": precision,
        "recall": recall,
        "log_loss": loss,
        "roc_auc": auc,
        "infer_ms_per_obs": ms_per_obs,
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
    }


def run_group_kfold_cv(model_cls: Any, model_kwargs: dict, df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> tuple[dict, list[dict]]:
    """Runs Group-K-Fold cross-validation grouped by trip_id."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["trip_id"].values
    X = df[feature_cols].values
    y = df["is_true_match"].values

    fold_results = []
    for f_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        df_va = df.iloc[val_idx]

        m = model_cls(**model_kwargs)
        m.fit(X_tr, y_tr)
        res = evaluate_model_performance(m, X_va, y_va, df_va)
        res["fold"] = f_idx + 1
        fold_results.append(res)

    summary = {
        "cv_macro_f1": float(np.mean([r["macro_f1"] for r in fold_results])),
        "cv_top1_accuracy": float(np.mean([r["top1_accuracy"] for r in fold_results])),
        "cv_log_loss": float(np.nanmean([r["log_loss"] for r in fold_results])),
        "cv_roc_auc": float(np.nanmean([r["roc_auc"] for r in fold_results])),
        "cv_precision": float(np.mean([r["precision"] for r in fold_results])),
        "cv_recall": float(np.mean([r["recall"] for r in fold_results])),
        "cv_infer_ms_per_obs": float(np.mean([r["infer_ms_per_obs"] for r in fold_results])),
    }
    return summary, fold_results
