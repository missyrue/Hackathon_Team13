"""
PathSnap Model Zoo (5 Distinct Machine Learning Model Families)
---------------------------------------------------------------
1. Ensemble: Random Forest Classifier
2. Boosting: LightGBM Classifier (or GradientBoostingClassifier fallback)
3. Neural Network: Multi-Layer Perceptron (MLPClassifier)
4. Instance-based: Distance-Weighted K-Nearest Neighbors (KNeighborsClassifier)
5. Linear & Probabilistic: Logistic Regression Baseline
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier
    HAS_LGBM = False

from src.pipeline.features import FEATURE_COLUMNS
from src.models.cv_harness import run_group_kfold_cv


def get_model_specs() -> dict[str, dict]:
    """Returns definitions and default hyperparameters for all 5 models."""
    return {
        "Random Forest": {
            "family": "Tree-based Ensemble",
            "constructor": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1
            ),
            "needs_scaling": False,
        },
        "LightGBM": {
            "family": "Gradient Boosting",
            "constructor": lambda: LGBMClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.08,
                random_state=42, n_jobs=-1, verbose=-1
            ) if HAS_LGBM else LGBMClassifier(max_iter=120, max_depth=6, random_state=42),
            "needs_scaling": False,
        },
        "MLP (Neural Net)": {
            "family": "Neural Network",
            "constructor": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True))
            ]),
            "needs_scaling": True,
        },
        "Distance-Weighted KNN": {
            "family": "Instance-based",
            "constructor": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1))
            ]),
            "needs_scaling": True,
        },
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "constructor": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
            ]),
            "needs_scaling": True,
        },
    }


def compare_all_models(df_train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Evaluates all 5 models with GroupKFold cross-validation on df_train
    and produces the Table 3.1 Model Comparison Summary.
    """
    specs = get_model_specs()
    results = []

    print(f"\nComparing {len(specs)} models across distinct algorithm families...")

    for model_name, spec in specs.items():
        print(f"--> Training and evaluating {model_name} ({spec['family']})...")
        wrapper_cls = spec["constructor"]
        # Use dummy callable wrapper for cv_harness
        class ModelWrapper:
            def __init__(self):
                self.m = wrapper_cls()
            def fit(self, X, y):
                self.m.fit(X, y)
                return self
            def predict_proba(self, X):
                if hasattr(self.m, "predict_proba"):
                    return self.m.predict_proba(X)
                elif hasattr(self.m, "decision_function"):
                    dec = self.m.decision_function(X)
                    p1 = 1.0 / (1.0 + np.exp(-dec))
                    return np.column_stack([1.0 - p1, p1])
                preds = self.m.predict(X).astype(float)
                return np.column_stack([1.0 - preds, preds])
            def predict(self, X):
                return self.m.predict(X)

        summary, _ = run_group_kfold_cv(ModelWrapper, {}, df_train, FEATURE_COLUMNS, n_splits=n_splits)

        results.append({
            "Model Name": model_name,
            "Algorithm Family": spec["family"],
            "CV Macro F1": round(summary["cv_macro_f1"], 4),
            "Top-1 Accuracy (%)": round(summary["cv_top1_accuracy"] * 100, 2),
            "Log Loss": round(summary["cv_log_loss"], 4),
            "ROC-AUC": round(summary["cv_roc_auc"], 4),
            "Precision": round(summary["cv_precision"], 4),
            "Recall": round(summary["cv_recall"], 4),
            "Inference Latency (ms/obs)": round(summary["cv_infer_ms_per_obs"], 2),
        })

    return pd.DataFrame(results)
