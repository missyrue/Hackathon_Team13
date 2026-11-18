"""
Bayesian Hyperparameter Optimization using Optuna
--------------------------------------------------
Tunes hyperparameters on the 80 training trips using GroupKFold
cross-validation to maximize validation Macro F1 score.
"""

from __future__ import annotations
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import joblib
from pathlib import Path

from src.pipeline.features import FEATURE_COLUMNS


def run_bayesian_optimization(
    df_train: pd.DataFrame,
    n_trials: int = 30,
    model_type: str = "rf",
    save_path: str = "artifacts/best_tuned_model.joblib",
) -> tuple[dict, optuna.Study]:
    """
    Runs Optuna Bayesian Optimization over the training DataFrame.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df_train[FEATURE_COLUMNS].values
    y = df_train["is_true_match"].values
    groups = df_train["trip_id"].values
    gkf = GroupKFold(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        if model_type == "rf":
            n_estimators = trial.suggest_int("n_estimators", 80, 250, step=20)
            max_depth = trial.suggest_int("max_depth", 8, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 4, 16)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 8)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )

        f1_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
            f1_scores.append(f1)

        return float(np.mean(f1_scores))

    print(f"\nStarting Bayesian Optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print("\nBest Trial Results:")
    print(f"  Best Validation Macro F1 : {study.best_value:.4f}")
    print("  Optimal Parameters:")
    for param, val in study.best_params.items():
        print(f"    - {param}: {val}")

    # Retrain best model on full training dataset
    best_params = study.best_params
    final_model = RandomForestClassifier(
        **best_params,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    # Save artifact
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "best_params": best_params,
        "best_cv_f1": study.best_value,
        "feature_cols": FEATURE_COLUMNS,
    }, save_path)
    print(f"Successfully saved tuned model artifact to {save_path}")

    return best_params, study
