import time
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing, make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


RANDOM_SEEDS = [42, 123, 2026]
N_FOLDS = 5
OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "plots"


@dataclass
class DatasetInfo:
    name: str
    task: str  # 'classification' or 'regression'
    X: pd.DataFrame
    y: pd.Series
    source: str


def build_datasets() -> List[DatasetInfo]:
    # Classification datasets
    bc = load_breast_cancer(as_frame=True)
    wine = load_wine(as_frame=True)

    # Regression datasets
    cali = fetch_california_housing(as_frame=True)
    X_syn, y_syn = make_regression(
        n_samples=1200,
        n_features=12,
        n_informative=10,
        noise=15.0,
        random_state=42,
    )

    datasets = [
        DatasetInfo(
            name="Breast Cancer Wisconsin",
            task="classification",
            X=bc.data,
            y=bc.target,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        ),
        DatasetInfo(
            name="Wine",
            task="classification",
            X=wine.data,
            y=wine.target,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
        ),
        DatasetInfo(
            name="California Housing",
            task="regression",
            X=cali.data,
            y=cali.target,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        ),
        DatasetInfo(
            name="Synthetic Regression",
            task="regression",
            X=pd.DataFrame(X_syn, columns=[f"x{i}" for i in range(X_syn.shape[1])]),
            y=pd.Series(y_syn, name="target"),
            source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html",
        ),
    ]
    return datasets


def get_models_and_grids(task: str) -> Dict[str, Tuple[Any, Dict[str, List[Any]], bool]]:
    if task == "classification":
        return {
            "LogisticRegression": (
                LogisticRegression(max_iter=2000),
                {"model__C": [0.1, 1.0, 10.0]},
                True,
            ),
            "DecisionTree": (
                DecisionTreeClassifier(),
                {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 10]},
                False,
            ),
            "MLP": (
                MLPClassifier(max_iter=800, early_stopping=True),
                {"model__hidden_layer_sizes": [(32,), (64,), (64, 32)], "model__alpha": [0.0001, 0.001]},
                True,
            ),
            "SVM-RBF": (
                SVC(kernel="rbf", probability=True),
                {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale", 0.1, 0.01]},
                True,
            ),
            "GaussianNB": (
                GaussianNB(),
                {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
                False,
            ),
        }

    return {
        "LinearRegression": (
            LinearRegression(),
            {"model__fit_intercept": [True, False]},
            True,
        ),
        "DecisionTree": (
            DecisionTreeRegressor(),
            {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 10]},
            False,
        ),
        "MLP": (
            MLPRegressor(max_iter=800, early_stopping=True),
            {"model__hidden_layer_sizes": [(32,), (64,), (64, 32)], "model__alpha": [0.0001, 0.001]},
            True,
        ),
        "SVM-RBF": (
            SVR(kernel="rbf"),
            {"model__C": [1.0, 5.0, 10.0], "model__gamma": ["scale", 0.1, 0.01]},
            True,
        ),
        "BayesianRidge": (
            BayesianRidge(),
            {"model__alpha_1": [1e-6, 1e-5], "model__lambda_1": [1e-6, 1e-5]},
            True,
        ),
    }


def make_pipeline(model: Any, use_scaling: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict[str, float]:
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    n_classes = len(np.unique(y_true))
    if n_classes == 2 and y_score is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        results["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return results


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def run_experiment(dataset: DatasetInfo) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task = dataset.task
    models = get_models_and_grids(task)
    all_rows = []
    best_params_rows = []

    for seed in RANDOM_SEEDS:
        stratify = dataset.y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=0.3,
            random_state=seed,
            stratify=stratify,
        )

        for model_name, (base_model, grid, use_scaling) in models.items():
            pipe = make_pipeline(clone(base_model), use_scaling)
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed) if task == "classification" else KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            scoring = "f1_weighted" if task == "classification" else "r2"

            gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            train_start = time.perf_counter()
            best_model.fit(X_train, y_train)
            train_time = time.perf_counter() - train_start

            pred_start = time.perf_counter()
            y_pred = best_model.predict(X_test)
            pred_time = time.perf_counter() - pred_start

            if task == "classification":
                y_score = None
                if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):
                    y_score = best_model.predict_proba(X_test)[:, 1]
                metrics = evaluate_classification(y_test.values, y_pred, y_score)
            else:
                metrics = evaluate_regression(y_test.values, y_pred)

            row = {
                "dataset": dataset.name,
                "task": task,
                "seed": seed,
                "model": model_name,
                "train_time_sec": train_time,
                "predict_time_sec": pred_time,
            }
            row.update(metrics)
            all_rows.append(row)

            best_params_rows.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "model": model_name,
                    "best_cv_score": gs.best_score_,
                    "best_params": gs.best_params_,
                }
            )

    return pd.DataFrame(all_rows), pd.DataFrame(best_params_rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results.columns if c not in ["dataset", "task", "seed", "model"]]
    summary = results.groupby(["dataset", "task", "model"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]
    return summary.reset_index()


def plot_key_metrics(summary: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in summary["dataset"].unique():
        ds = summary[summary["dataset"] == dataset_name]
        task = ds["task"].iloc[0]

        plt.figure(figsize=(10, 5))
        if task == "classification":
            y_col = "accuracy_mean"
            y_label = "Accuracy (mean over seeds)"
        else:
            y_col = "r2_mean"
            y_label = "R2 (mean over seeds)"

        sns.barplot(data=ds, x="model", y=y_col)
        plt.xticks(rotation=30)
        plt.title(f"{dataset_name}: Model Comparison ({y_label})")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{dataset_name.replace(' ', '_')}_{y_col}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=ds, x="model", y="train_time_sec_mean")
        plt.xticks(rotation=30)
        plt.title(f"{dataset_name}: Training Time Comparison")
        plt.ylabel("Training time (sec)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{dataset_name.replace(' ', '_')}_train_time.png", dpi=200)
        plt.close()


def save_dataset_overview(datasets: List[DatasetInfo]) -> pd.DataFrame:
    rows = []
    for d in datasets:
        rows.append(
            {
                "dataset": d.name,
                "task": d.task,
                "n_samples": d.X.shape[0],
                "n_features": d.X.shape[1],
                "target_summary": dict(pd.Series(d.y).value_counts().head(10)) if d.task == "classification" else f"mean={np.mean(d.y):.3f}, std={np.std(d.y):.3f}",
                "source": d.source,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Running comparative model experiments...")
    print(f"Python: {platform.python_version()} | NumPy: {np.__version__} | pandas: {pd.__version__}")

    datasets = build_datasets()
    dataset_overview = save_dataset_overview(datasets)
    dataset_overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

    all_results = []
    all_best_params = []

    for ds in datasets:
        print(f"\n=== Dataset: {ds.name} ({ds.task}) ===")
        results_df, params_df = run_experiment(ds)
        all_results.append(results_df)
        all_best_params.append(params_df)

    results = pd.concat(all_results, ignore_index=True)
    best_params = pd.concat(all_best_params, ignore_index=True)
    summary = summarize_results(results)

    results.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)
    best_params.to_csv(OUTPUT_DIR / "best_params.csv", index=False)

    plot_key_metrics(summary)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'dataset_overview.csv'}")
    print(f"- {OUTPUT_DIR / 'raw_results.csv'}")
    print(f"- {OUTPUT_DIR / 'summary_results.csv'}")
    print(f"- {OUTPUT_DIR / 'best_params.csv'}")
    print(f"- Plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
