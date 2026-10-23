#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Baseline HAI Prediction - Paper methodology, 3 seeds × 5-fold CV
=============================================================================
 - Predictors : admission / first-24h variables only
 - Excluded   : Ward, Patient complaint, Diagnosis (leaky / free-text / admin)
 - Sampling   : random undersampling  (1,660 pos + 1,660 neg = 3,320 rows, 1:1)
 - Models     : LR, RF, AdaBoost, XGBoost
 - Metrics    : Accuracy, Balanced Acc, Sensitivity, Specificity,
                Precision, F1, AUROC, AUPRC, MCC
 - Outputs    : CSV results, ROC plot, XGBoost feature importance (CSV + PNG)
=============================================================================
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
CSV_PATH  = BASE_DIR / "Hospital_infection_data.csv"
OUT_DIR   = BASE_DIR

SEEDS     = [42, 2024, 7]
N_SPLITS  = 5
LABEL_COL = "Label"

# Columns that must NOT enter the baseline model
DROP_COLS = [
    "#",                  # row index
    "Ward",               # administrative
    "Diagnosis",          # 1825-level ICD-10 (leaky & high-cardinality)
    "Patient complaint",  # free-text (reserved for NLP)
]

MODEL_NAMES = ["Logistic Regression", "Random Forest", "AdaBoost", "XGBoost"]
METRIC_KEYS = ["acc", "bacc", "sens", "spec",
               "prec", "f1", "auroc", "auprc", "mcc"]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (df.columns.astype(str)
                            .str.replace("\ufeff", "", regex=False)
                            .str.strip())
    print(f"[data] raw shape : {df.shape}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=present)
    print(f"[data] dropped columns : {present}")
    print(f"[data] remaining shape : {df.shape}")
    return df


# --------------------------------------------------------------------------
# Random undersampling (1:1)
# --------------------------------------------------------------------------
def random_undersample(df: pd.DataFrame, seed: int,
                       label_col: str = LABEL_COL) -> pd.DataFrame:
    """Randomly sample n_positive negatives to match the positive class (1:1)."""
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    n_pos = len(pos)
    n_neg_available = len(neg)

    if n_neg_available < n_pos:
        raise ValueError(
            f"Not enough negatives ({n_neg_available}) to match positives ({n_pos})"
        )

    neg_sel = neg.sample(n=n_pos, random_state=seed, replace=False)

    balanced = (
        pd.concat([pos, neg_sel], ignore_index=True)
          .sample(frac=1.0, random_state=seed)
          .reset_index(drop=True)
    )
    print(f"[sampling] seed={seed} | rows={len(balanced)} "
          f"({n_pos} pos + {n_pos} neg) | random undersampling")
    return balanced


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame,
                       impute_numeric: bool = True,
                       scale_numeric: bool = False) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_steps = []
    if impute_numeric:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols),
                      ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def get_preprocessor_for(name: str, X: pd.DataFrame) -> ColumnTransformer:
    if name == "Logistic Regression":
        return build_preprocessor(X, impute_numeric=True, scale_numeric=True)
    if name == "XGBoost":
        return build_preprocessor(X, impute_numeric=False, scale_numeric=False)
    return build_preprocessor(X, impute_numeric=True, scale_numeric=False)


# --------------------------------------------------------------------------
# Model factory — exactly as in paper
# --------------------------------------------------------------------------
def get_model(name: str, seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(penalty="l2", C=1.0,
                                  class_weight="balanced",
                                  solver="liblinear", max_iter=3000,
                                  random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=10,
                                      class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50, learning_rate=0.05, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             eval_metric="logloss",
                             tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(name)


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def evaluate_model(name: str, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    metrics = {k: [] for k in METRIC_KEYS}
    roc_curves = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        prep = get_preprocessor_for(name, X_tr)
        clf  = get_model(name, seed)

        if name == "XGBoost":
            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            clf.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_tr, y_tr)
        else:
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_tr, y_tr)

        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        metrics["acc"].append(accuracy_score(y_te, y_pred))
        metrics["bacc"].append(balanced_accuracy_score(y_te, y_pred))
        metrics["sens"].append(sens)
        metrics["spec"].append(spec)
        metrics["prec"].append(precision_score(y_te, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        metrics["auroc"].append(roc_auc_score(y_te, y_prob))
        metrics["auprc"].append(average_precision_score(y_te, y_prob))
        metrics["mcc"].append(matthews_corrcoef(y_te, y_pred))

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_curves.append((fpr, tpr))

    out = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
    out["_roc_curves"] = roc_curves
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_seed_results(name: str, seed: int, res: dict) -> None:
    print(f"\n--- {name} (seed={seed}) ---")
    for k in METRIC_KEYS:
        m, s = res[k]
        print(f"  {k:>6s}: {m:.4f} ± {s:.4f}")


def print_final_table(results: dict) -> None:
    headers = ["Accuracy", "BalancedAcc", "Sensitivity", "Specificity",
               "Precision", "F1", "AUROC", "AUPRC", "MCC"]

    line = "=" * 185
    print("\n" + line)
    print("FINAL RESULTS — random undersampling (3 seeds × 5 folds)")
    print(line)
    print(f"{'Model':<22}" + "".join(f"{h:<17}" for h in headers))
    print("-" * 185)
    for name in MODEL_NAMES:
        row = f"{name:<22}"
        for c in METRIC_KEYS:
            mean = float(np.mean([r[c][0] for r in results[name]]))
            std  = float(np.std ([r[c][0] for r in results[name]]))
            row += f"{mean:.3f}±{std:.3f}   "
        print(row)
    print(line + "\n")


def save_results_csv(results: dict, out_path: Path) -> None:
    rows = []
    for name in MODEL_NAMES:
        row = {"Model": name}
        for k in METRIC_KEYS:
            row[k]          = float(np.mean([x[k][0] for x in results[name]]))
            row[k + "_std"] = float(np.std ([x[k][0] for x in results[name]]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[save] results → {out_path}")


def plot_roc_curves(results: dict, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name in MODEL_NAMES:
        all_fpr = np.unique(np.concatenate(
            [c[0] for r in results[name] for c in r["_roc_curves"]]))
        mean_tpr = np.zeros_like(all_fpr)
        n = 0
        for r in results[name]:
            for fpr, tpr in r["_roc_curves"]:
                mean_tpr += np.interp(all_fpr, fpr, tpr)
                n += 1
        mean_tpr /= max(n, 1)
        auc = float(np.mean([x["auroc"][0] for x in results[name]]))
        plt.plot(all_fpr, mean_tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (random undersampling)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] ROC plot → {out_path}")


# --------------------------------------------------------------------------
# Feature importance (XGBoost, weight-based)
# --------------------------------------------------------------------------
def xgboost_feature_importance(X: pd.DataFrame, y: pd.Series,
                               seed: int, top_k: int = 15) -> pd.DataFrame:
    prep = build_preprocessor(X, impute_numeric=False, scale_numeric=False)
    Xt = prep.fit_transform(X)
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1,
        importance_type="weight",
    )
    model.fit(Xt, y)

    imp = pd.DataFrame({
        "feature": names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\nTop feature importances (XGBoost):")
    for i, row in imp.head(top_k).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<35s} {row['importance']:.5f}")

    head = imp.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * top_k)))
    plt.barh(head["feature"], head["importance"], color="#1f77b4")
    plt.xlabel("Importance (weight)")
    plt.title(f"Top {top_k} Feature Importance — XGBoost")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_xgb_importance.png", dpi=150)
    plt.close()
    print(f"[save] feature-importance plot → "
          f"{OUT_DIR / 'baseline_xgb_importance.png'}")

    return imp


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()

    df = load_dataset(CSV_PATH)
    df = drop_unused_columns(df)

    if LABEL_COL not in df.columns:
        raise SystemExit(f"Label column '{LABEL_COL}' not found.")
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    results = {name: [] for name in MODEL_NAMES}
    last_X = last_y = None

    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====")
        balanced = random_undersample(df, seed=seed)
        X = balanced.drop(columns=[LABEL_COL])
        y = balanced[LABEL_COL]
        last_X, last_y = X, y

        n_num = X.select_dtypes(include=[np.number]).shape[1]
        n_cat = X.select_dtypes(include=["object"]).shape[1]
        print(f"[feat] numeric={n_num} categorical={n_cat} "
              f"total={n_num + n_cat}")
        print(f"[feat] categoricals: "
              f"{X.select_dtypes(include=['object']).columns.tolist()}")

        for name in MODEL_NAMES:
            res = evaluate_model(name, X, y, seed)
            print_seed_results(name, seed, res)
            results[name].append(res)

    print_final_table(results)
    save_results_csv(results, OUT_DIR / "baseline_results_random_under.csv")
    plot_roc_curves(results, OUT_DIR / "baseline_roc_random_under.png")

    imp_df = xgboost_feature_importance(last_X, last_y,
                                        seed=SEEDS[0], top_k=15)
    imp_df.to_csv(OUT_DIR / "baseline_xgb_feature_importance.csv", index=False)
    print(f"[save] feature importance CSV → "
          f"{OUT_DIR / 'baseline_xgb_feature_importance.csv'}")

    print(f"\n✔ All artifacts saved under: {OUT_DIR}")
    print(f"✔ Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
