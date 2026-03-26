"""
Tox21 Model Training (Upgraded)
- Per-target model selection (Random Forest vs XGBoost)
- Validation-based threshold tuning
- Relatable performance plots for reporting
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42


def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def best_mcc_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 33)
    best_thr = 0.5
    best_mcc = -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if np.isnan(mcc):
            continue
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    if best_mcc < -0.99:
        return 0.5, np.nan
    return best_thr, best_mcc


def evaluate_probs(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def build_model_candidates(pos_weight, class_weight):
    rf_candidates = [
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ]

    xgb_candidates = [
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    ]

    return {
        "rf": rf_candidates,
        "xgb": xgb_candidates,
    }


def train_family_best(model_candidates, X_tr, y_tr, X_val, y_val):
    family_best = {}

    for family, candidates in model_candidates.items():
        best = None
        best_score = (-np.inf, -np.inf, -np.inf)

        for model in candidates:
            m = clone(model)
            m.fit(X_tr, y_tr)
            val_prob = m.predict_proba(X_val)[:, 1]
            val_roc = safe_roc_auc(y_val, val_prob)
            val_pr = safe_pr_auc(y_val, val_prob)
            tuned_thr, val_mcc_best = best_mcc_threshold(y_val, val_prob)

            # Prioritize PR-AUC for imbalanced targets, then ROC-AUC, then MCC.
            score = (
                -np.inf if np.isnan(val_pr) else val_pr,
                -np.inf if np.isnan(val_roc) else val_roc,
                -np.inf if np.isnan(val_mcc_best) else val_mcc_best,
            )
            if score > best_score:
                best_score = score
                best = {
                    "model": m,
                    "threshold": tuned_thr,
                    "val_metrics": {
                        "roc_auc": val_roc,
                        "pr_auc": val_pr,
                        "mcc": val_mcc_best,
                    },
                }

        family_best[family] = best

    return family_best


def plot_relatable_graphs(summary_df, best_df):
    sns.set_theme(style="whitegrid")

    # 1) Model comparison by target (ROC-AUC)
    plt.figure(figsize=(14, 5))
    comp = summary_df.pivot(index="target", columns="model", values="roc_auc").reset_index()
    comp = comp.melt(id_vars=["target"], var_name="model", value_name="roc_auc")
    sns.barplot(data=comp, x="target", y="roc_auc", hue="model")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Tox21 Target")
    plt.ylabel("ROC-AUC")
    plt.title("Per-Target ROC-AUC: Random Forest vs XGBoost")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("model_roc_auc_comparison.png", dpi=150)
    plt.close()

    # 2) Best model PR-AUC by target
    plt.figure(figsize=(14, 5))
    sns.barplot(data=best_df, x="target", y="pr_auc", hue="best_model", dodge=False)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Tox21 Target")
    plt.ylabel("PR-AUC")
    plt.title("Best Selected Model Per Target (PR-AUC)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("best_model_pr_auc_by_target.png", dpi=150)
    plt.close()

    # 3) Threshold profile by target
    plt.figure(figsize=(14, 5))
    sns.barplot(data=best_df, x="target", y="threshold", color="#4C78A8")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Tox21 Target")
    plt.ylabel("Optimal Threshold")
    plt.title("Validation-Tuned Decision Thresholds by Target")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("optimal_thresholds_by_target.png", dpi=150)
    plt.close()


print("Loading preprocessed splits ...")
with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

results = {}
summary_rows = []
best_rows = []

print(f"\n{'Target':<18} {'Selected':<14} {'Thr':>6} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>7}")
print("-" * 72)

for target, data in splits["targets"].items():
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Holdout validation split for model/threshold selection.
    val_size = 0.2
    stratify = y_train if len(np.unique(y_train)) == 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    candidates = build_model_candidates(
        pos_weight=data["pos_weight"],
        class_weight=data["class_weight"],
    )

    family_best = train_family_best(candidates, X_tr, y_tr, X_val, y_val)

    # Retrain each family best on full train for fair test comparison.
    family_test = {}
    for family in ["rf", "xgb"]:
        model = family_best[family]["model"]
        threshold = family_best[family]["threshold"]

        retrained = clone(model)
        retrained.fit(X_train, y_train)

        y_prob = retrained.predict_proba(X_test)[:, 1]
        metrics = evaluate_probs(y_test, y_prob, threshold)

        family_test[family] = {
            "model": retrained,
            "threshold": threshold,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "mcc": metrics["mcc"],
        }

        summary_rows.append(
            {
                "target": target,
                "model": "RandomForest" if family == "rf" else "XGBoost",
                "threshold": threshold,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "mcc": metrics["mcc"],
            }
        )

    # Select best family by PR-AUC then ROC-AUC.
    rf_key = (
        -np.inf if np.isnan(family_test["rf"]["pr_auc"]) else family_test["rf"]["pr_auc"],
        -np.inf if np.isnan(family_test["rf"]["roc_auc"]) else family_test["rf"]["roc_auc"],
    )
    xgb_key = (
        -np.inf if np.isnan(family_test["xgb"]["pr_auc"]) else family_test["xgb"]["pr_auc"],
        -np.inf if np.isnan(family_test["xgb"]["roc_auc"]) else family_test["xgb"]["roc_auc"],
    )

    selected_family = "rf" if rf_key >= xgb_key else "xgb"
    selected_name = "RandomForest" if selected_family == "rf" else "XGBoost"

    results[target] = {
        "rf": family_test["rf"],
        "xgb": family_test["xgb"],
        "best_model": selected_family,
        "best_threshold": family_test[selected_family]["threshold"],
    }

    best_rows.append(
        {
            "target": target,
            "best_model": selected_name,
            "threshold": family_test[selected_family]["threshold"],
            "roc_auc": family_test[selected_family]["roc_auc"],
            "pr_auc": family_test[selected_family]["pr_auc"],
            "mcc": family_test[selected_family]["mcc"],
        }
    )

    print(
        f"{target:<18} {selected_name:<14} "
        f"{family_test[selected_family]['threshold']:>6.2f} "
        f"{family_test[selected_family]['roc_auc']:>8.3f} "
        f"{family_test[selected_family]['pr_auc']:>8.3f} "
        f"{family_test[selected_family]['mcc']:>7.3f}"
    )

print("-" * 72)

summary_df = pd.DataFrame(summary_rows)
best_df = pd.DataFrame(best_rows)

print(
    f"\nMean ROC-AUC (best selected models): {best_df['roc_auc'].mean():.3f} | "
    f"Mean PR-AUC (best selected models): {best_df['pr_auc'].mean():.3f}"
)

print("\nSaving models and reports ...")
with open("models.pkl", "wb") as f:
    pickle.dump(results, f)

summary_df.to_csv("model_performance_all_models.csv", index=False)
best_df.to_csv("model_performance_best_models.csv", index=False)

plot_relatable_graphs(summary_df, best_df)

print("Saved: models.pkl")
print("Saved: model_performance_all_models.csv")
print("Saved: model_performance_best_models.csv")
print("Saved: model_roc_auc_comparison.png")
print("Saved: best_model_pr_auc_by_target.png")
print("Saved: optimal_thresholds_by_target.png")

