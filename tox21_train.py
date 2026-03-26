"""
Tox21 Model Training
XGBoost + Random Forest across all 12 targets
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── LOAD PREPROCESSED DATA ────────────────────────────────────────────────────

print("Loading preprocessed splits ...")
with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

# ── TRAIN & EVALUATE ──────────────────────────────────────────────────────────

results = {}

print(f"\n{'Target':<18} {'Model':<15} {'ROC-AUC':>8} {'PR-AUC':>8} {'MCC':>7}")
print("-" * 62)

for target, data in splits["targets"].items():
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    pw      = data["pos_weight"]

    results[target] = {}

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pw,      # handles class imbalance
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    results[target]["xgb"] = {"roc_auc": roc, "pr_auc": pr, "mcc": mcc, "model": xgb}
    print(f"{target:<18} {'XGBoost':<15} {roc:>8.3f} {pr:>8.3f} {mcc:>7.3f}")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=data["class_weight"],   # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)

    roc_rf = roc_auc_score(y_test, y_prob_rf)
    pr_rf  = average_precision_score(y_test, y_prob_rf)
    mcc_rf = matthews_corrcoef(y_test, y_pred_rf)
    results[target]["rf"] = {"roc_auc": roc_rf, "pr_auc": pr_rf, "mcc": mcc_rf, "model": rf}
    print(f"{target:<18} {'RandomForest':<15} {roc_rf:>8.3f} {pr_rf:>8.3f} {mcc_rf:>7.3f}")

print("-" * 62)

# ── SUMMARY ───────────────────────────────────────────────────────────────────

xgb_aucs = [results[t]["xgb"]["roc_auc"] for t in results]
rf_aucs  = [results[t]["rf"]["roc_auc"]  for t in results]
print(f"\nMean ROC-AUC  →  XGBoost: {np.mean(xgb_aucs):.3f}  |  RandomForest: {np.mean(rf_aucs):.3f}")

# ── SAVE MODELS ───────────────────────────────────────────────────────────────

print("\nSaving models ...")
with open("models.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved: models.pkl")
print("Next step: python tox21_explain.py")