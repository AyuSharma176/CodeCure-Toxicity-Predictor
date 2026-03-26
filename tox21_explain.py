"""
Tox21 SHAP Explainability + Feature Importance
Generates SHAP plots and saves molecule visualizations
"""

import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading models and data ...")
with open("models.pkl", "rb") as f:
    results = pickle.load(f)
with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

feat_names = list(np.load("feature_names.npy", allow_pickle=True))

# ── CONFIG: which targets to explain ─────────────────────────────────────────

# We explain the 3 most important targets (best ROC-AUC + most biologically relevant)
EXPLAIN_TARGETS = ["NR-AR-LBD", "SR-MMP", "NR-AhR"]

# ── SHAP ANALYSIS ─────────────────────────────────────────────────────────────

print("\nRunning SHAP analysis (this takes ~2 minutes) ...")

for target in EXPLAIN_TARGETS:
    print(f"\n  Explaining {target} ...")

    model_key = results[target].get("best_model", "rf")
    model = results[target][model_key]["model"]
    X_test   = splits["targets"][target]["X_test"]
    y_test   = splits["targets"][target]["y_test"]

    # Use a background sample for speed (200 samples is enough)
    bg_size  = min(200, len(X_test))
    bg_idx   = np.random.choice(len(X_test), bg_size, replace=False)
    X_bg     = X_test[bg_idx]

    # SHAP TreeExplainer — fast for tree-based models
    explainer   = shap.TreeExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # shap_values is a list [class0, class1] for classifiers — we want class 1 (toxic)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── PLOT 1: Summary bar plot (top 20 features) ────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test,
        feature_names=feat_names,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title(f"{target} — Top 20 Features by Mean |SHAP|", fontsize=13, pad=12)
    plt.tight_layout()
    fname = f"shap_bar_{target.replace('-', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {fname}")

    # ── PLOT 2: Beeswarm plot (shows direction of effect) ─────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test,
        feature_names=feat_names,
        max_display=20,
        show=False
    )
    plt.title(f"{target} — SHAP Beeswarm (red = high feature value → toxic)", fontsize=11, pad=12)
    plt.tight_layout()
    fname2 = f"shap_beeswarm_{target.replace('-', '_')}.png"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {fname2}")

    # ── PLOT 3: Waterfall for single toxic molecule ───────────────────────────
    toxic_idx = np.where(y_test == 1)[0]
    if len(toxic_idx) > 0:
        sample_idx = toxic_idx[0]
        # Extract class-1 (toxic) SHAP values for this single sample
        raw_vals = sv[sample_idx]
        if raw_vals.ndim == 2:
            raw_vals = raw_vals[:, 1]   # take toxic class column

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1]

        shap_exp = shap.Explanation(
            values        = raw_vals,
            base_values   = float(base),
            data          = X_test[sample_idx],
            feature_names = feat_names,
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title(f"{target} — Why this molecule is predicted TOXIC", fontsize=11, pad=12)
        plt.tight_layout()
        fname3 = f"shap_waterfall_{target.replace('-', '_')}.png"
        plt.savefig(fname3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {fname3}")

# ── GLOBAL FEATURE IMPORTANCE ACROSS ALL TARGETS ──────────────────────────────

print("\nComputing global feature importance across all 12 targets ...")

importance_sum = np.zeros(len(feat_names))

for target, data in results.items():
    model  = data["rf"]["model"]
    imp    = model.feature_importances_
    importance_sum += imp

importance_mean = importance_sum / len(results)

# Top 25 features overall
top_idx   = np.argsort(importance_mean)[::-1][:25]
top_names = [feat_names[i] for i in top_idx]
top_vals  = importance_mean[top_idx]

plt.figure(figsize=(10, 8))
colors = ["#e05c3a" if not n.startswith("morgan") else "#5b8dd9" for n in top_names]
bars = plt.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
plt.yticks(range(len(top_names)), top_names[::-1], fontsize=10)
plt.xlabel("Mean Feature Importance (across all 12 targets)", fontsize=11)
plt.title("Global Feature Importance — Top 25\n(orange = physicochemical descriptor, blue = Morgan fingerprint bit)",
          fontsize=11)
plt.tight_layout()
plt.savefig("global_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: global_feature_importance.png")

# ── PRINT TOP PHYSICOCHEMICAL DESCRIPTORS ─────────────────────────────────────

print("\n── Top physicochemical descriptors driving toxicity ──")
physchem_names = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
]
print(f"  {'Descriptor':<22} {'Importance':>12}")
print("  " + "-" * 36)
for name in physchem_names:
    if name in feat_names:
        idx = feat_names.index(name)
        print(f"  {name:<22} {importance_mean[idx]:>12.6f}")

print("\n✓ All SHAP plots saved!")
