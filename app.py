"""
CodeCure - Drug Toxicity Prediction App
Streamlit UI: Input SMILES → Predict across 12 Tox21 targets
"""

import streamlit as st
import numpy as np
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CodeCure — Drug Toxicity Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 CodeCure — Drug Toxicity Predictor")
st.markdown("**Predict toxicity of chemical compounds across 12 Tox21 targets using ML + SHAP explainability**")
st.markdown("---")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        results = pickle.load(f)
    return results

@st.cache_resource
def load_feature_names():
    return list(np.load("feature_names.npy", allow_pickle=True))

results     = load_models()
feat_names  = load_feature_names()

PHYSCHEM_DESCRIPTORS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
]

TOX21_TARGETS = list(results.keys())

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Morgan fingerprint
    gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp  = gen.GetFingerprintAsNumPy(mol).reshape(1, -1).astype(np.float32)

    # Physicochemical descriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCRIPTORS)
    desc = np.array(list(calc.CalcDescriptors(mol)), dtype=np.float32).reshape(1, -1)

    # Impute + scale (use simple approach for single molecule)
    desc = np.nan_to_num(desc, nan=0.0)

    X = np.hstack([fp, desc])
    return X, mol

def mol_to_image(mol, size=(400, 300)):
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

st.sidebar.header("Input Compound")
st.sidebar.markdown("Enter a SMILES string or pick an example:")

examples = {
    "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
    "Bisphenol A":   "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
    "Caffeine":      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Tamoxifen":     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Atrazine":      "CCNc1nc(Cl)nc(NC(C)C)n1",
    "Ethanol":       "CCO",
}

selected = st.sidebar.selectbox("Examples", ["Custom"] + list(examples.keys()))
if selected != "Custom":
    default_smiles = examples[selected]
else:
    default_smiles = ""

smiles_input = st.sidebar.text_area(
    "SMILES string",
    value=default_smiles,
    height=100,
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
)

model_choice = st.sidebar.radio(
    "Model",
    ["Random Forest", "XGBoost"],
    index=0
)

threshold = st.sidebar.slider(
    "Toxicity threshold",
    min_value=0.1, max_value=0.9,
    value=0.5, step=0.05,
    help="Probability above this = predicted TOXIC"
)

predict_btn = st.sidebar.button("🔬 Predict Toxicity", type="primary", use_container_width=True)

# ── MAIN PANEL ────────────────────────────────────────────────────────────────

if not predict_btn:
    st.info("👈 Enter a SMILES string in the sidebar and click **Predict Toxicity** to begin.")

    st.subheader("📊 Model Performance Summary")
    perf_data = []
    for target in TOX21_TARGETS:
        perf_data.append({
            "Target":        target,
            "RF ROC-AUC":    f"{results[target]['rf']['roc_auc']:.3f}",
            "XGB ROC-AUC":   f"{results[target]['xgb']['roc_auc']:.3f}",
            "RF PR-AUC":     f"{results[target]['rf']['pr_auc']:.3f}",
        })
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.subheader("🔬 Key Biological Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean RF ROC-AUC", "0.853", "across 12 targets")
    with col2:
        st.metric("Best Target", "NR-AR-LBD", "ROC-AUC 0.890")
    with col3:
        st.metric("Top Toxicity Driver", "MolWt + MolLogP", "global importance")

    st.markdown("""
    **What drives toxicity? (from SHAP analysis)**
    - **Molecular weight & lipophilicity (MolLogP)** are the strongest global predictors — heavier, more fat-soluble molecules are more toxic
    - **Ring count & aromatic rings** strongly predict androgen receptor (NR-AR-LBD) binding — multi-ring steroid-like structures are high risk
    - **FractionCSP3** (ratio of sp3 carbons) is protective — more 3D, saturated molecules tend to be less toxic
    - **TPSA** (polar surface area) influences membrane permeability and toxicity uptake
    """)

else:
    if not smiles_input.strip():
        st.error("Please enter a SMILES string.")
        st.stop()

    X, mol = smiles_to_features(smiles_input.strip())

    if mol is None:
        st.error(f"❌ Invalid SMILES: `{smiles_input}`. Please check your input.")
        st.stop()

    # ── RESULTS ───────────────────────────────────────────────────────────────

    model_key = "rf" if model_choice == "Random Forest" else "xgb"

    predictions = {}
    for target in TOX21_TARGETS:
        model  = results[target][model_key]["model"]
        prob   = model.predict_proba(X)[0, 1]
        label  = "🔴 TOXIC" if prob >= threshold else "🟢 Safe"
        predictions[target] = {"probability": prob, "label": label}

    n_toxic = sum(1 for v in predictions.values() if v["probability"] >= threshold)

    # ── LAYOUT ────────────────────────────────────────────────────────────────

    col_mol, col_summary = st.columns([1, 2])

    with col_mol:
        st.subheader("Molecule Structure")
        img_buf = mol_to_image(mol)
        st.image(img_buf, caption=smiles_input[:60], use_container_width=True)

        # Physicochemical properties
        calc  = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCRIPTORS)
        props = dict(zip(PHYSCHEM_DESCRIPTORS, calc.CalcDescriptors(mol)))
        st.subheader("Physicochemical Properties")
        prop_df = pd.DataFrame([
            {"Property": k, "Value": f"{v:.3f}"}
            for k, v in props.items()
        ])
        st.dataframe(prop_df, use_container_width=True, hide_index=True)

    with col_summary:
        st.subheader("Toxicity Prediction Results")

        # Overall verdict
        if n_toxic == 0:
            st.success(f"✅ **LOW TOXICITY RISK** — Toxic in 0 / {len(TOX21_TARGETS)} assays")
        elif n_toxic <= 3:
            st.warning(f"⚠️ **MODERATE RISK** — Toxic in {n_toxic} / {len(TOX21_TARGETS)} assays")
        else:
            st.error(f"🚨 **HIGH TOXICITY RISK** — Toxic in {n_toxic} / {len(TOX21_TARGETS)} assays")

        # Results table
        pred_df = pd.DataFrame([
            {
                "Target":      t,
                "Prediction":  predictions[t]["label"],
                "Probability": f"{predictions[t]['probability']:.3f}",
                "Risk":        "HIGH" if predictions[t]["probability"] >= threshold else "low"
            }
            for t in TOX21_TARGETS
        ])
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # ── PROBABILITY BAR CHART ─────────────────────────────────────────────────

    st.subheader("Toxicity Probability by Target")
    probs  = [predictions[t]["probability"] for t in TOX21_TARGETS]
    colors = ["#e05c3a" if p >= threshold else "#4caf7d" for p in probs]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(TOX21_TARGETS, probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.set_ylabel("Predicted Toxicity Probability", fontsize=11)
    ax.set_xlabel("Tox21 Target", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(TOX21_TARGETS, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── SHAP IMAGES ───────────────────────────────────────────────────────────

    st.subheader("SHAP Explainability — Feature Importance")
    shap_cols = st.columns(3)
    shap_files = [
        ("shap_bar_NR_AR_LBD.png",   "NR-AR-LBD Feature Importance"),
        ("shap_waterfall_NR_AR_LBD.png", "NR-AR-LBD Waterfall"),
        ("shap_waterfall_NR_AhR.png",    "NR-AhR Waterfall"),
    ]
    for col, (fname, caption) in zip(shap_cols, shap_files):
        try:
            col.image(fname, caption=caption, use_container_width=True)
        except Exception:
            col.info(f"{fname} not found")

    st.image("global_feature_importance.png",
             caption="Global Feature Importance across all 12 targets",
             use_container_width=True)