import io
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ------------------------------- PAGE CONFIG ---------------------------------

st.set_page_config(
    page_title="CodeCure - Drug Toxicity Predictor",
    page_icon="🧬",
    layout="wide",
)


def inject_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --primary: #0b7285;
            --primary-2: #0f766e;
            --sun: #f59e0b;
            --accent: #dc2626;
            --safe: #15803d;
            --warn: #b45309;
            --bg-card: #f8fbff;
            --bg-card-2: #eef7ff;
            --text-main: #0f172a;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 8%, #d7f5ff 0%, transparent 25%),
                radial-gradient(circle at 92% 6%, #fff3d6 0%, transparent 20%),
                radial-gradient(circle at 85% 96%, #d6ffe7 0%, transparent 20%),
                linear-gradient(180deg, #f6f9ff 0%, #f7f8fc 100%);
            color: var(--text-main);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: 0.2px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f3f7ff 0%, #eef5ff 100%);
            border-right: 1px solid #d9e4f5;
            padding-top: 0.4rem;
        }

        /* Keep sidebar text readable regardless of Streamlit base theme */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] div {
            color: #0f172a !important;
        }

        [data-testid="stSidebar"] .stCaption {
            color: #475569 !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSidebar"] .stCheckbox > label {
            background: #ffffff !important;
            border-color: #c7d7ee !important;
            color: #0f172a !important;
            border-radius: 12px !important;
        }

        [data-testid="stSidebar"] .stRadio > div,
        [data-testid="stSidebar"] .stSlider {
            padding: 0.45rem 0.5rem;
            border: 1px solid #d9e4f5;
        }

        [data-testid="stSidebar"] .stTextArea textarea {
            min-height: 90px !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(90deg, #ef4444, #fb7185) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            box-shadow: 0 8px 16px rgba(239, 68, 68, 0.22);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(239, 68, 68, 0.28);
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 2px dashed #93c5fd !important;
            border-radius: 14px !important;
            padding: 0.7rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #1e3a8a !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(90deg, #ef4444, #fb7185) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            box-shadow: 0 8px 16px rgba(239, 68, 68, 0.22) !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
            background: linear-gradient(90deg, #ef4444, #fb7185) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(239, 68, 68, 0.28) !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            padding: 1.35rem 1.3rem;
            background: linear-gradient(115deg, #0b7285 0%, #0f766e 52%, #166534 100%);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 14px 34px rgba(15, 118, 110, 0.24);
            margin-bottom: 0.9rem;
            animation: fadeInUp 0.55s ease;
        }

        .hero::before {
            content: "";
            position: absolute;
            top: -80px;
            right: -80px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0.0) 65%);
            pointer-events: none;
        }

        .hero::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 6px;
            background: linear-gradient(90deg, #7dd3fc, #fde68a, #86efac);
            opacity: 0.9;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.95rem;
            letter-spacing: 0.35px;
        }

        .hero p {
            margin: 0.4rem 0 0 0;
            opacity: 0.95;
            font-size: 1rem;
        }

        .section-card {
            background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-card-2) 100%);
            border: 1px solid #d9e6f7;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            animation: fadeInUp 0.55s ease;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }

        .chip {
            background: #ffffff;
            border: 1px solid #dbe7f8;
            color: #0f172a;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .kpi {
            border-radius: 14px;
            padding: 0.75rem;
            border: 1px solid #d7e3f7;
            background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
        }

        .kpi .label {
            font-size: 0.78rem;
            color: #475569;
            margin-bottom: 0.15rem;
        }

        .kpi .value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f172a;
        }

        .risk-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.78rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        .risk-low { background: #dcfce7; color: #166534; }
        .risk-mid { background: #fef3c7; color: #92400e; }
        .risk-high { background: #fee2e2; color: #991b1b; }

        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #d9e4f5;
            background: #ffffff;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid #bfdbfe;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            border: 1px solid #dbe4f2;
            background: #f8fbff;
            padding: 8px 14px;
            font-weight: 600;
            color: #1e293b !important;
        }

        .stTabs [aria-selected="true"] {
            background: #e0f2fe !important;
            border-color: #7dd3fc !important;
            color: #0f172a !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: transparent !important;
        }

        @media (max-width: 900px) {
            .kpi-row {
                grid-template-columns: 1fr;
            }
            .hero h1 {
                font-size: 1.5rem;
            }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_styles()

st.markdown(
    """
    <div class="hero">
      <h1>CodeCure Toxicity Intelligence</h1>
      <p>Predict toxicological risk across 12 Tox21 targets with tuned thresholds, model selection, and explainability visuals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- LOAD ASSETS ---------------------------------

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_preprocess_artifacts():
    with open("preprocess_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_names():
    return list(np.load("feature_names.npy", allow_pickle=True))


results = load_models()
prep_artifacts = load_preprocess_artifacts()
feat_names = load_feature_names()

PHYSCHEM_DESCRIPTORS = prep_artifacts["physchem_descriptors"]
TOX21_TARGETS = list(results.keys())


# ------------------------------- HELPERS -------------------------------------

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    gen = GetMorganGenerator(
        radius=prep_artifacts["morgan_radius"],
        fpSize=prep_artifacts["morgan_n_bits"],
    )
    fp = gen.GetFingerprintAsNumPy(mol).reshape(1, -1).astype(np.float32)

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCRIPTORS)
    desc = np.array(list(calc.CalcDescriptors(mol)), dtype=np.float32).reshape(1, -1)
    desc = prep_artifacts["imputer"].transform(desc)
    desc = prep_artifacts["scaler"].transform(desc)

    X = np.hstack([fp, desc])
    return X, mol


def mol_to_image(mol, size=(420, 300)):
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def prediction_dataframe(predictions):
    rows = []
    for target, p in predictions.items():
        risk = "HIGH" if p["probability"] >= p["threshold"] else "LOW"
        rows.append(
            {
                "Target": target,
                "Model": "RF" if p["model_used"] == "rf" else "XGB",
                "Threshold": round(float(p["threshold"]), 2),
                "Probability": round(float(p["probability"]), 3),
                "Prediction": "TOXIC" if risk == "HIGH" else "SAFE",
                "Risk": risk,
            }
        )
    df = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    return df


def risk_badge(n_toxic, total_targets):
    if n_toxic == 0:
        return "LOW", "risk-low", f"Low Risk: {n_toxic}/{total_targets} assays flagged"
    if n_toxic <= 3:
        return "MODERATE", "risk-mid", f"Moderate Risk: {n_toxic}/{total_targets} assays flagged"
    return "HIGH", "risk-high", f"High Risk: {n_toxic}/{total_targets} assays flagged"


def performance_summary_rows():
    rows = []
    for t in TOX21_TARGETS:
        rows.append(
            {
                "Target": t,
                "RF ROC-AUC": round(float(results[t]["rf"]["roc_auc"]), 3),
                "XGB ROC-AUC": round(float(results[t]["xgb"]["roc_auc"]), 3),
                "RF PR-AUC": round(float(results[t]["rf"]["pr_auc"]), 3),
                "XGB PR-AUC": round(float(results[t]["xgb"]["pr_auc"]), 3),
                "Best": "RF" if results[t].get("best_model", "rf") == "rf" else "XGB",
                "Tuned Threshold": round(float(results[t].get("best_threshold", 0.5)), 2),
            }
        )
    return pd.DataFrame(rows)


def get_target_policy(target, model_choice, use_tuned_threshold, manual_threshold):
    if model_choice == "Best per Target":
        model_key = results[target].get("best_model", "rf")
    else:
        model_key = "rf" if model_choice == "Random Forest" else "xgb"

    threshold = float(results[target].get("best_threshold", 0.5)) if use_tuned_threshold else float(manual_threshold)
    return model_key, threshold


def predict_targets_for_feature_vector(X, model_choice, use_tuned_threshold, manual_threshold):
    predictions = {}
    for target in TOX21_TARGETS:
        model_key, threshold = get_target_policy(target, model_choice, use_tuned_threshold, manual_threshold)
        model = results[target][model_key]["model"]
        prob = float(model.predict_proba(X)[0, 1])
        predictions[target] = {
            "probability": prob,
            "threshold": threshold,
            "model_used": model_key,
        }
    return predictions


# ------------------------------- SIDEBAR -------------------------------------

st.sidebar.header("Screening")
screening_mode = st.sidebar.radio(
    "Mode",
    ["Single Compound", "Batch Screening"],
    index=0,
)

model_choice = st.sidebar.radio(
    "Model policy",
    ["Best per Target", "Random Forest", "XGBoost"],
    index=0,
)

use_tuned_threshold = st.sidebar.checkbox(
    "Use tuned threshold per target",
    value=True,
)

manual_threshold = st.sidebar.slider(
    "Manual threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

if screening_mode == "Single Compound":
    st.sidebar.header("Compound Input")
    st.sidebar.caption("Choose an example or paste your own SMILES")

    examples = {
        "Custom": "",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Bisphenol A": "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
        "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Tamoxifen": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "Atrazine": "CCNc1nc(Cl)nc(NC(C)C)n1",
        "Ethanol": "CCO",
    }

    selected = st.sidebar.selectbox("Example molecules", list(examples.keys()), index=0)
    smiles_input = st.sidebar.text_area(
        "SMILES",
        value=examples[selected],
        height=95,
        placeholder="Example: CC(=O)Oc1ccccc1C(=O)O",
    )
    run_btn = st.sidebar.button("Predict Toxicity", type="primary", use_container_width=True)
else:
    st.sidebar.header("Batch Input")
    st.sidebar.caption("Upload a CSV file for multi-compound screening")
    uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
    batch_limit = st.sidebar.slider("Max rows to process", 10, 3000, 500, 10)
    run_btn = st.sidebar.button("Run Batch Screening", type="primary", use_container_width=True)


# ------------------------------- LANDING VIEW --------------------------------

if screening_mode == "Single Compound" and not run_btn:
    perf_df = performance_summary_rows()
    best_mean_roc = perf_df[["RF ROC-AUC", "XGB ROC-AUC"]].max(axis=1).mean()
    best_target_idx = perf_df[["RF ROC-AUC", "XGB ROC-AUC"]].max(axis=1).idxmax()
    best_target = perf_df.loc[best_target_idx, "Target"]
    best_target_score = perf_df[["RF ROC-AUC", "XGB ROC-AUC"]].max(axis=1).max()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Model Snapshot")
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi"><div class="label">Targets</div><div class="value">{len(TOX21_TARGETS)}</div></div>
          <div class="kpi"><div class="label">Mean Best ROC-AUC</div><div class="value">{best_mean_roc:.3f}</div></div>
          <div class="kpi"><div class="label">Best Target</div><div class="value">{best_target} ({best_target_score:.3f})</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="chip-row">
          <span class="chip">Multi-Target Toxicity</span>
          <span class="chip">Threshold-Tuned Decisions</span>
          <span class="chip">SHAP Explainability</span>
          <span class="chip">Chemistry-Aware Features</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Overview", "Performance", "Insights"])

    with tab1:
        st.info("Use the sidebar to run toxicity prediction for any SMILES string.")
        st.markdown(
            """
            This dashboard combines:
            - per-target model selection
            - tuned decision thresholds
            - SHAP-based interpretability assets
            """
        )

    with tab2:
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown(
            """
            - Higher MolWt and MolLogP are often linked to increased toxicity probability.
            - Aromatic/ring-heavy scaffolds are more likely to trigger receptor-based toxicity endpoints.
            - FractionCSP3 is frequently associated with lower toxicity risk.
            - Tuned thresholds reduce false calls compared to fixed 0.50 cutoffs.
            """
        )

elif screening_mode == "Batch Screening" and not run_btn:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Batch Screening Mode")
    st.info("Upload a CSV from the sidebar, then click **Run Batch Screening**.")
    st.markdown(
        """
        Expected CSV format:
        - Must include one SMILES column (for example: `smiles`, `SMILES`, `canonical_smiles`, `smi`)
        - Additional metadata columns are allowed and will be preserved in your source file
        """
    )
    st.dataframe(
        pd.DataFrame(
            {
                "smiles": [
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
                ],
                "compound_id": ["cmpd_001", "cmpd_002"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif screening_mode == "Batch Screening" and run_btn:
    if uploaded_file is None:
        st.error("Please upload a CSV file before running batch screening.")
        st.stop()

    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    if batch_df.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    candidate_cols = [
        c for c in batch_df.columns
        if str(c).strip().lower() in {"smiles", "canonical_smiles", "smi"}
    ]
    default_col = candidate_cols[0] if candidate_cols else batch_df.columns[0]
    smiles_col = st.sidebar.selectbox("SMILES column", options=list(batch_df.columns), index=list(batch_df.columns).index(default_col))

    data = batch_df.head(batch_limit).copy()
    data[smiles_col] = data[smiles_col].astype(str).str.strip()

    out_rows = []
    prog = st.progress(0)
    status = st.empty()

    for idx, row in data.reset_index(drop=True).iterrows():
        smi = row[smiles_col]
        X, mol = smiles_to_features(smi)
        if X is None:
            out_rows.append(
                {
                    "row_id": idx,
                    "smiles": smi,
                    "status": "invalid_smiles",
                    "overall_risk": "INVALID",
                    "flagged_assays": np.nan,
                    "mean_probability": np.nan,
                    "top_endpoint": "",
                    "top_endpoint_probability": np.nan,
                }
            )
        else:
            pred = predict_targets_for_feature_vector(X, model_choice, use_tuned_threshold, manual_threshold)
            probs = {t: pred[t]["probability"] for t in TOX21_TARGETS}
            flags = {t: int(pred[t]["probability"] >= pred[t]["threshold"]) for t in TOX21_TARGETS}

            flagged_assays = int(sum(flags.values()))
            mean_probability = float(np.mean(list(probs.values())))
            top_endpoint = max(probs, key=probs.get)
            top_endpoint_probability = float(probs[top_endpoint])

            if flagged_assays == 0:
                overall_risk = "LOW"
            elif flagged_assays <= 3:
                overall_risk = "MODERATE"
            else:
                overall_risk = "HIGH"

            result_row = {
                "row_id": idx,
                "smiles": smi,
                "status": "ok",
                "overall_risk": overall_risk,
                "flagged_assays": flagged_assays,
                "mean_probability": round(mean_probability, 4),
                "top_endpoint": top_endpoint,
                "top_endpoint_probability": round(top_endpoint_probability, 4),
            }
            for t in TOX21_TARGETS:
                result_row[f"prob_{t}"] = round(float(probs[t]), 4)
            out_rows.append(result_row)

        prog.progress((idx + 1) / len(data))
        status.text(f"Processed {idx + 1}/{len(data)} rows")

    result_df = pd.DataFrame(out_rows)
    valid_df = result_df[result_df["status"] == "ok"].copy()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Batch Screening Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows processed", f"{len(result_df)}")
    c2.metric("Valid SMILES", f"{len(valid_df)}")
    c3.metric("Invalid SMILES", f"{len(result_df) - len(valid_df)}")
    high_count = int((valid_df["overall_risk"] == "HIGH").sum()) if len(valid_df) else 0
    c4.metric("High-risk compounds", f"{high_count}")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(valid_df):
        ranked = valid_df.sort_values(["flagged_assays", "mean_probability"], ascending=[False, False]).reset_index(drop=True)
        st.markdown("### Ranked Screening Results")
        st.dataframe(ranked, use_container_width=True, hide_index=True)
    else:
        st.warning("No valid SMILES rows were found in the processed batch.")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Screening Report (CSV)",
        data=csv_bytes,
        file_name="batch_screening_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    if not smiles_input.strip():
        st.error("Please enter a SMILES string.")
        st.stop()

    X, mol = smiles_to_features(smiles_input.strip())
    if mol is None:
        st.error("Invalid SMILES string. Please verify the structure.")
        st.stop()

    predictions = predict_targets_for_feature_vector(X, model_choice, use_tuned_threshold, manual_threshold)

    pred_df = prediction_dataframe(predictions)
    n_toxic = int((pred_df["Risk"] == "HIGH").sum())
    overall_risk, risk_class, risk_text = risk_badge(n_toxic, len(TOX21_TARGETS))
    mean_prob = float(pred_df["Probability"].mean())
    top_target = pred_df.iloc[0]["Target"]
    top_prob = float(pred_df.iloc[0]["Probability"])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Prediction Summary")
    st.markdown(
        f"""
        <span class="risk-pill {risk_class}">{overall_risk} RISK</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi"><div class="label">Flagged Assays</div><div class="value">{n_toxic}/{len(TOX21_TARGETS)}</div></div>
          <div class="kpi"><div class="label">Mean Toxicity Probability</div><div class="value">{mean_prob:.3f}</div></div>
          <div class="kpi"><div class="label">Highest-Risk Endpoint</div><div class="value">{top_target} ({top_prob:.3f})</div></div>
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:0.25rem;'>{risk_text}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Molecule")
        st.image(mol_to_image(mol), use_container_width=True)

        calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCRIPTORS)
        props = dict(zip(PHYSCHEM_DESCRIPTORS, calc.CalcDescriptors(mol)))
        prop_df = pd.DataFrame(
            [{"Property": k, "Value": round(float(v), 3)} for k, v in props.items()]
        )
        st.markdown("#### Physicochemical Profile")
        st.dataframe(prop_df, use_container_width=True, hide_index=True)

    with right:
        tabs = st.tabs(["Predictions", "Explainability", "Model Diagnostics"])

        with tabs[0]:
            st.markdown("#### Target-wise Toxicity")
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            probs = pred_df["Probability"].tolist()
            labels = pred_df["Target"].tolist()
            th_map = dict(zip(pred_df["Target"], pred_df["Threshold"]))
            th_line = [th_map[t] for t in labels]
            colors = ["#ef4444" if p >= t else "#16a34a" for p, t in zip(probs, th_line)]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(labels, probs, color=colors, edgecolor="white", linewidth=0.6)
            ax.plot(labels, th_line, color="#334155", linestyle="--", linewidth=1.3, label="Threshold")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Toxicity Probability")
            ax.set_xlabel("Tox21 Target")
            ax.tick_params(axis="x", rotation=45)
            ax.set_facecolor("#f8fafc")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            risk_counts = pred_df["Risk"].value_counts().reindex(["HIGH", "LOW"]).fillna(0)
            fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
            ax2.pie(
                risk_counts.values,
                labels=risk_counts.index,
                colors=["#ef4444", "#16a34a"],
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
            )
            ax2.set_title("High vs Low Risk Endpoints")
            st.pyplot(fig2)
            plt.close()

        with tabs[1]:
            st.markdown("#### SHAP Evidence")
            shap_cols = st.columns(3)
            shap_files = [
                ("shap_bar_NR_AR_LBD.png", "NR-AR-LBD: top drivers"),
                ("shap_waterfall_NR_AR_LBD.png", "NR-AR-LBD: local explanation"),
                ("shap_waterfall_NR_AhR.png", "NR-AhR: local explanation"),
            ]
            for col, (fname, caption) in zip(shap_cols, shap_files):
                try:
                    col.image(fname, caption=caption, use_container_width=True)
                except Exception:
                    col.info(f"{fname} not found")

            try:
                st.image(
                    "global_feature_importance.png",
                    caption="Global feature importance across all targets",
                    use_container_width=True,
                )
            except Exception:
                st.info("global_feature_importance.png not found")

        with tabs[2]:
            st.markdown("#### Training Diagnostics")
            diag_cols = st.columns(3)
            diag_files = [
                ("model_roc_auc_comparison.png", "ROC-AUC by model and target"),
                ("best_model_pr_auc_by_target.png", "Best model PR-AUC"),
                ("optimal_thresholds_by_target.png", "Tuned thresholds"),
            ]
            for col, (fname, cap) in zip(diag_cols, diag_files):
                try:
                    col.image(fname, caption=cap, use_container_width=True)
                except Exception:
                    col.info(f"{fname} not found")

            if st.checkbox("Show full model performance table"):
                try:
                    perf = pd.read_csv("model_performance_best_models.csv")
                    st.dataframe(perf, use_container_width=True, hide_index=True)
                except Exception:
                    st.warning("model_performance_best_models.csv not found")
