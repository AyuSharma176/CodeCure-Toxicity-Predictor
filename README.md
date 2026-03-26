# 🧬 CodeCure — Drug Toxicity Predictor
### CodeCure AI Hackathon | Track A: Drug Toxicity Prediction

Predicts potential toxicity of chemical compounds across **12 Tox21 assay targets** using machine learning on molecular fingerprints and physicochemical descriptors.

---

## 🎯 Problem Statement
Drug development frequently fails due to unexpected toxicity. Early AI-based prediction of toxic compounds can significantly reduce development costs and improve patient safety.

---

## 🚀 Demo
> Input any SMILES string → Get toxicity predictions across 12 targets → See SHAP explanation of WHY

![App Screenshot](screenshots/app_demo.png)

---

## 📊 Model Performance

| Model | Mean ROC-AUC | Best Target |
|-------|-------------|-------------|
| Random Forest | **0.853** | NR-AR-LBD (0.890) |
| XGBoost | 0.832 | SR-MMP (0.898) |

---

## 🔬 Key Biological Insights

From SHAP analysis across all 12 targets:

- **Molecular Weight & MolLogP** are the strongest toxicity predictors — larger, more lipophilic molecules are consistently more toxic
- **Ring Count & Aromatic Rings** strongly predict androgen receptor binding (NR-AR-LBD) — steroid-like multi-ring structures are high risk
- **FractionCSP3** is protective — more 3D saturated molecules tend to be safer
- **TPSA** influences membrane permeability and cellular uptake of toxic compounds

---

## 🗂️ Project Structure
```
CodeCure/
├── tox21_preprocess.py   # Data loading, SMILES validation, feature extraction
├── tox21_train.py        # XGBoost + Random Forest training across 12 targets
├── tox21_explain.py      # SHAP feature importance + waterfall plots
├── app.py                # Streamlit prediction interface
└── README.md
```

## ⚙️ Pipeline
```
Tox21 Dataset (12k compounds)
        ↓
SMILES Validation (RDKit)
        ↓
Feature Extraction
  ├── Morgan Fingerprints (ECFP4, 2048 bits)
  └── Physicochemical Descriptors (MolWt, LogP, TPSA...)
        ↓
Model Training (per target, class-imbalance handled)
  ├── Random Forest (class_weight balanced)
  └── XGBoost (scale_pos_weight)
        ↓
SHAP Explainability → Feature Attribution
        ↓
Streamlit Prediction App
```

## 🛠️ Setup
```bash
git clone https://github.com/YOURUSERNAME/CodeCure-Toxicity-Predictor
cd CodeCure-Toxicity-Predictor
python -m venv venv
venv\Scripts\activate
pip install rdkit scikit-learn xgboost shap streamlit pandas numpy matplotlib seaborn
```

Download [Tox21 dataset](https://www.kaggle.com/datasets/epicskills/tox21-dataset) and place CSV in project folder, then:
```bash
python tox21_preprocess.py
python tox21_train.py
python tox21_explain.py
streamlit run app.py
```

## 📦 Dataset
- **Primary:** [Tox21 Dataset](https://www.kaggle.com/datasets/epicskills/tox21-dataset) — ~12,000 compounds, 12 toxicity assays
- **Features:** 2,058 total (2,048 Morgan fingerprint bits + 10 physicochemical descriptors)

## 🧪 Example Predictions

| Compound | Known Risk | Our Prediction |
|----------|-----------|----------------|
| Bisphenol A | Endocrine disruptor | HIGH RISK |
| Aspirin | Safe at normal doses | LOW RISK |
| Tamoxifen | ER modulator | MODERATE RISK |
| Atrazine | Herbicide / toxic | HIGH RISK |

## 👥 Team
Built for CodeCure AI Hackathon — Track A
```

---

**Step 3 — Take screenshots (5 mins)**

Create a `screenshots/` folder and take 3 screenshots of your app:
- The main dashboard with model performance table
- A prediction result for Bisphenol A (high risk compound)
- The SHAP plots section

Add them to your README.

---

**Step 4 — Final checklist before submission**
```
✅ tox21_preprocess.py
✅ tox21_train.py
✅ tox21_explain.py
✅ app.py
✅ README.md with results table
✅ requirements.txt
✅ SHAP plots in repo
✅ GitHub repo public