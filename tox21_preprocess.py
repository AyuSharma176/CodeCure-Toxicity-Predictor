"""
Tox21 Dataset - Loading & Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TOX21_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

PHYSCHEM_DESCRIPTORS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_tox21(data_dir="."):
    data_dir = Path(data_dir)
    for fname in ["tox21.csv", "tox21_10k_data_all.csv", "train.csv", "data.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            print(f"Loading {fpath} ...")
            df = pd.read_csv(fpath)
            print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
            return df
    raise FileNotFoundError(
        f"No CSV found in '{data_dir}'.\n"
        "Download from: https://www.kaggle.com/datasets/epicskills/tox21-dataset\n"
        "Then place the CSV file in your project folder."
    )

# ── VALIDATE SMILES ───────────────────────────────────────────────────────────

def validate_smiles(df):
    print("\n[Step 1] Validating SMILES ...")
    smiles_col = None
    for col in df.columns:
        if col.lower() in {"smiles", "canonical_smiles", "smi"}:
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})

    df = df.copy()
    df["mol"] = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    n_invalid = df["mol"].isna().sum()
    df = df[df["mol"].notna()].reset_index(drop=True)
    print(f"  Removed {n_invalid} invalid SMILES. Valid: {len(df):,}")
    return df

# ── FEATURES ──────────────────────────────────────────────────────────────────

def compute_morgan_fingerprints(df, radius=2, n_bits=2048):
    print(f"\n[Step 2a] Morgan fingerprints (ECFP{radius*2}, {n_bits} bits) ...")
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [gen.GetFingerprintAsNumPy(mol) for mol in df["mol"]]
    X = np.stack(fps)
    print(f"  Shape: {X.shape}")
    return X

def compute_physchem_descriptors(df):
    print(f"[Step 2b] Physicochemical descriptors ({len(PHYSCHEM_DESCRIPTORS)}) ...")
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(PHYSCHEM_DESCRIPTORS)
    rows = []
    for mol in df["mol"]:
        try:
            rows.append(list(calc.CalcDescriptors(mol)))
        except Exception:
            rows.append([np.nan] * len(PHYSCHEM_DESCRIPTORS))
    X = np.array(rows, dtype=np.float32)
    print(f"  Shape: {X.shape}")
    return X

def build_feature_blocks(df):
    X_fp = compute_morgan_fingerprints(df)
    X_desc_raw = compute_physchem_descriptors(df)
    return X_fp, X_desc_raw

# ── TARGETS ───────────────────────────────────────────────────────────────────

def prepare_targets(df):
    print("\n[Step 3] Preparing targets ...")
    available = [t for t in TOX21_TARGETS if t in df.columns]
    missing   = [t for t in TOX21_TARGETS if t not in df.columns]
    if missing:
        print(f"  Note — targets not in this file: {missing}")

    y_df = df[available].apply(pd.to_numeric, errors="coerce")

    print(f"\n  {'Target':<18} {'Labelled':>8} {'Toxic':>7} {'Toxic%':>7}")
    print("  " + "-"*45)
    stats = {}
    for col in y_df.columns:
        total   = y_df[col].notna().sum()
        n_pos   = (y_df[col] == 1).sum()
        pct     = 100 * n_pos / total if total > 0 else 0
        stats[col] = {"total": total, "n_pos": n_pos, "pos_pct": pct}
        flag = " ⚠" if pct < 5 else ""
        print(f"  {col:<18} {total:>8,} {n_pos:>7,} {pct:>6.1f}%{flag}")

    return y_df, stats

# ── SPLIT ─────────────────────────────────────────────────────────────────────

def split_dataset(X, y_df, test_size=0.2, random_state=42):
    print(f"\n[Step 4] Train/test split ({int((1-test_size)*100)}/{int(test_size*100)}) ...")
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)

    splits = {
        "X_train": X[idx_train],
        "X_test":  X[idx_test],
        "targets": {}
    }

    for target in y_df.columns:
        y = y_df[target].values
        train_mask = ~np.isnan(y[idx_train])
        test_mask  = ~np.isnan(y[idx_test])
        y_train = y[idx_train][train_mask].astype(int)
        y_test  = y[idx_test][test_mask].astype(int)
        n_neg   = (y_train == 0).sum()
        n_pos   = (y_train == 1).sum()
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        splits["targets"][target] = {
            "X_train": X[idx_train][train_mask],
            "y_train": y_train,
            "X_test":  X[idx_test][test_mask],
            "y_test":  y_test,
            "pos_weight": pos_weight,
            "class_weight": {0: 1.0, 1: pos_weight},
        }

    print(f"  Train: {len(idx_train):,}  |  Test: {len(idx_test):,}")
    return splits

def transform_descriptors(X_desc_raw, idx_train):
    print("\n[Step 2c] Fitting descriptor preprocessing on train split only ...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_desc_train = imputer.fit_transform(X_desc_raw[idx_train])
    X_desc_train = scaler.fit_transform(X_desc_train)

    X_desc_all = imputer.transform(X_desc_raw)
    X_desc_all = scaler.transform(X_desc_all)

    print("  Descriptor preprocessing fitted and applied.")
    return X_desc_all.astype(np.float32), imputer, scaler

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=" * 55)
    print("  Tox21 Preprocessing Pipeline")
    print("=" * 55)

    df              = load_tox21(data_dir)
    df              = validate_smiles(df)
    X_fp, X_desc_raw = build_feature_blocks(df)
    y_df, stats     = prepare_targets(df)

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_desc, imputer, scaler = transform_descriptors(X_desc_raw, idx_train)
    X = np.hstack([X_fp, X_desc]).astype(np.float32)
    feat_names = ([f"morgan_{i}" for i in range(X_fp.shape[1])] + PHYSCHEM_DESCRIPTORS)
    print(f"\n  Final feature matrix: {X.shape}")

    splits = split_dataset(X, y_df, test_size=0.2, random_state=42)

    # Save outputs
    np.save("X_train.npy",      splits["X_train"])
    np.save("X_test.npy",       splits["X_test"])
    np.save("feature_names.npy", np.array(feat_names))
    with open("preprocess_artifacts.pkl", "wb") as f:
        pickle.dump({
            "imputer": imputer,
            "scaler": scaler,
            "physchem_descriptors": PHYSCHEM_DESCRIPTORS,
            "morgan_radius": 2,
            "morgan_n_bits": 2048,
        }, f)
    with open("splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    with open("target_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("\n✓ Done! Saved: X_train.npy, X_test.npy, splits.pkl, preprocess_artifacts.pkl")
