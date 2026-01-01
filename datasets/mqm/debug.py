"""
Comprehensive comparison across all translators:
  - var_transform: log vs sqrt
  - model: Ridge vs Hurdle
  - features: full | llm_only | emb_only | intercept
Reports R2 and Spearman for every combination.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from datasets.mqm.MRR import (
    TARGET,
    build_feature_table,
    load_embeddings,
    load_human_mqm,
    load_llm_mqm,
)

ALPHAS = np.logspace(-4, 4, 40)
SEED = 42
FOLDS = 5
EMB_PCA = 64


def make_pre(feature_cols, seed):
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]
    if emb_cols:
        pca_n = min(EMB_PCA, len(emb_cols))
        return ColumnTransformer(transformers=[
            ("base", Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
            ]), non_emb),
            ("emb", Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=seed)),
            ]), emb_cols),
        ])
    elif non_emb:
        return Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())])
    return None


def transform_fold(X_tr, X_te, features, seed):
    pre = make_pre(features, seed)
    if pre is None:
        return X_tr.values, X_te.values
    return pre.fit_transform(X_tr), pre.transform(X_te)


def cv_ridge(df, features):
    X, y = df[features], df[TARGET].astype(float).values
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = transform_fold(X.iloc[tr], X.iloc[te], features, SEED)
        m = RidgeCV(alphas=ALPHAS)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
    return oof


def cv_hurdle(df, features):
    X, y = df[features], df[TARGET].astype(float).values
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = transform_fold(X.iloc[tr], X.iloc[te], features, SEED)
        y_tr = y[tr]
        gate = LogisticRegressionCV(Cs=10, cv=3, max_iter=2000, random_state=SEED)
        gate.fit(Xtr, (y_tr > 0).astype(int))
        p_pos = gate.predict_proba(Xte)[:, 1]
        pos = y_tr > 0
        if pos.sum() < 10:
            oof[te] = p_pos * y_tr[pos].mean() if pos.any() else 0.0
            continue
        amt = RidgeCV(alphas=ALPHAS)
        amt.fit(Xtr[pos], y_tr[pos])
        e_pos = amt.predict(Xte).clip(min=0)
        oof[te] = p_pos * e_pos
    return oof


def evaluate(y, p):
    return {
        "R2": r2_score(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "Spearman": pd.Series(y).corr(pd.Series(p), method="spearman"),
    }


def run_system(system, human, llm_agg, emb):
    """Run all comparisons for one system, return list of result dicts."""
    rows = []
    y_cache = {}

    for vt in ["log", "sqrt"]:
        # full: LLM scores + uncertainty + embeddings
        df_full, feats_full = build_feature_table(
            human, llm_agg, emb, use_uncertainty=True,
            system_filter=system, var_transform=vt,
        )
        y = df_full[TARGET].astype(float).values
        y_cache[vt] = y

        for model_name, runner in [("ridge", cv_ridge), ("hurdle", cv_hurdle)]:
            oof = runner(df_full, feats_full)
            m = evaluate(y, oof)
            rows.append({"system": system, "features": "full", "var_transform": vt,
                         "model": model_name, **m})

        # llm_only: LLM scores + uncertainty, no embeddings
        df_llm, feats_llm = build_feature_table(
            human, llm_agg, None, use_uncertainty=True,
            system_filter=system, var_transform=vt,
        )
        for model_name, runner in [("ridge", cv_ridge), ("hurdle", cv_hurdle)]:
            oof = runner(df_llm, feats_llm)
            m = evaluate(y, oof)
            rows.append({"system": system, "features": "llm_only", "var_transform": vt,
                         "model": model_name, **m})

    # emb_only: embeddings only, no LLM scores (var_transform irrelevant)
    df_emb, _ = build_feature_table(
        human, llm_agg, emb, use_uncertainty=False, system_filter=system,
    )
    emb_feats = [c for c in df_emb.columns if c.startswith("emb__")]
    y = df_emb[TARGET].astype(float).values
    for model_name, runner in [("ridge", cv_ridge), ("hurdle", cv_hurdle)]:
        oof = runner(df_emb, emb_feats)
        m = evaluate(y, oof)
        rows.append({"system": system, "features": "emb_only", "var_transform": "-",
                      "model": model_name, **m})

    # intercept: predict the mean (no model/var_transform distinction)
    y_int = y_cache.get("sqrt", y)
    rmse_int = float(np.std(y_int))
    rows.append({"system": system, "features": "intercept", "var_transform": "-",
                 "model": "-", "R2": 0.0, "RMSE": rmse_int, "Spearman": 0.0})

    return rows


def main():
    root = Path(__file__).resolve().parent.parent.parent
    print("Loading data...", flush=True)
    human = load_human_mqm(root / "datasets/mqm/data/mqm_newstest2020_ende.tsv")
    llm_agg = load_llm_mqm(root / "datasets/mqm/data")
    emb = load_embeddings(root / "datasets/mqm/results/text_embedding_3_large_mqm_embeddings_separate.npz")
    print(f"  human={len(human)}  llm_agg={len(llm_agg)}  emb={len(emb)}\n", flush=True)

    systems = sorted(human["system"].unique())
    all_rows = []

    for i, system in enumerate(systems):
        print(f"[{i+1}/{len(systems)}] {system}...", flush=True)
        rows = run_system(system, human, llm_agg, emb)
        all_rows.extend(rows)
        # print quick summary for this system
        best = max(rows, key=lambda r: r["R2"])
        print(f"  best R2={best['R2']:.4f} ({best['features']}|{best['var_transform']}|{best['model']})", flush=True)

    # pooled
    print(f"[pooled] all systems...", flush=True)
    rows = run_system(None, human, llm_agg, emb)
    for r in rows:
        r["system"] = "pooled"
    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # pivot: for each system, show R2 and Spearman for every config
    out_dir = root / "datasets" / "mqm" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "debug_comprehensive.csv", index=False)

    # summary tables
    print(f"\n{'='*90}", flush=True)
    print("  R2 COMPARISON", flush=True)
    print(f"{'='*90}", flush=True)

    df["config"] = df["features"] + "|" + df["var_transform"] + "|" + df["model"]
    pivot_r2 = df.pivot_table(index="system", columns="config", values="R2")
    cols_sorted = sorted(pivot_r2.columns, key=lambda c: -pivot_r2[c].mean())
    pivot_r2 = pivot_r2[cols_sorted]
    print(pivot_r2.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    print(f"\n{'='*90}", flush=True)
    print("  SPEARMAN COMPARISON", flush=True)
    print(f"{'='*90}", flush=True)

    pivot_sp = df.pivot_table(index="system", columns="config", values="Spearman")
    pivot_sp = pivot_sp[cols_sorted]
    print(pivot_sp.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    # condensed: best config per system
    print(f"\n{'='*90}", flush=True)
    print("  BEST CONFIG PER SYSTEM (by R2)", flush=True)
    print(f"{'='*90}", flush=True)
    best_per_sys = df.loc[df.groupby("system")["R2"].idxmax()]
    print(best_per_sys[["system", "features", "var_transform", "model", "R2", "Spearman"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print(f"\nSaved: {out_dir / 'debug_comprehensive.csv'}", flush=True)


if __name__ == "__main__":
    main()
