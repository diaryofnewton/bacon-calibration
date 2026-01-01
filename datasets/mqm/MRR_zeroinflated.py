"""
Zero-inflated regression models for MQM prediction.

Benchmarks Tweedie GLM, Hurdle (Logistic+Ridge), and Zero-inflated Gamma
against the existing Ridge baseline on systems with problematic R2.

The MQM target is non-negative with a point mass at zero (~10-30%),
which violates Ridge's Gaussian assumption and allows negative predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegressionCV,
    RidgeCV,
    TweedieRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
GLM_ALPHAS = np.logspace(-2, 4, 15)


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing: fit once per outer fold, reuse transformed data
# ─────────────────────────────────────────────────────────────────────────

def make_preprocessor(feature_cols: list[str], emb_pca: int | None, seed: int):
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]

    if emb_cols:
        pca_n = min(emb_pca or 64, len(emb_cols))
        return ColumnTransformer(transformers=[
            ("base", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]), non_emb),
            ("emb", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=seed)),
            ]), emb_cols),
        ])
    else:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])


def _transform_fold(X_tr_raw, X_te_raw, features, emb_pca, seed):
    """Fit preprocessor on training data, return transformed arrays."""
    pre = make_preprocessor(features, emb_pca, seed)
    Xtr = pre.fit_transform(X_tr_raw)
    Xte = pre.transform(X_te_raw)
    return Xtr, Xte


# ─────────────────────────────────────────────────────────────────────────
# Model 0: Ridge baseline
# ─────────────────────────────────────────────────────────────────────────

def crossfit_ridge(df, features, folds, seed, emb_pca):
    X = df[features]
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = _transform_fold(X.iloc[tr], X.iloc[te], features, emb_pca, seed)
        ridge = RidgeCV(alphas=ALPHAS)
        ridge.fit(Xtr, y[tr])
        oof[te] = ridge.predict(Xte)
    return oof


# ─────────────────────────────────────────────────────────────────────────
# Model 1: Tweedie GLM (with inner alpha CV on transformed data)
# ─────────────────────────────────────────────────────────────────────────

def _cv_alpha_tweedie(Xtr, ytr, seed, power):
    """Fast inner 3-fold CV for alpha on already-transformed data."""
    best_alpha, best_mse = GLM_ALPHAS[0], np.inf
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed + 1)
    for a in GLM_ALPHAS:
        fold_mses = []
        for itr, ite in inner_kf.split(Xtr):
            try:
                m = TweedieRegressor(power=power, alpha=a, link="log", max_iter=1000)
                m.fit(Xtr[itr], ytr[itr])
                pred = m.predict(Xtr[ite])
                fold_mses.append(mean_squared_error(ytr[ite], pred))
            except Exception:
                fold_mses.append(np.inf)
        mse = np.mean(fold_mses)
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    return best_alpha


def crossfit_tweedie(df, features, folds, seed, emb_pca, power=1.5):
    X = df[features]
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        Xtr, Xte = _transform_fold(X.iloc[tr], X.iloc[te], features, emb_pca, seed)
        alpha = _cv_alpha_tweedie(Xtr, y[tr], seed, power)
        m = TweedieRegressor(power=power, alpha=alpha, link="log", max_iter=1000)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict(Xte)
    return oof


# ─────────────────────────────────────────────────────────────────────────
# Model 2: Hurdle (Logistic gate + Ridge on positives)
# ─────────────────────────────────────────────────────────────────────────

def crossfit_hurdle(df, features, folds, seed, emb_pca):
    X = df[features]
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)

    for tr, te in kf.split(X):
        Xtr, Xte = _transform_fold(X.iloc[tr], X.iloc[te], features, emb_pca, seed)
        y_tr = y[tr]

        z_tr = (y_tr > 0).astype(int)
        gate = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, random_state=seed)
        gate.fit(Xtr, z_tr)
        p_pos = gate.predict_proba(Xte)[:, 1]

        pos_mask = y_tr > 0
        if pos_mask.sum() < 10:
            oof[te] = p_pos * y_tr[pos_mask].mean() if pos_mask.any() else 0.0
            continue

        amount = RidgeCV(alphas=ALPHAS)
        amount.fit(Xtr[pos_mask], y_tr[pos_mask])
        e_pos = amount.predict(Xte).clip(min=0)

        oof[te] = p_pos * e_pos

    return oof


# ─────────────────────────────────────────────────────────────────────────
# Model 3: Zero-inflated Gamma (Logistic gate + Gamma GLM on positives)
# ─────────────────────────────────────────────────────────────────────────

def _cv_alpha_gamma(Xtr_pos, ytr_pos, seed):
    """Fast inner 3-fold CV for Gamma alpha on transformed positive data."""
    best_alpha, best_mse = GLM_ALPHAS[0], np.inf
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed + 1)
    for a in GLM_ALPHAS:
        fold_mses = []
        for itr, ite in inner_kf.split(Xtr_pos):
            try:
                m = TweedieRegressor(power=2, alpha=a, link="log", max_iter=1000)
                m.fit(Xtr_pos[itr], ytr_pos[itr])
                pred = m.predict(Xtr_pos[ite])
                fold_mses.append(mean_squared_error(ytr_pos[ite], pred))
            except Exception:
                fold_mses.append(np.inf)
        mse = np.mean(fold_mses)
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    return best_alpha


def crossfit_zi_gamma(df, features, folds, seed, emb_pca):
    X = df[features]
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)

    for tr, te in kf.split(X):
        Xtr, Xte = _transform_fold(X.iloc[tr], X.iloc[te], features, emb_pca, seed)
        y_tr = y[tr]

        z_tr = (y_tr > 0).astype(int)
        gate = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, random_state=seed)
        gate.fit(Xtr, z_tr)
        p_pos = gate.predict_proba(Xte)[:, 1]

        pos_mask = y_tr > 0
        if pos_mask.sum() < 10:
            oof[te] = p_pos * y_tr[pos_mask].mean() if pos_mask.any() else 0.0
            continue

        alpha = _cv_alpha_gamma(Xtr[pos_mask], y_tr[pos_mask], seed)
        gamma_glm = TweedieRegressor(power=2, alpha=alpha, link="log", max_iter=1000)
        gamma_glm.fit(Xtr[pos_mask], y_tr[pos_mask])
        e_pos = gamma_glm.predict(Xte)

        oof[te] = p_pos * e_pos

    return oof


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    n_neg = (y_pred < 0).sum()
    return {
        "model": model_name,
        "R2": r2,
        "RMSE": rmse,
        "Spearman": spearman,
        "neg_preds": int(n_neg),
        "pred_min": float(y_pred.min()),
        "pred_max": float(y_pred.max()),
        "pred_mean": float(y_pred.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "ridge": crossfit_ridge,
    "hurdle": crossfit_hurdle,
}


def main():
    parser = argparse.ArgumentParser(description="Zero-inflated MQM regression benchmark.")
    parser.add_argument("--tsv-path", type=Path,
                        default=Path("datasets/mqm/data/mqm_newstest2020_ende.tsv"))
    parser.add_argument("--llm-dir", type=Path,
                        default=Path("datasets/mqm/data"))
    parser.add_argument("--emb-path", type=Path,
                        default=Path("datasets/mqm/results/text_embedding_3_large_mqm_embeddings_separate.npz"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("datasets/mqm/results"))
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Systems to run on. Omit for all systems.")
    parser.add_argument("--models", nargs="*",
                        default=list(MODEL_RUNNERS.keys()),
                        choices=list(MODEL_RUNNERS.keys()) + ["tweedie", "zi_gamma"],
                        help="Models to benchmark.")
    parser.add_argument("--tweedie-power", type=float, default=1.5)
    parser.add_argument("--pooled", action="store_true",
                        help="Also run a pooled model across all selected systems.")
    parser.add_argument("--var-transform", choices=["log", "sqrt"], default="log",
                        help="Variance encoding: log (original) or sqrt")
    args = parser.parse_args()

    extra_runners = {}
    if "tweedie" in args.models:
        extra_runners["tweedie"] = lambda df, f, fo, s, e: crossfit_tweedie(
            df, f, fo, s, e, power=args.tweedie_power)
    if "zi_gamma" in args.models:
        extra_runners["zi_gamma"] = crossfit_zi_gamma
    runners = {k: MODEL_RUNNERS[k] for k in args.models if k in MODEL_RUNNERS}
    runners.update(extra_runners)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    human = load_human_mqm(args.tsv_path.resolve())
    llm_agg = load_llm_mqm(args.llm_dir.resolve())
    emb = load_embeddings(args.emb_path.resolve())
    print(f"  human: {len(human)} | llm_agg: {len(llm_agg)} | emb: {len(emb)}", flush=True)

    systems = args.systems if args.systems else sorted(human["system"].unique())
    print(f"  systems: {systems}", flush=True)

    all_results = []

    for system in systems:
        print(f"\n{'='*60}", flush=True)
        print(f"  System: {system}", flush=True)
        print(f"{'='*60}", flush=True)

        df, features = build_feature_table(
            human, llm_agg, emb, use_uncertainty=True, system_filter=system,
            var_transform=args.var_transform,
        )
        if len(df) < 20:
            print(f"  Skipping (only {len(df)} rows)", flush=True)
            continue

        y = df[TARGET].astype(float).values
        n_zero = (y == 0).sum()
        print(f"  n={len(df)}  features={len(features)}  "
              f"zeros={n_zero} ({n_zero/len(df):.1%})  "
              f"y_mean={y.mean():.3f}  y_std={y.std():.3f}", flush=True)

        oof_dict = {}
        for model_name, runner in runners.items():
            print(f"  Running {model_name}...", flush=True)
            oof = runner(df, features, FOLDS, SEED, EMB_PCA)
            oof_dict[model_name] = oof
            res = evaluate(y, oof, model_name)
            res["system"] = system
            all_results.append(res)

        tag = system.replace(".", "_")
        oof_df = pd.DataFrame({"id": df["id"].values, "y_true": y, **oof_dict})
        oof_df.to_csv(out_dir / f"mrr_mqm_zeroinflated_oof_{tag}.csv", index=False)

    if args.pooled and len(systems) > 1:
        print(f"\n{'='*60}", flush=True)
        print(f"  Pooled (all {len(systems)} systems)", flush=True)
        print(f"{'='*60}", flush=True)

        df, features = build_feature_table(
            human, llm_agg, emb, use_uncertainty=True, system_filter=None,
            var_transform=args.var_transform,
        )
        y = df[TARGET].astype(float).values
        n_zero = (y == 0).sum()
        print(f"  n={len(df)}  features={len(features)}  "
              f"zeros={n_zero} ({n_zero/len(df):.1%})  "
              f"y_mean={y.mean():.3f}  y_std={y.std():.3f}", flush=True)

        oof_dict = {}
        for model_name, runner in runners.items():
            print(f"  Running {model_name}...", flush=True)
            oof = runner(df, features, FOLDS, SEED, EMB_PCA)
            oof_dict[model_name] = oof
            res = evaluate(y, oof, model_name)
            res["system"] = "pooled"
            all_results.append(res)

        oof_df = pd.DataFrame({"id": df["id"].values, "y_true": y, **oof_dict})
        oof_df.to_csv(out_dir / "mrr_mqm_zeroinflated_oof_pooled.csv", index=False)

    summary = pd.DataFrame(all_results)
    col_order = ["system", "model", "R2", "RMSE", "Spearman",
                 "neg_preds", "pred_min", "pred_max", "pred_mean"]
    summary = summary[col_order]

    print(f"\n{'='*60}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    summary.to_csv(out_dir / "mrr_mqm_zeroinflated_summary.csv", index=False)
    print(f"\nSaved: {out_dir / 'mrr_mqm_zeroinflated_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
