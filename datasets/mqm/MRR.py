"""
MRR for MQM newstest2020 en-de + two-stage hardness.

Unit of analysis: (system, seg_id) = one translated segment.
We iterate over each MT system ("translator") and also run a pooled model.

Judge model:
    y = b + L w [+ variance interactions] [+ embedding intercept]
    where L = vector of per-LLM mean MQM scores for the item

Uncertainty:
    sample-level variance of LLM MQM scores per (model, system, seg_id),
    computed across raters and repeated samples.

Hardness model (nested cross-fit):
    target = log(residual^2 + eps), features = embeddings only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "human_avg_mqm"
ALPHAS = np.logspace(-4, 4, 40)


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def _mqm_weight(category: str, severity: str) -> float:
    sev = severity.strip().lower()
    cat = category.strip().lower()
    if "non-translation" in cat:
        return 25.0
    if sev == "neutral":
        return 0.0
    if sev == "minor" and cat.startswith("fluency/punctuation"):
        return 0.1
    if sev == "minor":
        return 1.0
    if sev == "major":
        return 5.0
    return 0.0


def load_human_mqm(tsv_path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns [system, seg_id, human_avg_mqm]."""
    hdf = pd.read_csv(tsv_path, sep="\t")
    sev = hdf["severity"].str.strip().str.lower()
    cat = hdf["category"].str.strip().str.lower()

    weight = pd.Series(0.0, index=hdf.index)
    weight[sev == "minor"] = 1.0
    weight[sev == "major"] = 5.0
    weight[(sev == "minor") & cat.str.startswith("fluency/punctuation")] = 0.1
    weight[cat.str.contains("non-translation", na=False)] = 25.0
    hdf["weight"] = weight

    per_rater = hdf.groupby(["system", "seg_id", "rater"])["weight"].sum().reset_index()
    per_rater.rename(columns={"weight": "rater_mqm"}, inplace=True)

    human_avg = (
        per_rater.groupby(["system", "seg_id"])["rater_mqm"]
        .mean()
        .reset_index()
        .rename(columns={"rater_mqm": TARGET})
    )
    human_avg["seg_id"] = human_avg["seg_id"].astype(str)
    return human_avg


def load_llm_mqm(data_dir: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Load LLM JSONL files, compute per-row MQM scores, and aggregate to
    per-(model, system, seg_id) mean and variance."""
    frames = []
    for fpath in sorted(data_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in open(fpath)]
        frames.append(pd.DataFrame(records))
    df = pd.concat(frames, ignore_index=True)

    if models:
        df = df[df["model"].isin(models)]

    # vectorized MQM score computation
    errors_lists = df["mqm"].apply(
        lambda x: x.get("errors", []) if isinstance(x, dict) else []
    )
    has_err = errors_lists.apply(len) > 0
    err_series = errors_lists[has_err].explode().dropna()
    err_detail = pd.DataFrame(err_series.tolist(), index=err_series.index)
    err_detail.columns = [c.lower() for c in err_detail.columns]

    cat_l = err_detail["category"].str.strip().str.lower()
    sev_l = err_detail["severity"].str.strip().str.lower()
    w = pd.Series(0.0, index=err_detail.index)
    w[sev_l == "minor"] = 1.0
    w[sev_l == "major"] = 5.0
    w[(sev_l == "minor") & cat_l.str.startswith("fluency/punctuation")] = 0.1
    w[cat_l.str.contains("non-translation", na=False)] = 25.0

    row_scores = w.groupby(level=0).sum()
    df["mqm_score"] = 0.0
    df.loc[row_scores.index, "mqm_score"] = row_scores.values
    df["seg_id"] = df["seg_id"].astype(str)

    # aggregate: mean and variance per (model, system, seg_id)
    agg = (
        df.groupby(["model", "system", "seg_id"])["mqm_score"]
        .agg(["mean", "var", "count"])
        .rename(columns={"mean": "llm_mean", "var": "llm_var", "count": "n_samples"})
        .reset_index()
    )
    agg["llm_var"] = agg["llm_var"].fillna(0.0)
    return agg


def load_embeddings(npz_path: Path) -> pd.DataFrame:
    arr = np.load(npz_path, allow_pickle=True)
    ids = [str(x) for x in arr["task_ids"]]
    emb = arr["embeddings"]
    cols = [f"emb__{i}" for i in range(emb.shape[1])]
    e = pd.DataFrame(emb, columns=cols)
    e.insert(0, "task_id", ids)
    return e.drop_duplicates("task_id")


# ─────────────────────────────────────────────────────────────────────────
# Feature table
# ─────────────────────────────────────────────────────────────────────────

def build_feature_table(
    human_df: pd.DataFrame,
    llm_agg: pd.DataFrame,
    emb_df: pd.DataFrame | None,
    use_uncertainty: bool,
    system_filter: str | None = None,
    var_transform: str = "log",
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix for one system (or pooled).

    var_transform:
        "log"  – log(var.clip(1e-8)) + score×logvar  (original)
        "sqrt" – sqrt(var) + has_var indicator + score×sqrt(var)
    """
    h = human_df.copy()
    la = llm_agg.copy()

    if system_filter is not None:
        h = h[h["system"] == system_filter].copy()
        la = la[la["system"] == system_filter].copy()

    pivot_mean = la.pivot_table(
        index=["system", "seg_id"], columns="model", values="llm_mean"
    )
    model_names = list(pivot_mean.columns)
    pivot_mean.columns = [f"s__{m}" for m in model_names]
    pivot_mean = pivot_mean.reset_index()

    base = h.merge(pivot_mean, on=["system", "seg_id"], how="inner").copy()
    feature_cols = [f"s__{m}" for m in model_names]

    if use_uncertainty:
        pivot_var = la.pivot_table(
            index=["system", "seg_id"], columns="model", values="llm_var"
        )
        pivot_var.columns = [f"var__{m}" for m in pivot_var.columns]
        pivot_var = pivot_var.reset_index()
        base = base.merge(pivot_var, on=["system", "seg_id"], how="left")

        for m in model_names:
            s_col = f"s__{m}"
            v_col = f"var__{m}"
            base[v_col] = base[v_col].fillna(0.0)

            if var_transform == "log":
                t_col = f"logvar__{m}"
                base[t_col] = np.log(base[v_col].clip(lower=1e-8))
                sxv = f"sxv__{m}"
                base[sxv] = base[s_col] * base[t_col]
                feature_cols.extend([t_col, sxv])
            elif var_transform == "sqrt":
                sq_col = f"sqrtvar__{m}"
                hv_col = f"hasvar__{m}"
                sxv = f"sxv__{m}"
                base[sq_col] = np.sqrt(base[v_col])
                base[hv_col] = (base[v_col] > 1e-8).astype(float)
                base[sxv] = base[s_col] * base[sq_col]
                feature_cols.extend([sq_col, hv_col, sxv])
            else:
                raise ValueError(f"Unknown var_transform: {var_transform!r}")

    if emb_df is not None:
        base["task_id"] = base["system"] + "::" + base["seg_id"]
        base = base.merge(emb_df, on="task_id", how="left")
        emb_cols = [c for c in base.columns if c.startswith("emb__")]
        feature_cols.extend(emb_cols)

    base["id"] = base["system"] + "::" + base["seg_id"]
    base.replace([np.inf, -np.inf], np.nan, inplace=True)
    return base, feature_cols


# ─────────────────────────────────────────────────────────────────────────
# Model pipeline (reused from PERSUADE)
# ─────────────────────────────────────────────────────────────────────────

def make_pipe(feature_cols: list[str], seed: int, emb_pca: int | None) -> Pipeline:
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]
    if emb_cols:
        pca_n = min(emb_pca or 64, len(emb_cols))
        pre = ColumnTransformer(
            transformers=[
                ("base", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]), non_emb),
                ("emb", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_n, random_state=seed)),
                ]), emb_cols),
            ]
        )
    else:
        pre = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
    return Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=ALPHAS, fit_intercept=True))])


def crossfit_predict(
    df: pd.DataFrame, features: list[str], target: str,
    folds: int, seed: int, emb_pca: int | None,
) -> tuple[np.ndarray, dict]:
    X = df[features]
    y = df[target].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    alphas = []
    for tr, te in kf.split(X):
        pipe = make_pipe(features, seed=seed, emb_pca=emb_pca)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict(X.iloc[te])
        alphas.append(float(pipe.named_steps["ridge"].alpha_))
    return oof, {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "cv_r2": float(r2_score(y, oof)),
        "mean_alpha": float(np.mean(alphas)),
    }


# ─────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────

def run_judge(
    df: pd.DataFrame, features: list[str],
    folds: int, seed: int, emb_pca: int | None,
    out_dir: Path, exp: str,
) -> pd.DataFrame:
    oof, m = crossfit_predict(df, features, TARGET, folds, seed, emb_pca)
    y = df[TARGET].astype(float).values
    sp = float(pd.Series(y).corr(pd.Series(oof), method="spearman"))
    metrics = pd.DataFrame([{
        "experiment": exp,
        "n_samples": len(df),
        "n_features_raw": len(features),
        "cv_rmse": m["cv_rmse"],
        "cv_r2": m["cv_r2"],
        "cv_spearman": sp,
        "mean_alpha": m["mean_alpha"],
    }])
    preds = pd.DataFrame({
        "id": df["id"].values, "y_true": y,
        "y_pred_oof": oof, "residual_sq_oof": (y - oof) ** 2,
    })
    metrics.to_csv(out_dir / f"mrr_mqm_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_mqm_judge_oof_{exp}.csv", index=False)
    return metrics


def run_intercept_only(
    df: pd.DataFrame, folds: int, seed: int, out_dir: Path, exp: str,
) -> pd.DataFrame:
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(df):
        oof[te] = float(np.mean(y[tr]))
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    r2 = float(r2_score(y, oof))
    sp = float(pd.Series(y).corr(pd.Series(oof), method="spearman"))
    metrics = pd.DataFrame([{
        "experiment": exp, "n_samples": len(df),
        "n_features_raw": 0, "cv_rmse": rmse, "cv_r2": r2,
        "cv_spearman": sp, "mean_alpha": np.nan,
    }])
    preds = pd.DataFrame({
        "id": df["id"].values, "y_true": y,
        "y_pred_oof": oof, "residual_sq_oof": (y - oof) ** 2,
    })
    metrics.to_csv(out_dir / f"mrr_mqm_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_mqm_judge_oof_{exp}.csv", index=False)
    return metrics


def run_two_stage_hardness(
    df: pd.DataFrame,
    judge_features: list[str],
    hardness_features: list[str],
    folds: int, seed: int,
    judge_emb_pca: int | None,
    hardness_emb_pca: int | None,
    out_dir: Path, tag: str,
    eps: float = 1e-6,
) -> pd.DataFrame:
    y = df[TARGET].astype(float).values
    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_j = np.zeros_like(y, dtype=float)
    oof_r2 = np.zeros_like(y, dtype=float)
    oof_h = np.zeros_like(y, dtype=float)

    for tr_idx, te_idx in outer.split(df):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)

        tr_j_oof, _ = crossfit_predict(tr, judge_features, TARGET, folds, seed, judge_emb_pca)
        tr_resid2 = (tr[TARGET].astype(float).values - tr_j_oof) ** 2
        tr_h_target = np.log(tr_resid2 + eps)

        h_pipe = make_pipe(hardness_features, seed=seed, emb_pca=hardness_emb_pca)
        h_pipe.fit(tr[hardness_features], tr_h_target)

        j_pipe = make_pipe(judge_features, seed=seed, emb_pca=judge_emb_pca)
        j_pipe.fit(tr[judge_features], tr[TARGET].astype(float).values)

        te_j_pred = j_pipe.predict(te[judge_features])
        te_resid2 = (te[TARGET].astype(float).values - te_j_pred) ** 2
        te_h_pred = h_pipe.predict(te[hardness_features])

        oof_j[te_idx] = te_j_pred
        oof_r2[te_idx] = te_resid2
        oof_h[te_idx] = te_h_pred

    y_h = np.log(oof_r2 + eps)
    q90 = float(np.quantile(oof_h, 0.9))
    denom = float(np.mean(oof_r2))
    lift = float(np.mean(oof_r2[oof_h >= q90]) / denom) if denom > 0 else np.nan

    out = pd.DataFrame([{
        "tag": tag,
        "n_samples": len(df),
        "judge_cv_rmse": float(np.sqrt(mean_squared_error(y, oof_j))),
        "judge_cv_r2": float(r2_score(y, oof_j)),
        "hardness_cv_r2_log_resid_sq": float(r2_score(y_h, oof_h)),
        "hardness_spearman_pred_vs_resid_sq": float(
            pd.Series(oof_h).corr(pd.Series(oof_r2), method="spearman")
        ),
        "hardness_top_decile_lift": lift,
    }])
    rows = pd.DataFrame({
        "id": df["id"].values, "y_true": y,
        "judge_pred_oof": oof_j, "residual_sq_oof": oof_r2,
        "hardness_pred_oof": oof_h, "hardness_true_oof": y_h,
    })
    out.to_csv(out_dir / f"mrr_mqm_hardness_metrics_{tag}.csv", index=False)
    rows.to_csv(out_dir / f"mrr_mqm_hardness_oof_{tag}.csv", index=False)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Run all experiments for one system (or pooled)
# ─────────────────────────────────────────────────────────────────────────

def run_all_for_system(
    human_df: pd.DataFrame,
    llm_agg: pd.DataFrame,
    emb_df: pd.DataFrame | None,
    system_filter: str | None,
    folds: int, seed: int,
    judge_emb_pca: int, hardness_emb_pca: int,
    out_dir: Path,
    var_transform: str = "log",
) -> dict:
    tag = system_filter.replace(".", "_") if system_filter else "pooled"
    label = system_filter or "POOLED"
    print(f"\n{'='*60}")
    print(f"  System: {label}")
    print(f"{'='*60}")

    # full feature table
    full_df, full_feats = build_feature_table(
        human_df, llm_agg, emb_df, use_uncertainty=True,
        system_filter=system_filter, var_transform=var_transform,
    )
    if len(full_df) < 20:
        print(f"  Skipping {label}: only {len(full_df)} rows")
        return {}

    print(f"  rows={len(full_df)}  features(raw)={len(full_feats)}")

    # 1. Full judge
    m_full = run_judge(full_df, full_feats, folds, seed, judge_emb_pca, out_dir, f"{tag}_full")
    print(f"  [full]       R2={m_full['cv_r2'].iloc[0]:.4f}  RMSE={m_full['cv_rmse'].iloc[0]:.4f}  Spearman={m_full['cv_spearman'].iloc[0]:.4f}")

    # 2. LLM scores + uncertainty only (no embeddings)
    llm_df, llm_feats = build_feature_table(
        human_df, llm_agg, None, use_uncertainty=True,
        system_filter=system_filter, var_transform=var_transform,
    )
    m_llm = run_judge(llm_df, llm_feats, folds, seed, None, out_dir, f"{tag}_llm_only")
    print(f"  [llm_only]   R2={m_llm['cv_r2'].iloc[0]:.4f}  RMSE={m_llm['cv_rmse'].iloc[0]:.4f}  Spearman={m_llm['cv_spearman'].iloc[0]:.4f}")

    # 3. Embeddings only
    if emb_df is not None:
        emb_only_df, _ = build_feature_table(
            human_df, llm_agg, emb_df, use_uncertainty=False, system_filter=system_filter,
        )
        emb_feats = [c for c in emb_only_df.columns if c.startswith("emb__")]
        m_emb = run_judge(emb_only_df, emb_feats, folds, seed, judge_emb_pca, out_dir, f"{tag}_emb_only")
        print(f"  [emb_only]   R2={m_emb['cv_r2'].iloc[0]:.4f}  RMSE={m_emb['cv_rmse'].iloc[0]:.4f}  Spearman={m_emb['cv_spearman'].iloc[0]:.4f}")
    else:
        m_emb = None

    # 4. Intercept-only baseline
    m_int = run_intercept_only(full_df, folds, seed, out_dir, f"{tag}_intercept")
    print(f"  [intercept]  R2={m_int['cv_r2'].iloc[0]:.4f}  RMSE={m_int['cv_rmse'].iloc[0]:.4f}  Spearman={m_int['cv_spearman'].iloc[0]:.4f}")

    # 5. Two-stage hardness
    h_feats = [c for c in full_df.columns if c.startswith("emb__")]
    if h_feats:
        m_hard = run_two_stage_hardness(
            full_df, full_feats, h_feats, folds, seed,
            judge_emb_pca, hardness_emb_pca, out_dir, tag,
        )
        print(f"  [hardness]   R2_log_resid={m_hard['hardness_cv_r2_log_resid_sq'].iloc[0]:.4f}  "
              f"Spearman={m_hard['hardness_spearman_pred_vs_resid_sq'].iloc[0]:.4f}  "
              f"lift_q90={m_hard['hardness_top_decile_lift'].iloc[0]:.2f}")
    else:
        m_hard = None

    return {
        "system": label,
        "n": len(full_df),
        "full_r2": float(m_full["cv_r2"].iloc[0]),
        "full_rmse": float(m_full["cv_rmse"].iloc[0]),
        "full_spearman": float(m_full["cv_spearman"].iloc[0]),
        "llm_only_r2": float(m_llm["cv_r2"].iloc[0]),
        "llm_only_spearman": float(m_llm["cv_spearman"].iloc[0]),
        "emb_only_r2": float(m_emb["cv_r2"].iloc[0]) if m_emb is not None else np.nan,
        "emb_only_spearman": float(m_emb["cv_spearman"].iloc[0]) if m_emb is not None else np.nan,
        "intercept_r2": float(m_int["cv_r2"].iloc[0]),
        "hardness_r2": float(m_hard["hardness_cv_r2_log_resid_sq"].iloc[0]) if m_hard is not None else np.nan,
        "hardness_lift": float(m_hard["hardness_top_decile_lift"].iloc[0]) if m_hard is not None else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MQM MRR + hardness.")
    parser.add_argument("--tsv-path", type=Path,
                        default=Path("datasets/mqm/data/mqm_newstest2020_ende.tsv"))
    parser.add_argument("--llm-dir", type=Path,
                        default=Path("datasets/mqm/data"))
    parser.add_argument("--emb-path", type=Path,
                        default=Path("datasets/mqm/results/text_embedding_3_large_mqm_embeddings_separate.npz"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("datasets/mqm/results"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-emb-pca", type=int, default=64)
    parser.add_argument("--hardness-emb-pca", type=int, default=64)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional LLM model subset")
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Optional MT system subset (default: all + pooled)")
    parser.add_argument("--var-transform", choices=["log", "sqrt"], default="log",
                        help="Variance encoding: log (original) or sqrt")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading human MQM...")
    human = load_human_mqm(args.tsv_path.resolve())
    print(f"  {len(human)} (system, seg_id) entries")

    print("Loading LLM MQM...")
    llm_agg = load_llm_mqm(args.llm_dir.resolve(), models=args.models)
    print(f"  {len(llm_agg)} (model, system, seg_id) entries")
    print(f"  LLM models: {sorted(llm_agg['model'].unique())}")

    print("Loading embeddings...")
    emb = load_embeddings(args.emb_path.resolve())
    print(f"  {len(emb)} task embeddings, dim={emb.shape[1] - 1}")

    systems = args.systems or sorted(human["system"].unique())

    all_results = []

    # per-system models
    for sys in systems:
        res = run_all_for_system(
            human, llm_agg, emb, system_filter=sys,
            folds=args.cv_folds, seed=args.seed,
            judge_emb_pca=args.judge_emb_pca,
            hardness_emb_pca=args.hardness_emb_pca,
            out_dir=out_dir,
            var_transform=args.var_transform,
        )
        if res:
            all_results.append(res)

    # pooled model (all systems together)
    res_pooled = run_all_for_system(
        human, llm_agg, emb, system_filter=None,
        folds=args.cv_folds, seed=args.seed,
        judge_emb_pca=args.judge_emb_pca,
        hardness_emb_pca=args.hardness_emb_pca,
        out_dir=out_dir,
        var_transform=args.var_transform,
    )
    if res_pooled:
        all_results.append(res_pooled)

    # summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out_dir / "mrr_mqm_summary.csv", index=False)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out_dir / 'mrr_mqm_summary.csv'}")

    (out_dir / "mrr_mqm_summary.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
