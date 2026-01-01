"""
Multiple Ridge Regression (MRR) for WebDesign domains.

Supports:
- Judge model feature modes:
  1) scores
  2) scores_uncertainty
  3) scores_uncertainty_siglip_intercept
- Optional SigLIP-only baseline (y ~ embedding features)
- Optional two-stage nested crossfit hardness model (hardness uses SigLIP-only features)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTCOMES = ["AE", "TRU", "TYP", "EXMPL", "AVG", "US"]
ALPHAS = np.logspace(-4, 4, 40)

DOMAIN_CONFIG = {
    "unis": {
        "folder": "Universitites",
        "ratings_avg": "ratings.avg.unis.txt",
        "vlm_scores": "vlm_scores_unis.jsonl",
        "file_prefix": "mrr_unis",
    },
    "banks": {
        "folder": "Commercial Banks",
        "ratings_avg": "ratings.avg.banks.txt",
        "vlm_scores": "vlm_scores_banks.jsonl",
        "file_prefix": "mrr_banks",
    },
    "fashion": {
        "folder": "eCommerce",
        "ratings_avg": "ratings.avg.fashion.txt",
        "vlm_scores": "vlm_scores_fashion.jsonl",
        "file_prefix": "mrr_fashion",
    },
    "homeware": {
        "folder": "eCommerce",
        "ratings_avg": "ratings.avg.homeware.txt",
        "vlm_scores": "vlm_scores_homeware.jsonl",
        "file_prefix": "mrr_homeware",
    },
}


def make_ridge_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS, fit_intercept=True)),
        ]
    )


def crossfit_predict(X: pd.DataFrame, y: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    alphas = []
    for tr, te in kf.split(X):
        pipe = make_ridge_pipe()
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict(X.iloc[te])
        alphas.append(float(pipe.named_steps["ridge"].alpha_))
    return oof, np.asarray(alphas, dtype=float)


def load_human_avg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    req = {"stimulusId", *OUTCOMES}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in human ratings: {sorted(missing)}")
    return df[["stimulusId", *OUTCOMES]].copy()


def load_vlm_scores(path: Path, models: list[str] | None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if models:
        df = df[df["model"].isin(models)].copy()
    if "stimulus_id" in df.columns:
        df = df.rename(columns={"stimulus_id": "stimulusId"})
    req = {"stimulusId", "model", "mean_entropy", "perplexity", *[f"score_{o}" for o in OUTCOMES]}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in VLM scores: {sorted(missing)}")
    return df.reset_index(drop=True)


def load_siglip_embeddings(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    paths = arr["paths"]
    emb = arr["embeddings"]
    stimulus = [Path(str(p)).name for p in paths]
    cols = [f"siglip__{i}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=cols)
    out.insert(0, "stimulusId", stimulus)
    out = out.drop_duplicates("stimulusId")
    return out


def build_outcome_feature_table(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    outcome: str,
    use_uncertainty: bool,
    siglip_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    score_col = f"score_{outcome}"
    pivot_score = llm_df.pivot_table(index="stimulusId", columns="model", values=score_col, aggfunc="mean")
    model_names = sorted(pivot_score.columns.tolist())
    pivot_score.columns = [f"s__{m}" for m in model_names]

    merged = human_df[["stimulusId", outcome]].rename(columns={outcome: "y"}).merge(
        pivot_score.reset_index(), on="stimulusId", how="inner"
    )
    feature_cols = [f"s__{m}" for m in model_names]

    if use_uncertainty:
        pivot_entropy = llm_df.pivot_table(index="stimulusId", columns="model", values="mean_entropy", aggfunc="mean")
        pivot_ppl = llm_df.pivot_table(index="stimulusId", columns="model", values="perplexity", aggfunc="mean")
        pivot_entropy.columns = [f"entropy__{m}" for m in pivot_entropy.columns]
        pivot_ppl.columns = [f"ppl__{m}" for m in pivot_ppl.columns]

        merged = merged.merge(pivot_entropy.reset_index(), on="stimulusId", how="left")
        merged = merged.merge(pivot_ppl.reset_index(), on="stimulusId", how="left")
        for m in model_names:
            s_col = f"s__{m}"
            e_col = f"entropy__{m}"
            p_col = f"ppl__{m}"
            lp_col = f"logppl__{m}"
            sxe_col = f"sxe__{m}"
            sxlp_col = f"sxlp__{m}"
            merged[lp_col] = np.log(merged[p_col].clip(lower=1e-8))
            merged[sxe_col] = merged[s_col] * merged[e_col]
            merged[sxlp_col] = merged[s_col] * merged[lp_col]
            feature_cols.extend([sxe_col, sxlp_col])

    if siglip_df is not None:
        merged = merged.merge(siglip_df, on="stimulusId", how="left")
        siglip_cols = [c for c in merged.columns if c.startswith("siglip__")]
        feature_cols.extend(siglip_cols)

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged, feature_cols


def fit_one_outcome(
    df: pd.DataFrame,
    outcome: str,
    feature_cols: list[str],
    folds: int,
    seed: int,
    experiment: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    keep = df.dropna(subset=["y"]).copy()
    X = keep[feature_cols]
    y = keep["y"].astype(float).to_numpy()
    oof, alpha_folds = crossfit_predict(X, y, folds=folds, seed=seed)

    final_pipe = make_ridge_pipe()
    final_pipe.fit(X, y)
    ridge = final_pipe.named_steps["ridge"]

    metrics = {
        "outcome": outcome,
        "n_samples": int(len(keep)),
        "n_features": int(len(feature_cols)),
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "cv_r2": float(r2_score(y, oof)),
        "cv_spearman": float(pd.Series(y).corr(pd.Series(oof), method="spearman")),
        "mean_alpha_cv_folds": float(np.mean(alpha_folds)),
        "std_alpha_cv_folds": float(np.std(alpha_folds)),
        "final_alpha_full_fit": float(ridge.alpha_),
        "intercept": float(ridge.intercept_),
        "experiment": experiment,
    }

    coef_df = pd.DataFrame(
        {
            "outcome": outcome,
            "feature": feature_cols,
            "weight": ridge.coef_.ravel().astype(float),
            "experiment": experiment,
        }
    )
    oof_df = pd.DataFrame(
        {
            "stimulusId": keep["stimulusId"].to_numpy(),
            "outcome": outcome,
            "y_true": y,
            "y_pred_oof": oof,
            "residual_oof": y - oof,
            "experiment": experiment,
        }
    )
    return metrics, coef_df, oof_df


def run_judge_experiment(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    siglip_df: pd.DataFrame | None,
    outcomes: list[str],
    folds: int,
    seed: int,
    use_uncertainty: bool,
    experiment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    coef_rows = []
    oof_rows = []
    for outcome in outcomes:
        table, feature_cols = build_outcome_feature_table(
            human_df=human_df,
            llm_df=llm_df,
            outcome=outcome,
            use_uncertainty=use_uncertainty,
            siglip_df=siglip_df,
        )
        m, c, o = fit_one_outcome(
            df=table,
            outcome=outcome,
            feature_cols=feature_cols,
            folds=folds,
            seed=seed,
            experiment=experiment,
        )
        metrics_rows.append(m)
        coef_rows.append(c)
        oof_rows.append(o)
    return (
        pd.DataFrame(metrics_rows),
        pd.concat(coef_rows, ignore_index=True),
        pd.concat(oof_rows, ignore_index=True),
    )


def run_siglip_only_baseline(
    human_df: pd.DataFrame,
    siglip_df: pd.DataFrame,
    outcomes: list[str],
    folds: int,
    seed: int,
    pca_components: int,
) -> pd.DataFrame:
    rows = []
    sig_cols = [c for c in siglip_df.columns if c.startswith("siglip__")]
    for outcome in outcomes:
        merged = human_df[["stimulusId", outcome]].rename(columns={outcome: "y"}).merge(
            siglip_df, on="stimulusId", how="inner"
        )
        merged = merged.dropna(subset=["y"]).copy()
        X_raw = merged[sig_cols].to_numpy(dtype=float)
        y = merged["y"].astype(float).to_numpy()

        pca_n = min(pca_components, X_raw.shape[1], X_raw.shape[0] - 1)
        pre = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=seed)),
            ]
        )
        X = pre.fit_transform(X_raw)
        X_df = pd.DataFrame(X)
        oof, _ = crossfit_predict(X_df, y, folds=folds, seed=seed)

        rows.append(
            {
                "outcome": outcome,
                "n_samples": int(len(merged)),
                "n_features": int(X.shape[1]),
                "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
                "cv_r2": float(r2_score(y, oof)),
                "cv_spearman": float(pd.Series(y).corr(pd.Series(oof), method="spearman")),
                "experiment": "siglip_only",
            }
        )
    return pd.DataFrame(rows)


def run_intercept_only_baseline(
    human_df: pd.DataFrame,
    outcomes: list[str],
    folds: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for outcome in outcomes:
        merged = human_df[["stimulusId", outcome]].rename(columns={outcome: "y"}).dropna(subset=["y"]).copy()
        y = merged["y"].astype(float).to_numpy()
        oof = np.zeros_like(y, dtype=float)
        for tr, te in kf.split(merged):
            oof[te] = float(np.mean(y[tr]))

        rows.append(
            {
                "outcome": outcome,
                "n_samples": int(len(merged)),
                "n_features": 0,
                "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
                "cv_r2": float(r2_score(y, oof)),
                "cv_spearman": float(pd.Series(y).corr(pd.Series(oof), method="spearman")),
                "experiment": "intercept_only",
            }
        )
    return pd.DataFrame(rows)


def run_raw_avg_baseline(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    """Baseline: average all VLM judges' raw scores per stimulus per outcome, compare to human.
    No calibration or model fitting."""
    rows = []
    for outcome in outcomes:
        score_col = f"score_{outcome}"
        pivot = llm_df.pivot_table(index="stimulusId", columns="model", values=score_col, aggfunc="mean")
        raw_avg = pivot.mean(axis=1, skipna=True).reset_index().rename(columns={0: "raw_avg"})
        merged = human_df[["stimulusId", outcome]].rename(columns={outcome: "y"}).merge(
            raw_avg, on="stimulusId", how="inner"
        ).dropna(subset=["y", "raw_avg"])
        y   = merged["y"].astype(float).values
        yp  = merged["raw_avg"].astype(float).values
        mask = np.isfinite(yp)
        y, yp = y[mask], yp[mask]
        if len(y) < 5:
            continue
        rows.append({
            "outcome":    outcome,
            "n_samples":  int(len(y)),
            "n_features": int(pivot.shape[1]),
            "cv_rmse":    float(np.sqrt(mean_squared_error(y, yp))),
            "cv_r2":      float(r2_score(y, yp)),
            "cv_spearman": float(pd.Series(y).corr(pd.Series(yp), method="spearman")),
            "experiment": "raw_avg",
        })
    return pd.DataFrame(rows)


def run_raw_single_judge_avg(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    """Baseline: per-judge raw metrics per outcome averaged across all judges.
    No calibration or model fitting."""
    model_names = sorted(llm_df["model"].dropna().unique().tolist())
    per_model_rows = []
    for model_name in model_names:
        sub = llm_df[llm_df["model"] == model_name]
        for outcome in outcomes:
            score_col = f"score_{outcome}"
            s = sub[["stimulusId", score_col]].drop_duplicates("stimulusId")
            merged = human_df[["stimulusId", outcome]].rename(columns={outcome: "y"}).merge(
                s, on="stimulusId", how="inner"
            ).dropna(subset=["y", score_col])
            y   = merged["y"].astype(float).values
            yp  = merged[score_col].astype(float).values
            mask = np.isfinite(yp)
            y, yp = y[mask], yp[mask]
            if len(y) < 5:
                continue
            per_model_rows.append({
                "model":   model_name,
                "outcome": outcome,
                "r2":      float(r2_score(y, yp)),
                "rmse":    float(np.sqrt(mean_squared_error(y, yp))),
                "sp":      float(pd.Series(y).corr(pd.Series(yp), method="spearman")),
            })

    per_df = pd.DataFrame(per_model_rows)
    # Average first over models per outcome, then over outcomes
    outcome_avg = per_df.groupby("outcome")[["r2", "rmse", "sp"]].mean().reset_index()
    rows = []
    for _, row in outcome_avg.iterrows():
        rows.append({
            "outcome":    row["outcome"],
            "n_samples":  0,
            "n_features": 1,
            "cv_rmse":    float(row["rmse"]),
            "cv_r2":      float(row["r2"]),
            "cv_spearman": float(row["sp"]),
            "experiment": "raw_single_judge_avg",
        })
    return pd.DataFrame(rows)


def make_hardness_pipe(seed: int, pca_components: int | None) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))
    steps.append(("ridge", RidgeCV(alphas=ALPHAS, fit_intercept=True)))
    return Pipeline(steps=steps)


def run_two_stage_crossfit(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    siglip_df: pd.DataFrame,
    outcomes: list[str],
    folds: int,
    seed: int,
    use_uncertainty: bool,
    use_siglip_intercept: bool,
    hardness_siglip_pca_components: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics_rows = []
    oof_rows = []
    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    summary = {}

    for outcome in outcomes:
        judge_df, judge_features = build_outcome_feature_table(
            human_df=human_df,
            llm_df=llm_df,
            outcome=outcome,
            use_uncertainty=use_uncertainty,
            siglip_df=siglip_df if use_siglip_intercept else None,
        )
        hard_df, hard_features = build_outcome_feature_table(
            human_df=human_df,
            llm_df=llm_df,
            outcome=outcome,
            use_uncertainty=False,
            siglip_df=siglip_df,
        )
        hard_sig_cols = [c for c in hard_features if c.startswith("siglip__")]

        merged = judge_df.merge(
            hard_df[["stimulusId", *hard_sig_cols]].drop_duplicates("stimulusId"),
            on="stimulusId",
            how="inner",
            suffixes=("", "_h"),
        )
        merged = merged.dropna(subset=["y"]).reset_index(drop=True)
        y = merged["y"].to_numpy(dtype=float)
        Xj = merged[judge_features]
        Xh = merged[hard_sig_cols]

        oof_judge = np.zeros_like(y, dtype=float)
        oof_r2 = np.zeros_like(y, dtype=float)
        oof_hard = np.zeros_like(y, dtype=float)

        for tr_idx, te_idx in outer.split(Xj):
            Xj_tr, Xj_te = Xj.iloc[tr_idx], Xj.iloc[te_idx]
            Xh_tr, Xh_te = Xh.iloc[tr_idx], Xh.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            tr_judge_oof, _ = crossfit_predict(Xj_tr, y_tr, folds=folds, seed=seed)
            tr_r2 = np.square(y_tr - tr_judge_oof)
            tr_log_r2 = np.log(tr_r2 + 1e-6)

            hard_pipe = make_hardness_pipe(seed=seed, pca_components=hardness_siglip_pca_components)
            hard_pipe.fit(Xh_tr, tr_log_r2)

            judge_pipe = make_ridge_pipe()
            judge_pipe.fit(Xj_tr, y_tr)
            te_j = judge_pipe.predict(Xj_te)
            te_r2 = np.square(y_te - te_j)
            te_h = hard_pipe.predict(Xh_te)

            oof_judge[te_idx] = te_j
            oof_r2[te_idx] = te_r2
            oof_hard[te_idx] = te_h

        true_log_r2 = np.log(oof_r2 + 1e-6)
        hard_rmse = float(np.sqrt(mean_squared_error(true_log_r2, oof_hard)))
        hard_r2 = float(r2_score(true_log_r2, oof_hard))
        hard_spearman = float(pd.Series(oof_hard).corr(pd.Series(oof_r2), method="spearman"))
        q90 = float(np.quantile(oof_hard, 0.9))
        lift = float(np.mean(oof_r2[oof_hard >= q90]) / np.mean(oof_r2))

        metrics_rows.append(
            {
                "outcome": outcome,
                "n_samples": int(len(merged)),
                "judge_cv_rmse_outer": float(np.sqrt(mean_squared_error(y, oof_judge))),
                "judge_cv_r2_outer": float(r2_score(y, oof_judge)),
                "hardness_cv_rmse_log_resid_sq": hard_rmse,
                "hardness_cv_r2_log_resid_sq": hard_r2,
                "hardness_spearman_pred_vs_resid_sq": hard_spearman,
                "hardness_top_decile_resid_sq_lift": lift,
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    "stimulusId": merged["stimulusId"].to_numpy(),
                    "outcome": outcome,
                    "y_true": y,
                    "judge_pred_oof_outer": oof_judge,
                    "residual_sq_oof_outer": oof_r2,
                    "hardness_log_resid_sq_pred_oof_outer": oof_hard,
                    "hardness_log_resid_sq_true_oof_outer": true_log_r2,
                }
            )
        )
        summary[outcome] = {"n": int(len(merged))}

    return pd.DataFrame(metrics_rows), pd.concat(oof_rows, ignore_index=True), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuilt WebDesign MRR pipeline.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/WebDesign"))
    parser.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()), default="unis")
    parser.add_argument("--feature-mode", choices=["scores", "scores_uncertainty", "scores_uncertainty_siglip_intercept"], default="scores_uncertainty_siglip_intercept")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-siglip-only", action="store_true")
    parser.add_argument("--siglip-only-pca-components", type=int, default=64)
    parser.add_argument("--run-two-stage-crossfit", action="store_true")
    parser.add_argument("--hardness-siglip-pca-components", type=int, default=64)
    args = parser.parse_args()

    cfg = DOMAIN_CONFIG[args.domain]
    domain_dir = args.base_dir.resolve() / cfg["folder"]
    data_dir = domain_dir / "data"
    results_dir = domain_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    human = load_human_avg(data_dir / cfg["ratings_avg"])
    llm = load_vlm_scores(results_dir / cfg["vlm_scores"], models=args.models)
    siglip = load_siglip_embeddings(results_dir / "siglip_google_siglip_base_patch16_224_embeddings.npz")

    use_uncertainty = args.feature_mode in {"scores_uncertainty", "scores_uncertainty_siglip_intercept"}
    use_siglip_intercept = args.feature_mode == "scores_uncertainty_siglip_intercept"
    exp = args.feature_mode
    prefix = cfg["file_prefix"]

    metrics_df, coef_df, oof_df = run_judge_experiment(
        human_df=human,
        llm_df=llm,
        siglip_df=siglip if use_siglip_intercept else None,
        outcomes=OUTCOMES,
        folds=args.cv_folds,
        seed=args.seed,
        use_uncertainty=use_uncertainty,
        experiment=exp,
    )

    metrics_path = results_dir / f"{prefix}_metrics_{exp}.csv"
    coef_path = results_dir / f"{prefix}_coefficients_{exp}.csv"
    oof_path = results_dir / f"{prefix}_oof_predictions_{exp}.csv"
    summary_path = results_dir / f"{prefix}_summary_{exp}.json"

    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    summary = {
        "experiment": exp,
        "models_used": sorted(llm["model"].dropna().unique().tolist()),
        "outcomes": OUTCOMES,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "avg_cv_rmse_across_outcomes": float(metrics_df["cv_rmse"].mean()),
        "avg_cv_r2_across_outcomes": float(metrics_df["cv_r2"].mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {metrics_path}")
    print(f"Saved: {coef_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {summary_path}")

    if args.run_siglip_only:
        sig_only = run_siglip_only_baseline(
            human_df=human,
            siglip_df=siglip,
            outcomes=OUTCOMES,
            folds=args.cv_folds,
            seed=args.seed,
            pca_components=args.siglip_only_pca_components,
        )
        sig_only_path = results_dir / f"{prefix}_metrics_siglip_only.csv"
        sig_only.to_csv(sig_only_path, index=False)
        print(f"Saved: {sig_only_path}")

    intercept_only = run_intercept_only_baseline(
        human_df=human,
        outcomes=OUTCOMES,
        folds=args.cv_folds,
        seed=args.seed,
    )
    intercept_only_path = results_dir / f"{prefix}_metrics_intercept_only.csv"
    intercept_only.to_csv(intercept_only_path, index=False)
    print(f"Saved: {intercept_only_path}")

    raw_avg = run_raw_avg_baseline(human_df=human, llm_df=llm, outcomes=OUTCOMES)
    raw_avg_path = results_dir / f"{prefix}_metrics_raw_avg.csv"
    raw_avg.to_csv(raw_avg_path, index=False)
    print(f"Saved: {raw_avg_path}")
    print(f"  Raw avg   – avg R²={raw_avg['cv_r2'].mean():.4f}  RMSE={raw_avg['cv_rmse'].mean():.4f}  Sp={raw_avg['cv_spearman'].mean():.4f}")

    raw_single = run_raw_single_judge_avg(human_df=human, llm_df=llm, outcomes=OUTCOMES)
    raw_single_path = results_dir / f"{prefix}_metrics_raw_single_judge_avg.csv"
    raw_single.to_csv(raw_single_path, index=False)
    print(f"Saved: {raw_single_path}")
    print(f"  Raw single– avg R²={raw_single['cv_r2'].mean():.4f}  RMSE={raw_single['cv_rmse'].mean():.4f}  Sp={raw_single['cv_spearman'].mean():.4f}")

    if args.run_two_stage_crossfit:
        two_metrics, two_rows, _ = run_two_stage_crossfit(
            human_df=human,
            llm_df=llm,
            siglip_df=siglip,
            outcomes=OUTCOMES,
            folds=args.cv_folds,
            seed=args.seed,
            use_uncertainty=use_uncertainty,
            use_siglip_intercept=use_siglip_intercept,
            hardness_siglip_pca_components=args.hardness_siglip_pca_components,
        )
        two_metrics_path = results_dir / f"{prefix}_two_stage_hardness_metrics.csv"
        two_rows_path = results_dir / f"{prefix}_two_stage_hardness_oof_rows.csv"
        two_summary_path = results_dir / f"{prefix}_two_stage_hardness_summary.json"
        two_metrics.to_csv(two_metrics_path, index=False)
        two_rows.to_csv(two_rows_path, index=False)
        two_summary = {
            "domain": args.domain,
            "feature_mode": exp,
            "hardness_siglip_pca_components": args.hardness_siglip_pca_components,
            "avg_judge_cv_r2_outer": float(two_metrics["judge_cv_r2_outer"].mean()),
            "avg_hardness_cv_r2_log_resid_sq": float(two_metrics["hardness_cv_r2_log_resid_sq"].mean()),
        }
        two_summary_path.write_text(json.dumps(two_summary, indent=2))
        print(f"Saved: {two_metrics_path}")
        print(f"Saved: {two_rows_path}")
        print(f"Saved: {two_summary_path}")


if __name__ == "__main__":
    main()
