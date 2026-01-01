"""
Uniform-sampling DR simulation for WebDesign (Universities) data.

Four outcome models, six uniform sampling budgets eta in {0.05,...,0.30}.
All six outcomes (AE, TRU, TYP, EXMPL, AVG, US) share the same sampled
item set per trial — i.e., 200 trials × 6 budgets, each trial evaluating
all 6 outcomes under the same uniform sample.

Models:
  1. intercept   – sample mean (DR = sample mean, intercept-only)
  2. ridge_full  – Ridge: VLM scores + uncertainty + SigLIP PCA
  3. ridge_vlm   – Ridge: VLM scores + uncertainty, no SigLIP
  4. ridge_emb   – Ridge: SigLIP PCA only

For each (outcome, model, budget, trial):
  - Fit outcome model on uniformly sampled items (shared mask across outcomes)
  - DR estimate: mean_i[ f_hat_i + S_i*(Y_i - f_hat_i) / eta ]
  - Record DR error = estimate - true_mean
  - Record held-out outcome model metrics: R², RMSE, Spearman

Output (per outcome, two figures each):
  - Figure 1: DR estimator bias / SD / RMSE vs eta  (4 curves ± 1 SE band)
  - Figure 2: outcome model R² / RMSE / Spearman vs eta (3 curves ± 1 SE band)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_FOLDERS = ["Universitites", "Universities"]  # handle original typo
RATINGS_FILE   = "ratings.avg.unis.txt"
VLM_FILE       = "vlm_scores_unis.jsonl"
SIGLIP_FILE    = "siglip_google_siglip_base_patch16_224_embeddings.npz"

OUTCOMES = ["AE", "TRU", "TYP", "EXMPL", "AVG", "US"]

DEFAULT_BUDGETS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_TRIALS   = 200
DEFAULT_SEED     = 42
DEFAULT_EMB_PCA  = 64

RIDGE_ALPHAS = np.logspace(-4, 4, 40)

MODEL_LABELS = {
    "intercept":  "Intercept-only (sample mean)",
    "ridge_full": "Ridge: scores + unc + SigLIP",
    "ridge_vlm":  "Ridge: scores + unc, no SigLIP",
    "ridge_emb":  "Ridge: SigLIP only",
}
MODEL_COLORS = {
    "intercept":  "#888888",
    "ridge_full": "#1f77b4",
    "ridge_vlm":  "#ff7f0e",
    "ridge_emb":  "#2ca02c",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — returns common-item features for all outcomes
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_domain_dir(base_dir: Path) -> Path:
    for name in DOMAIN_FOLDERS:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find domain folder under {base_dir}")


def load_data(
    base_dir: Path,
    emb_pca: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Return (X_emb, X_vlm_by_outcome, y_by_outcome, common_ids).

    All arrays are aligned to the same N common items (human ∩ VLM ∩ embeddings,
    no NaN in any of the six outcomes). X_emb is shared; X_vlm_by_outcome and
    y_by_outcome are dicts keyed by outcome name.
    """
    domain_dir = _resolve_domain_dir(base_dir)
    data_dir   = domain_dir / "data"
    res_dir    = domain_dir / "results"

    # Human averaged ratings
    human = pd.read_csv(data_dir / RATINGS_FILE, sep="\t")
    human["stimulusId"] = human["stimulusId"].astype(str)
    human = human[["stimulusId", *OUTCOMES]].dropna(subset=OUTCOMES)

    # VLM scores + uncertainties
    vlm = pd.read_json(res_dir / VLM_FILE, lines=True)
    if "stimulus_id" in vlm.columns:
        vlm = vlm.rename(columns={"stimulus_id": "stimulusId"})
    vlm["stimulusId"] = vlm["stimulusId"].astype(str)
    model_names = sorted(vlm["model"].dropna().unique().tolist())

    # Pivot: shared uncertainty, per-outcome scores
    piv_e = (
        vlm.pivot_table(index="stimulusId", columns="model", values="mean_entropy", aggfunc="mean")
        .rename(columns=lambda m: f"e__{m}").reset_index()
    )
    piv_p = (
        vlm.pivot_table(index="stimulusId", columns="model", values="perplexity", aggfunc="mean")
        .rename(columns=lambda m: f"p__{m}").reset_index()
    )

    # SigLIP embeddings
    arr     = np.load(res_dir / SIGLIP_FILE, allow_pickle=True)
    emb_ids = [Path(str(p)).name for p in arr["paths"]]
    emb_mat = arr["embeddings"].astype(np.float32)
    emb_map: dict[str, int] = {}
    for i, eid in enumerate(emb_ids):
        if eid not in emb_map:
            emb_map[eid] = i

    # Build base table: human ∩ vlm_uncertainty ∩ embeddings
    base = (
        human
        .merge(piv_e, on="stimulusId", how="inner")
        .merge(piv_p, on="stimulusId", how="inner")
    )
    base = base[base["stimulusId"].map(lambda x: x in emb_map)].reset_index(drop=True)

    # Per-outcome VLM score pivots — only keep items that also have all outcome scores
    for outcome in OUTCOMES:
        piv_s = (
            vlm.pivot_table(index="stimulusId", columns="model", values=f"score_{outcome}", aggfunc="mean")
            .rename(columns=lambda m: f"s_{outcome}__{m}").reset_index()
        )
        base = base.merge(piv_s, on="stimulusId", how="inner")

    base = base.reset_index(drop=True)
    N = len(base)
    print(f"  Common item set: N={N}")

    # ── SigLIP PCA (shared, unsupervised) ───────────────────────────────────
    E_raw = emb_mat[[emb_map[s] for s in base["stimulusId"]]].astype(np.float64)
    n_pca = min(emb_pca, E_raw.shape[1], N - 1)
    pca   = PCA(n_components=n_pca, random_state=seed)
    E_pca = pca.fit_transform(StandardScaler().fit_transform(E_raw))
    X_emb = StandardScaler().fit_transform(E_pca)

    # ── Per-outcome VLM features and labels ─────────────────────────────────
    X_vlm_by_outcome: dict[str, np.ndarray] = {}
    y_by_outcome: dict[str, np.ndarray]     = {}

    for outcome in OUTCOMES:
        vlm_cols: list[str] = []
        for m in model_names:
            s_col  = f"s_{outcome}__{m}"
            e_col  = f"e__{m}"
            p_col  = f"p__{m}"
            if s_col not in base.columns:
                continue
            s  = base[s_col].fillna(0.0).values
            e  = base[e_col].fillna(0.0).values if e_col in base.columns else np.zeros(N)
            p  = base[p_col].fillna(1.0).values if p_col in base.columns else np.ones(N)
            lp = np.log(np.clip(p, 1e-8, None))
            sxe_col  = f"sxe_{outcome}__{m}"
            sxlp_col = f"sxlp_{outcome}__{m}"
            base[sxe_col]  = s * e
            base[sxlp_col] = s * lp
            vlm_cols.extend([s_col, sxe_col, sxlp_col])

        X_vlm_raw = base[vlm_cols].values.astype(np.float64)
        X_vlm_raw = np.where(np.isfinite(X_vlm_raw), X_vlm_raw, 0.0)
        X_vlm_by_outcome[outcome] = StandardScaler().fit_transform(X_vlm_raw)
        y_by_outcome[outcome]     = base[outcome].astype(np.float64).values
        print(
            f"  [{outcome}] X_vlm={X_vlm_by_outcome[outcome].shape}  "
            f"y mean={y_by_outcome[outcome].mean():.3f}  std={y_by_outcome[outcome].std():.3f}"
        )

    return X_emb, X_vlm_by_outcome, y_by_outcome, base["stimulusId"].values


# ─────────────────────────────────────────────────────────────────────────────
# Ridge pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _make_ridge() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ridge",   RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Single trial — shared sampled mask, all outcomes
# ─────────────────────────────────────────────────────────────────────────────

def _record_model(
    key: str,
    f_hat_all: np.ndarray,   # (N,)  predictions on ALL items
    f_hat_oof: np.ndarray,   # (n_sampled,)  cross-fitted predictions on sampled items
    y: np.ndarray,
    sampled: np.ndarray,
    row: dict,
) -> None:
    """Hajék DR + held-out metrics. Residuals use OOF predictions."""
    dr = float(np.mean(f_hat_all) + np.mean(y[sampled] - f_hat_oof))
    row[f"{key}_dr_error"] = dr - float(np.mean(y))

    heldout = ~sampled
    if heldout.sum() > 5:
        yh, fh = y[heldout], f_hat_all[heldout]
        row[f"{key}_r2"]   = float(r2_score(yh, fh))
        row[f"{key}_rmse"] = float(np.sqrt(mean_squared_error(yh, fh)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = spearmanr(yh, fh).statistic
        row[f"{key}_spearman"] = float(sp) if np.isfinite(sp) else np.nan
    else:
        row[f"{key}_r2"] = row[f"{key}_rmse"] = row[f"{key}_spearman"] = np.nan


def _fit_ridge_crossfit(
    X_s: np.ndarray,
    y_s: np.ndarray,
    X_all: np.ndarray,
    fallback: float,
    seed: int = 0,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f_hat_all, f_hat_oof).

    f_hat_all – fit on all sampled, predict on all N (for mean term)
    f_hat_oof – k-fold cross-fitted predictions on sampled items (for residuals)

    5-fold gives each OOF model 80% of sampled items as training data,
    yielding near-unbiased residuals with much lower variance than 2-fold.
    """
    def _fit(X_tr, y_tr, X_pred):
        try:
            m = _make_ridge()
            m.fit(X_tr, y_tr)
            return m.predict(X_pred).astype(np.float64)
        except Exception:
            return np.full(len(X_pred), fallback)

    f_hat_all = _fit(X_s, y_s, X_all)

    n_s = len(y_s)
    f_hat_oof = np.full(n_s, fallback, dtype=np.float64)
    k = min(n_splits, n_s) if n_s >= 2 else 0
    if k >= 2:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X_s):
            f_hat_oof[te_idx] = _fit(X_s[tr_idx], y_s[tr_idx], X_s[te_idx])

    return f_hat_all, f_hat_oof


def run_trial(
    X_emb: np.ndarray,
    X_vlm_by_outcome: dict[str, np.ndarray],
    y_by_outcome: dict[str, np.ndarray],
    eta: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Run one trial (shared sampled mask) for all outcomes. Returns list of per-outcome rows."""
    N       = X_emb.shape[0]
    cf_seed = int(rng.integers(0, 1_000_000))

    sampled = rng.binomial(1, eta, N).astype(bool)
    if sampled.sum() < 10:
        extra = rng.choice(np.where(~sampled)[0], size=10 - sampled.sum(), replace=False)
        sampled[extra] = True

    n_s  = int(sampled.sum())
    rows = []
    for outcome in OUTCOMES:
        y      = y_by_outcome[outcome]
        X_vlm  = X_vlm_by_outcome[outcome]
        X_full = np.hstack([X_vlm, X_emb])
        fallback = float(np.mean(y[sampled]))

        row = {"outcome": outcome, "n_sampled": n_s}

        # 1. Intercept-only: Hajék with constant f_hat = sample_mean
        _record_model("intercept",
                      np.full(N, fallback), np.full(n_s, fallback),
                      y, sampled, row)

        # 2-4. Ridge variants with 5-fold cross-fitting
        f_all, f_oof = _fit_ridge_crossfit(X_full[sampled], y[sampled], X_full, fallback, cf_seed)
        _record_model("ridge_full", f_all, f_oof, y, sampled, row)

        f_all, f_oof = _fit_ridge_crossfit(X_vlm[sampled], y[sampled], X_vlm, fallback, cf_seed)
        _record_model("ridge_vlm", f_all, f_oof, y, sampled, row)

        f_all, f_oof = _fit_ridge_crossfit(X_emb[sampled], y[sampled], X_emb, fallback, cf_seed)
        _record_model("ridge_emb", f_all, f_oof, y, sampled, row)

        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results_df: pd.DataFrame, budgets: list[float]) -> dict:
    methods = list(MODEL_LABELS.keys())
    agg: dict[str, dict] = {m: {} for m in methods}

    for method in methods:
        for eta in budgets:
            sub    = results_df[results_df["eta"] == eta]
            errors = sub[f"{method}_dr_error"].dropna().values
            n      = len(errors)
            bias   = float(np.mean(errors))
            sd     = float(np.std(errors, ddof=1))
            rmse   = float(np.sqrt(np.mean(errors ** 2)))

            r2_v = sub[f"{method}_r2"].dropna().values
            rm_v = sub[f"{method}_rmse"].dropna().values
            sp_v = sub[f"{method}_spearman"].dropna().values

            def _se(v):
                return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

            agg[method][eta] = {
                "bias":        bias,
                "bias_se":     float(sd / np.sqrt(n)),
                "sd":          sd,
                "sd_se":       float(sd / np.sqrt(2 * max(n - 1, 1))),
                "dr_rmse":     rmse,
                "dr_rmse_se":  float(np.std(errors ** 2) / (2 * max(rmse, 1e-12) * np.sqrt(n))),
                "r2":          float(np.mean(r2_v)) if len(r2_v) else np.nan,
                "r2_se":       _se(r2_v),
                "out_rmse":    float(np.mean(rm_v)) if len(rm_v) else np.nan,
                "out_rmse_se": _se(rm_v),
                "spearman":    float(np.mean(sp_v)) if len(sp_v) else np.nan,
                "spearman_se": _se(sp_v),
            }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")


def make_plots(
    results_df: pd.DataFrame,
    budgets: list[float],
    outcome: str,
    out_dir: Path,
    n_trials: int,
) -> None:
    agg     = aggregate(results_df, budgets)
    methods = list(MODEL_LABELS.keys())
    xs      = budgets

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Figure 1: DR estimator quality ──────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics1 = [
        ("bias",    "bias_se",    "Bias"),
        ("sd",      "sd_se",      "Standard Deviation"),
        ("dr_rmse", "dr_rmse_se", "RMSE"),
    ]
    for ax, (key, se_key, ylabel) in zip(axes, metrics1):
        for method in methods:
            ys  = np.array([agg[method][e][key]    for e in xs])
            ses = np.array([agg[method][e][se_key] for e in xs])
            c   = MODEL_COLORS[method]
            ax.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
        if key == "bias":
            ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)

    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig1.legend(handles, labels, loc="upper center", ncol=4,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig1.tight_layout()
    p1 = out_dir / f"uniform_dr_bias_sd_rmse_{outcome}.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p1.name}")

    # ── Figure 2: Outcome model quality on held-out items ───────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics2 = [
        ("r2",       "r2_se",        "R²"),
        ("out_rmse", "out_rmse_se",  "RMSE"),
        ("spearman", "spearman_se",  "Spearman ρ"),
    ]
    outcome_methods = [m for m in methods if m != "intercept"]
    for ax, (key, se_key, ylabel) in zip(axes, metrics2):
        for method in outcome_methods:
            ys  = np.array([agg[method][e][key]    for e in xs])
            ses = np.array([agg[method][e][se_key] for e in xs])
            c   = MODEL_COLORS[method]
            ax.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)

    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig2.legend(handles, labels, loc="upper center", ncol=3,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig2.tight_layout()
    p2 = out_dir / f"uniform_dr_outcome_r2_rmse_spearman_{outcome}.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform DR simulation on WebDesign Universities.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/WebDesign"))
    parser.add_argument("--trials",   type=int,   default=DEFAULT_TRIALS)
    parser.add_argument("--budgets",  nargs="+",  type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    parser.add_argument("--emb-pca",   type=int,   default=DEFAULT_EMB_PCA)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; reload saved CSV and regenerate plots.")
    args = parser.parse_args()

    domain_dir = _resolve_domain_dir(args.base_dir.resolve())
    out_dir    = domain_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        csv_path = out_dir / "uniform_dr_simulation_trials_unis.csv"
        print(f"Loading saved trials from {csv_path.name} ...")
        results_df = pd.read_csv(csv_path)
        budgets = sorted(results_df["eta"].unique().tolist())
        for outcome in OUTCOMES:
            odf = results_df[results_df["outcome"] == outcome].copy()
            make_plots(odf, budgets, outcome, out_dir, args.trials)
        return

    print("Loading data...")
    X_emb, X_vlm_by_outcome, y_by_outcome, _ = load_data(
        args.base_dir, emb_pca=args.emb_pca, seed=args.seed
    )

    rng         = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(1, 10_000_000, size=args.trials).tolist()

    # Collect all rows keyed by (outcome, eta)
    all_rows: list[dict] = []

    for eta in args.budgets:
        print(f"\nBudget η={eta:.2f}  ({args.trials} trials × {len(OUTCOMES)} outcomes)...")
        for t, s in enumerate(trial_seeds):
            trial_rng = np.random.default_rng(s)
            outcome_rows = run_trial(X_emb, X_vlm_by_outcome, y_by_outcome, eta, trial_rng)
            for row in outcome_rows:
                row["eta"] = eta
            all_rows.extend(outcome_rows)
            if (t + 1) % 50 == 0:
                print(f"  trial {t+1}/{args.trials}")

    results_df = pd.DataFrame(all_rows)
    csv_path   = out_dir / "uniform_dr_simulation_trials_unis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved trial data: {csv_path.name}")

    # Per-outcome summary + plots
    summary_rows: list[dict] = []
    for outcome in OUTCOMES:
        print(f"\n--- {outcome} ---")
        odf = results_df[results_df["outcome"] == outcome].copy()
        make_plots(odf, args.budgets, outcome, out_dir, args.trials)

        for eta in args.budgets:
            sub = odf[odf["eta"] == eta]
            for method in MODEL_LABELS:
                errors = sub[f"{method}_dr_error"].dropna().values
                summary_rows.append({
                    "outcome":  outcome,
                    "eta":      eta,
                    "model":    method,
                    "dr_bias":  float(np.mean(errors)),
                    "dr_sd":    float(np.std(errors, ddof=1)),
                    "dr_rmse":  float(np.sqrt(np.mean(errors ** 2))),
                    "r2":       float(sub[f"{method}_r2"].mean()),
                    "out_rmse": float(sub[f"{method}_rmse"].mean()),
                    "spearman": float(sub[f"{method}_spearman"].mean()),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "uniform_dr_simulation_summary_unis.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path.name}")
    print("\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
