"""
Uniform-sampling DR simulation for PERSUADE grade-10 essays.

Four outcome models, six uniform sampling budgets eta in {0.05,...,0.30}.

Models:
  1. intercept      – sample mean (DR = sample mean, intercept-only)
  2. ordinal_full   – Ordinal (prop. odds): scores + uncertainty + embedding PCA
  3. ordinal_no_emb – Ordinal (prop. odds): scores + uncertainty, no embeddings
  4. ordinal_emb    – Ordinal (prop. odds): embedding PCA only

For each (model, budget, trial):
  - Fit outcome model on uniformly sampled items
  - DR estimate: mean_i[ f_hat_i + S_i*(Y_i - f_hat_i) / eta ]
  - Record DR error = estimate - true_mean
  - Record held-out outcome model metrics: R², RMSE, Spearman

Output: two figures (3 subplots each):
  - Figure 1: DR estimator bias / SD / RMSE  vs eta  (4 curves)
  - Figure 2: outcome model R² / RMSE / Spearman vs eta (3 curves, no intercept)
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
from mord import LogisticAT
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV  # noqa: F401 (kept for potential reuse)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DATA_CSV = Path("datasets/persuade/data/persuade_corpus_2.0_train.csv")
LLM_PATH = Path("datasets/persuade/results/llm_scores.jsonl")
EMB_PATH = Path("datasets/persuade/results/openai_text_embedding_3_large_essay_embeddings.npz")
OUT_DIR = Path("datasets/persuade/results")

GRADE = 10
DEFAULT_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_TRIALS = 100
DEFAULT_SEED = 42
DEFAULT_EMB_PCA = 64

RIDGE_ALPHAS = np.logspace(-4, 4, 40)
ORDINAL_ALPHAS = np.logspace(-4, 4, 20)

MODEL_LABELS = {
    "intercept":      "Intercept-only (sample mean)",
    "ordinal_full":   "Ordinal: scores + unc + emb",
    "ordinal_no_emb": "Ordinal: scores + unc, no emb",
    "ordinal_emb":    "Ordinal: emb only",
}
MODEL_COLORS = {
    "intercept":      "#888888",
    "ordinal_full":   "#1f77b4",
    "ordinal_no_emb": "#ff7f0e",
    "ordinal_emb":    "#2ca02c",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(emb_pca: int, seed: int):
    """Return y, X_full, X_llm, X_emb for grade-10 essays.

    PCA and scalers are fit on the full dataset — features are always fully
    observed (no label required), so this introduces no label leakage.
    Only outcome-model weights are fitted on sampled data within each trial.
    """
    # Human scores
    human = pd.read_csv(DATA_CSV, low_memory=False)
    human = human.drop_duplicates("essay_id_comp")[
        ["essay_id_comp", "holistic_essay_score", "grade_level"]
    ].copy()
    human = human[human["grade_level"] == GRADE].rename(
        columns={"essay_id_comp": "id", "holistic_essay_score": "score"}
    )
    human["id"] = human["id"].astype(str)

    # LLM scores + token uncertainties
    llm = pd.read_json(LLM_PATH, lines=True)
    llm = llm.rename(columns={"essay_id_comp": "id"})
    llm["id"] = llm["id"].astype(str)
    llm = llm[llm["id"].isin(human["id"])].copy()

    model_names = sorted(llm["model"].unique().tolist())

    piv_s = llm.pivot_table(index="id", columns="model", values="predicted_score",  aggfunc="first")
    piv_e = llm.pivot_table(index="id", columns="model", values="mean_entropy",     aggfunc="first")
    piv_p = llm.pivot_table(index="id", columns="model", values="perplexity",       aggfunc="first")

    llm_feats = pd.DataFrame(index=piv_s.index)
    for m in model_names:
        s  = piv_s[m]
        e  = piv_e[m]
        lp = np.log(piv_p[m].clip(lower=1e-8))
        llm_feats[f"s__{m}"]    = s
        llm_feats[f"sxe__{m}"]  = s * e
        llm_feats[f"sxlp__{m}"] = s * lp
    llm_feats = llm_feats.reset_index()

    # Embeddings
    arr      = np.load(EMB_PATH, allow_pickle=True)
    emb_ids  = [str(x) for x in arr["essay_ids"]]
    emb_mat  = arr["embeddings"].astype(np.float32)
    emb_map  = {eid: i for i, eid in enumerate(emb_ids)}

    # Merge: human ∩ LLM features ∩ embeddings
    base = human.merge(llm_feats, on="id", how="inner")
    base["id"] = base["id"].astype(str)
    has_emb = base["id"].map(lambda i: i in emb_map)
    base = base[has_emb].reset_index(drop=True)

    E_raw = emb_mat[[emb_map[i] for i in base["id"]]]  # (N, D)

    # PCA on embeddings (unsupervised – no label leakage)
    n_pca = min(emb_pca, E_raw.shape[1], len(base) - 1)
    pca   = PCA(n_components=n_pca, random_state=seed)
    E_pca = pca.fit_transform(E_raw)  # (N, n_pca)

    # Feature matrices
    llm_cols  = [c for c in base.columns if c.startswith(("s__", "sxe__", "sxlp__"))]
    X_llm_raw = np.where(np.isfinite(base[llm_cols].values), base[llm_cols].values, 0.0)

    sc_llm = StandardScaler()
    X_llm  = sc_llm.fit_transform(X_llm_raw).astype(np.float64)

    sc_emb = StandardScaler()
    X_emb  = sc_emb.fit_transform(E_pca).astype(np.float64)

    X_full = np.hstack([X_llm, X_emb])

    y = base["score"].astype(float).values
    return y, X_full, X_llm, X_emb


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal alpha pre-selection
# ─────────────────────────────────────────────────────────────────────────────

def select_ordinal_alpha(X: np.ndarray, y: np.ndarray, seed: int, label: str) -> float:
    """5-fold CV on full data to choose LogisticAT alpha (minimises MSE of E[Y])."""
    classes = np.sort(np.unique(y.astype(int))).astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_alpha, best_mse = ORDINAL_ALPHAS[0], np.inf
    for a in ORDINAL_ALPHAS:
        fold_mses = []
        for tr, te in kf.split(X):
            try:
                m = LogisticAT(alpha=a, max_iter=2000)
                m.fit(X[tr], y[tr].astype(int))
                ev = m.predict_proba(X[te]) @ classes
                fold_mses.append(float(mean_squared_error(y[te], ev)))
            except Exception:
                fold_mses.append(np.inf)
        mse = float(np.mean(fold_mses))
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    print(f"  [{label}] selected alpha={best_alpha:.3g}  (5-fold CV MSE={best_mse:.4f})")
    return best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _record_model(
    key: str,
    f_hat_all: np.ndarray,   # (N,)  predictions on ALL items
    f_hat_oof: np.ndarray,   # (n_sampled,)  cross-fitted predictions on sampled items
    y: np.ndarray,
    sampled: np.ndarray,
    row: dict,
) -> None:
    """Hajék DR + held-out outcome metrics; write into row dict.

    Hajék: mean_all(f_hat) + mean_sampled(Y - f_hat_OOF)
    Residuals use OOF predictions to avoid overfitting bias.
    """
    dr = float(np.mean(f_hat_all) + np.mean(y[sampled] - f_hat_oof))
    row[f"{key}_dr_error"] = dr - row["true_mean"]

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


def _fit_ordinal(X_tr, y_tr, X_all, alpha, fallback_mean):
    try:
        mdl = LogisticAT(alpha=alpha, max_iter=2000)
        mdl.fit(X_tr, y_tr)
        return mdl.predict_proba(X_all) @ mdl.classes_.astype(float)
    except Exception:
        return np.full(len(X_all), fallback_mean)


def _fit_ordinal_crossfit(
    X_s: np.ndarray,      # sampled features
    y_s: np.ndarray,      # sampled labels (float)
    X_all: np.ndarray,    # all N items
    alpha: float,
    fallback: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f_hat_all, f_hat_oof).

    f_hat_all  – fit on all sampled, predict on all N (used for mean term)
    f_hat_oof  – 2-fold cross-fitted predictions on sampled items (used for residuals)
    """
    y_int    = y_s.astype(int)
    n_s      = len(y_s)
    f_hat_all = _fit_ordinal(X_s, y_int, X_all, alpha, fallback)

    f_hat_oof = np.full(n_s, fallback, dtype=np.float64)
    if n_s >= 4:  # need at least 2 items per fold
        kf = KFold(n_splits=2, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X_s):
            f_hat_oof[te_idx] = _fit_ordinal(
                X_s[tr_idx], y_int[tr_idx], X_s[te_idx], alpha, fallback
            )

    return f_hat_all, f_hat_oof


def run_trial(
    y: np.ndarray,
    X_full: np.ndarray,
    X_llm: np.ndarray,
    X_emb: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    alpha_ord_full: float,
    alpha_ord_llm: float,
    alpha_ord_emb: float,
) -> dict:
    N   = len(y)
    cf_seed = int(rng.integers(0, 1_000_000))  # reproducible per-trial cross-fit seed

    sampled = rng.binomial(1, eta, N).astype(bool)
    if sampled.sum() < 20:
        extra = rng.choice(np.where(~sampled)[0], size=20 - sampled.sum(), replace=False)
        sampled[extra] = True

    row      = {"true_mean": float(np.mean(y)), "n_sampled": int(sampled.sum())}
    fallback = float(np.mean(y[sampled]))
    n_s      = int(sampled.sum())

    # 1. Intercept-only: f_hat = sample_mean everywhere; OOF = same constant
    #    Hajék: mean(fallback) + mean(Y[s] - fallback) = mean(Y[s]) = sample mean ✓
    _record_model("intercept",
                  np.full(N, fallback), np.full(n_s, fallback),
                  y, sampled, row)

    # 2. Ordinal full  (2-fold cross-fit)
    f_all, f_oof = _fit_ordinal_crossfit(X_full[sampled], y[sampled], X_full,
                                          alpha_ord_full, fallback, cf_seed)
    _record_model("ordinal_full", f_all, f_oof, y, sampled, row)

    # 3. Ordinal no-emb
    f_all, f_oof = _fit_ordinal_crossfit(X_llm[sampled], y[sampled], X_llm,
                                          alpha_ord_llm, fallback, cf_seed)
    _record_model("ordinal_no_emb", f_all, f_oof, y, sampled, row)

    # 4. Ordinal emb-only
    f_all, f_oof = _fit_ordinal_crossfit(X_emb[sampled], y[sampled], X_emb,
                                          alpha_ord_emb, fallback, cf_seed)
    _record_model("ordinal_emb", f_all, f_oof, y, sampled, row)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + plotting
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

            r2_vals  = sub[f"{method}_r2"].dropna().values
            rm_vals  = sub[f"{method}_rmse"].dropna().values
            sp_vals  = sub[f"{method}_spearman"].dropna().values

            def _se(v):
                return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

            agg[method][eta] = {
                "bias":       bias,
                "bias_se":    float(sd / np.sqrt(n)),
                "sd":         sd,
                "sd_se":      float(sd / np.sqrt(2 * max(n - 1, 1))),   # chi-sq approx
                "dr_rmse":    rmse,
                "dr_rmse_se": float(np.std(errors ** 2) / (2 * max(rmse, 1e-12) * np.sqrt(n))),
                "r2":         float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_se":      _se(r2_vals),
                "out_rmse":   float(np.mean(rm_vals))  if len(rm_vals)  else np.nan,
                "out_rmse_se": _se(rm_vals),
                "spearman":   float(np.mean(sp_vals))  if len(sp_vals)  else np.nan,
                "spearman_se": _se(sp_vals),
            }
    return agg


def _style_ax(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")


def compute_raw_baseline_stats() -> dict:
    """Compute bias, R², RMSE, Spearman for the uncalibrated raw-judge-average baseline.

    Bias = mean(raw_avg_score) - mean(y_true) over all grade-10 essays.
    Metrics are on the full dataset (no CV), serving as η-independent reference lines.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    human = pd.read_csv(DATA_CSV, low_memory=False)
    human = human.drop_duplicates("essay_id_comp")[
        ["essay_id_comp", "holistic_essay_score", "grade_level"]
    ].copy()
    human = human[human["grade_level"] == GRADE].rename(
        columns={"essay_id_comp": "id", "holistic_essay_score": "score"}
    ).reset_index(drop=True)

    llm = pd.read_json(LLM_PATH, lines=True).rename(columns={"essay_id_comp": "id"})
    pivot = llm.pivot_table(index="id", columns="model", values="predicted_score", aggfunc="first")
    raw_avg = pivot.mean(axis=1, skipna=True).reset_index().rename(columns={0: "raw_avg"})

    merged = human.merge(raw_avg, on="id", how="inner")
    mask   = np.isfinite(merged["raw_avg"].values)
    y      = merged["score"].astype(float).values[mask]
    yp     = merged["raw_avg"].values[mask]

    sp = float(pd.Series(y).corr(pd.Series(yp), method="spearman"))
    return {
        "bias":    float(np.mean(yp) - np.mean(y)),
        "r2":      float(r2_score(y, yp)),
        "rmse":    float(np.sqrt(mean_squared_error(y, yp))),
        "spearman": sp,
    }


def make_plots(
    results_df: pd.DataFrame,
    budgets: list[float],
    out_dir: Path,
    raw_stats: dict | None = None,
) -> None:
    agg     = aggregate(results_df, budgets)
    methods = list(MODEL_LABELS.keys())
    xs      = budgets

    RAW_COLOR = "#9467bd"   # purple, distinct from existing palette

    sns.set_theme(style="whitegrid", font_scale=1.1)

    def _annotate_hline(ax: plt.Axes, val: float, color: str, text: str) -> None:
        """Draw a dashed horizontal line and place a text label at the right edge."""
        ax.axhline(val, color=color, linestyle="--", linewidth=1.8, alpha=0.85)
        xlim = ax.get_xlim()
        x_pos = xlim[0] + 0.98 * (xlim[1] - xlim[0])
        ax.text(x_pos, val, text, color=color, fontsize=9, fontweight="bold",
                ha="right", va="bottom")

    # ── Figure 1: DR estimator quality ──────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics1 = [
        ("bias",    "bias_se",     "Bias"),
        ("sd",      "sd_se",       "Standard Deviation"),
        ("dr_rmse", "dr_rmse_se",  "RMSE"),
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
            if raw_stats is not None:
                _annotate_hline(ax, raw_stats["bias"], RAW_COLOR, "raw judge avg")
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig1.legend(handles, labels, loc="upper center", ncol=4,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig1.tight_layout()
    p1 = out_dir / "uniform_dr_bias_sd_rmse.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")

    # ── Figure 2: Outcome model quality on held-out items ───────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics2 = [
        ("r2",       "r2_se",        "R²",         "r2"),
        ("out_rmse", "out_rmse_se",  "RMSE",        "rmse"),
        ("spearman", "spearman_se",  "Spearman ρ",  "spearman"),
    ]
    outcome_methods = [m for m in methods if m != "intercept"]
    for ax, (key, se_key, ylabel, raw_key) in zip(axes, metrics2):
        for method in outcome_methods:
            ys  = np.array([agg[method][e][key]    for e in xs])
            ses = np.array([agg[method][e][se_key] for e in xs])
            c   = MODEL_COLORS[method]
            ax.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
        if raw_stats is not None:
            _annotate_hline(ax, raw_stats[raw_key], RAW_COLOR, "raw judge avg")
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)

    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig2.legend(handles, labels, loc="upper center", ncol=3,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig2.tight_layout()
    p2 = out_dir / "uniform_dr_outcome_r2_rmse_spearman.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Uniform DR simulation on PERSUADE.")
    parser.add_argument("--trials",    type=int,   default=DEFAULT_TRIALS)
    parser.add_argument("--budgets",   nargs="+",  type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--seed",      type=int,   default=DEFAULT_SEED)
    parser.add_argument("--emb-pca",   type=int,   default=DEFAULT_EMB_PCA)
    parser.add_argument("--out-dir",   type=Path,  default=OUT_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; reload saved CSV and regenerate plots.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        csv_path = out_dir / "uniform_dr_simulation_trials.csv"
        print(f"Loading saved trials from {csv_path} ...")
        results_df = pd.read_csv(csv_path)
        budgets = sorted(results_df["eta"].unique().tolist())
        raw_stats = compute_raw_baseline_stats()
        print(f"Raw baseline stats: {raw_stats}")
        make_plots(results_df, budgets, out_dir, raw_stats=raw_stats)
        return

    print("Loading data...")
    y, X_full, X_llm, X_emb = load_data(emb_pca=args.emb_pca, seed=args.seed)
    print(
        f"  N={len(y)} grade-{GRADE} essays | "
        f"X_full={X_full.shape} | X_llm={X_llm.shape} | X_emb={X_emb.shape}\n"
        f"  y mean={y.mean():.3f}  std={y.std():.3f}"
    )

    print("\nPre-selecting ordinal alphas (5-fold CV on full data)...")
    alpha_ord_full = select_ordinal_alpha(X_full, y, args.seed, "ordinal_full")
    alpha_ord_llm  = select_ordinal_alpha(X_llm,  y, args.seed, "ordinal_no_emb")
    alpha_ord_emb  = select_ordinal_alpha(X_emb,  y, args.seed, "ordinal_emb")

    rng         = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(1, 10_000_000, size=args.trials).tolist()

    rows: list[dict] = []
    for eta in args.budgets:
        print(f"\nBudget η={eta:.2f}  ({args.trials} trials)...")
        for t, s in enumerate(trial_seeds):
            trial_rng = np.random.default_rng(s)
            row       = run_trial(y, X_full, X_llm, X_emb, eta, trial_rng,
                                  alpha_ord_full, alpha_ord_llm, alpha_ord_emb)
            row["eta"] = eta
            rows.append(row)
            if (t + 1) % 20 == 0:
                print(f"  trial {t+1}/{args.trials}")

    results_df = pd.DataFrame(rows)
    csv_path   = out_dir / "uniform_dr_simulation_trials.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved trial data: {csv_path}")

    # Summary table
    summary_rows = []
    for eta in args.budgets:
        sub = results_df[results_df["eta"] == eta]
        for method in MODEL_LABELS:
            errors = sub[f"{method}_dr_error"].dropna().values
            summary_rows.append({
                "eta":        eta,
                "model":      method,
                "dr_bias":    float(np.mean(errors)),
                "dr_sd":      float(np.std(errors, ddof=1)),
                "dr_rmse":    float(np.sqrt(np.mean(errors ** 2))),
                "r2":         float(sub[f"{method}_r2"].mean()),
                "out_rmse":   float(sub[f"{method}_rmse"].mean()),
                "spearman":   float(sub[f"{method}_spearman"].mean()),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "uniform_dr_simulation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    raw_stats = compute_raw_baseline_stats()
    print(f"Raw baseline stats: {raw_stats}")
    make_plots(results_df, args.budgets, out_dir, raw_stats=raw_stats)


if __name__ == "__main__":
    main()
