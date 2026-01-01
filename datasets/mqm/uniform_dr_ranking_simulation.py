"""
Uniform-sampling DR simulation for MQM newstest2020 en-de.

Goal: rank 10 MT systems by mean MQM score using a small sampling budget.

Four outcome models, six uniform sampling budgets eta ∈ {0.05,...,0.30}.
Each trial:
  - For each of 10 systems: sample Bernoulli(eta) segments independently
  - Estimate system mean MQM via each method
  - Build Bonferroni-corrected 95% CI:
      half-width = z_bonf * sqrt(var(Y - f_hat | sampled) / n_sampled)
      z_bonf = norm.ppf(1 - 0.05 / (2 * 10))
  - Rank 10 systems by estimated mean → compare to true ranking

Methods:
  1. intercept      – sample mean per system (f_hat = 0, residuals = Y)
  2. hurdle_full    – Hurdle DR: LLM scores + uncertainty + embedding PCA
  3. hurdle_no_emb  – Hurdle DR: LLM scores + uncertainty, no embeddings
  4. hurdle_emb     – Hurdle DR: embedding PCA only

Output (three figures):
  Figure 1: DR bias / SD / RMSE vs eta  (4 curves ± 1 SE band)
  Figure 2: CI coverage vs eta          (Bonferroni 95%, dashed target line)
  Figure 3: System ranking quality      (Kendall τ and Spearman ρ vs eta)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

# Must be set before any numpy/sklearn import to limit BLAS threads per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
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

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("datasets/mqm/data")
TSV_PATH    = DATA_DIR / "mqm_newstest2020_ende.tsv"
EMB_PATH    = Path("datasets/mqm/results/text_embedding_3_large_mqm_embeddings_separate.npz")
OUT_DIR     = Path("datasets/mqm/results")

DEFAULT_BUDGETS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_TRIALS   = 200
DEFAULT_SEED     = 42
DEFAULT_EMB_PCA  = 64

RIDGE_ALPHAS = np.logspace(-3, 4, 20)
N_SYSTEMS    = 10
ALPHA        = 0.05  # family-wise error rate
Z_BONF       = float(norm.ppf(1.0 - ALPHA / (2.0 * N_SYSTEMS)))  # ≈ 2.807

MODEL_LABELS = {
    "intercept":     "Intercept-only (sample mean)",
    "hurdle_full":   "Hurdle: scores + unc + emb",
    "hurdle_no_emb": "Hurdle: scores + unc, no emb",
    "hurdle_emb":    "Hurdle: emb only",
}
MODEL_COLORS = {
    "intercept":     "#888888",
    "hurdle_full":   "#1f77b4",
    "hurdle_no_emb": "#ff7f0e",
    "hurdle_emb":    "#2ca02c",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(
    emb_pca: int,
    seed: int,
    models: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {system: {y, X_full, X_no_emb, X_emb}} for all 10 systems.

    PCA is fit on pooled embeddings across all systems (no label leakage).
    LLM features are scaled globally too. Only outcome models are fitted
    on sampled data within each trial.
    """
    human   = load_human_mqm(TSV_PATH)
    llm_agg = load_llm_mqm(DATA_DIR, models=models)
    emb_df  = load_embeddings(EMB_PATH)

    systems = sorted(human["system"].unique().tolist())
    print(f"  Systems ({len(systems)}): {systems}")

    # Build pooled feature table (all systems together) for global scaling/PCA
    df_all, feat_cols = build_feature_table(
        human, llm_agg, emb_df,
        use_uncertainty=True,
        system_filter=None,
        var_transform="sqrt",
    )

    emb_cols    = [c for c in feat_cols if c.startswith("emb__")]
    non_emb_cols = [c for c in feat_cols if c not in emb_cols]

    # Global PCA on embeddings (unsupervised — no label leakage)
    X_emb_raw = df_all[emb_cols].values.astype(np.float32)
    imp_emb   = SimpleImputer(strategy="mean").fit(X_emb_raw)
    X_emb_imp = imp_emb.transform(X_emb_raw)
    sc_emb    = StandardScaler().fit(X_emb_imp)
    X_emb_z   = sc_emb.transform(X_emb_imp)
    n_pca     = min(emb_pca, X_emb_z.shape[1], X_emb_z.shape[0] - 1)
    pca       = PCA(n_components=n_pca, random_state=seed).fit(X_emb_z)
    X_emb_pca = pca.transform(X_emb_z).astype(np.float32)
    sc_pca    = StandardScaler().fit(X_emb_pca)
    X_emb_all = sc_pca.transform(X_emb_pca).astype(np.float32)

    # Global scaling + winsorization (5th–95th pct) on non-emb features
    X_ne_raw  = df_all[non_emb_cols].values.astype(np.float32)
    imp_ne    = SimpleImputer(strategy="mean").fit(X_ne_raw)
    sc_ne     = StandardScaler().fit(imp_ne.transform(X_ne_raw))
    X_ne_all  = sc_ne.transform(imp_ne.transform(X_ne_raw)).astype(np.float32)
    ne_lo     = np.percentile(X_ne_all, 5,  axis=0).astype(np.float32)
    ne_hi     = np.percentile(X_ne_all, 95, axis=0).astype(np.float32)
    X_ne_all  = np.clip(X_ne_all, ne_lo, ne_hi)
    X_full_all = np.hstack([X_ne_all, X_emb_all])

    df_all["_row"] = np.arange(len(df_all))

    result: dict[str, dict[str, np.ndarray]] = {}
    for sys in systems:
        mask = (df_all["system"] == sys).values
        rows = df_all.loc[mask, "_row"].values
        y    = df_all.loc[mask, TARGET].astype(np.float32).values
        result[sys] = {
            "y":       y,
            "X_full":  X_full_all[rows],
            "X_no_emb": X_ne_all[rows],
            "X_emb":   X_emb_all[rows],
        }
        print(
            f"  [{sys}] N={len(y)}  y_mean={y.mean():.3f}  "
            f"X_full={X_full_all[rows].shape}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hurdle model
# ─────────────────────────────────────────────────────────────────────────────

class HurdleModel:
    """Hurdle model: fixed-C logistic gate on embeddings × Ridge amount on full features.

    Gate uses embedding features only (stable, low-dimensional).
    Amount uses whatever features are passed as X (can be full or emb-only).
    Fixed C=0.1 regularization avoids inner-CV instability at small sample sizes.
    """

    def fit(self, X_amount: np.ndarray, X_gate: np.ndarray, y: np.ndarray) -> "HurdleModel":
        z = (y > 0).astype(int)
        self.frac_pos = float(z.mean())

        gate_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("lr",  LogisticRegression(C=0.1, max_iter=500, random_state=42, solver="lbfgs")),
        ])
        n_pos, n_neg = z.sum(), (1 - z).sum()
        if n_pos >= 3 and n_neg >= 3:
            try:
                gate_pipe.fit(X_gate, z)
                self.gate = gate_pipe
            except Exception:
                self.gate = None
        else:
            self.gate = None

        pos = y > 0
        self.pos_mean = float(y[pos].mean()) if pos.any() else 0.0
        if pos.sum() >= 5:
            self.amount = Pipeline([
                ("imp",   SimpleImputer(strategy="median")),
                ("sc",    StandardScaler()),
                ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
            ])
            self.amount.fit(X_amount[pos], y[pos])
        else:
            self.amount = None
        return self

    def predict(self, X_amount: np.ndarray, X_gate: np.ndarray) -> np.ndarray:
        p_pos = (
            self.gate.predict_proba(X_gate)[:, 1]
            if self.gate is not None
            else np.full(X_amount.shape[0], self.frac_pos)
        )
        e_pos = (
            self.amount.predict(X_amount).clip(min=0.0)
            if self.amount is not None
            else np.full(X_amount.shape[0], self.pos_mean)
        )
        return (p_pos * e_pos).astype(np.float64)


def _fit_hurdle(
    X_amount_s: np.ndarray,
    X_gate_s: np.ndarray,
    y_s: np.ndarray,
    X_amount_all: np.ndarray,
    X_gate_all: np.ndarray,
    fallback: float,
) -> np.ndarray:
    """Fit Hurdle on sampled items; return predictions on all N items."""
    try:
        m = HurdleModel()
        m.fit(X_amount_s, X_gate_s, y_s)
        return m.predict(X_amount_all, X_gate_all)
    except Exception:
        return np.full(len(X_amount_all), fallback)


def _fit_hurdle_crossfit(
    X_amount_s: np.ndarray,
    X_gate_s: np.ndarray,
    y_s: np.ndarray,
    X_amount_all: np.ndarray,
    X_gate_all: np.ndarray,
    fallback: float,
    seed: int = 0,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Hurdle; return (f_hat_all, f_hat_oof) for Hajék cross-fit DR."""
    try:
        m = HurdleModel()
        m.fit(X_amount_s, X_gate_s, y_s)
        f_hat_all = m.predict(X_amount_all, X_gate_all)
    except Exception:
        f_hat_all = np.full(len(X_amount_all), fallback, dtype=np.float64)

    n_s = len(y_s)
    f_hat_oof = np.full(n_s, fallback, dtype=np.float64)
    k = min(n_splits, n_s) if n_s >= 2 else 0
    if k >= 2:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(X_amount_s):
            try:
                m = HurdleModel()
                m.fit(X_amount_s[tr_idx], X_gate_s[tr_idx], y_s[tr_idx])
                f_hat_oof[te_idx] = m.predict(X_amount_s[te_idx], X_gate_s[te_idx])
            except Exception:
                pass
    return f_hat_all, f_hat_oof


# ─────────────────────────────────────────────────────────────────────────────
# DR estimate + CI for one (system, method)  — Hajék, no cross-fitting
# ─────────────────────────────────────────────────────────────────────────────

def _dr_estimate_and_ci(
    f_hat_all: np.ndarray,
    y: np.ndarray,
    sampled: np.ndarray,
    f_hat_oof: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Hajék DR + Bonferroni-corrected 95% CI.

    If f_hat_oof is given (cross-fit), residuals use OOF predictions.
    Otherwise residuals are in-sample: y[sampled] - f_hat_all[sampled].
    """
    resid = y[sampled] - (f_hat_oof if f_hat_oof is not None else f_hat_all[sampled])
    dr    = float(np.mean(f_hat_all) + np.mean(resid))
    n_s   = max(int(sampled.sum()), 1)
    se    = float(np.sqrt(np.var(resid, ddof=1) / n_s)) if n_s > 1 else 0.0
    hw    = Z_BONF * se
    return dr, dr - hw, dr + hw


# ─────────────────────────────────────────────────────────────────────────────
# Single trial — all 10 systems
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    data: dict[str, dict[str, np.ndarray]],
    true_means: dict[str, float],
    true_ranks: dict[str, int],
    eta: float,
    rng: np.random.Generator,
    use_crossfit: bool = False,
) -> list[dict]:
    """Run one trial for all systems under eta. Returns a flat list of per-(system, method) rows."""
    methods  = list(MODEL_LABELS.keys())
    systems  = list(data.keys())
    rows: list[dict] = []

    # Estimates for ranking (keyed by method)
    estimates: dict[str, dict[str, float]] = {m: {} for m in methods}
    ci_covers: dict[str, dict[str, bool]]  = {m: {} for m in methods}
    cf_seed = int(rng.integers(0, 1_000_000))

    for sys in systems:
        y      = data[sys]["y"].astype(np.float64)
        X_full = data[sys]["X_full"]
        X_ne   = data[sys]["X_no_emb"]
        X_emb  = data[sys]["X_emb"]
        N      = len(y)

        sampled = rng.binomial(1, eta, N).astype(bool)
        if sampled.sum() < 5:
            extra = rng.choice(np.where(~sampled)[0], size=5 - sampled.sum(), replace=False)
            sampled[extra] = True

        fallback = float(np.mean(y[sampled]))
        true_mu  = true_means[sys]

        # 1. Intercept-only: constant f_hat = sample mean
        f_int = np.full(N, fallback, dtype=np.float64)
        dr_int, lo_int, hi_int = _dr_estimate_and_ci(f_int, y, sampled)
        estimates["intercept"][sys] = dr_int
        ci_covers["intercept"][sys] = lo_int <= true_mu <= hi_int

        if use_crossfit:
            # Cross-fit: OOF residuals for unbiased DR variance
            f_all, f_oof = _fit_hurdle_crossfit(X_full[sampled], X_emb[sampled], y[sampled], X_full, X_emb, fallback, seed=cf_seed)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled, f_oof)
            estimates["hurdle_full"][sys] = dr; ci_covers["hurdle_full"][sys] = lo <= true_mu <= hi

            f_all, f_oof = _fit_hurdle_crossfit(X_ne[sampled], X_emb[sampled], y[sampled], X_ne, X_emb, fallback, seed=cf_seed)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled, f_oof)
            estimates["hurdle_no_emb"][sys] = dr; ci_covers["hurdle_no_emb"][sys] = lo <= true_mu <= hi

            f_all, f_oof = _fit_hurdle_crossfit(X_emb[sampled], X_emb[sampled], y[sampled], X_emb, X_emb, fallback, seed=cf_seed)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled, f_oof)
            estimates["hurdle_emb"][sys] = dr; ci_covers["hurdle_emb"][sys] = lo <= true_mu <= hi
        else:
            # In-sample residuals
            f_all = _fit_hurdle(X_full[sampled], X_emb[sampled], y[sampled], X_full, X_emb, fallback)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled)
            estimates["hurdle_full"][sys] = dr; ci_covers["hurdle_full"][sys] = lo <= true_mu <= hi

            f_all = _fit_hurdle(X_ne[sampled], X_emb[sampled], y[sampled], X_ne, X_emb, fallback)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled)
            estimates["hurdle_no_emb"][sys] = dr; ci_covers["hurdle_no_emb"][sys] = lo <= true_mu <= hi

            f_all = _fit_hurdle(X_emb[sampled], X_emb[sampled], y[sampled], X_emb, X_emb, fallback)
            dr, lo, hi = _dr_estimate_and_ci(f_all, y, sampled)
            estimates["hurdle_emb"][sys] = dr; ci_covers["hurdle_emb"][sys] = lo <= true_mu <= hi

        # Per-system rows (for bias/SD/RMSE computation)
        for method in methods:
            est = estimates[method][sys]
            rows.append({
                "system":    sys,
                "method":    method,
                "true_mean": true_mu,
                "estimate":  est,
                "dr_error":  est - true_mu,
                "ci_covers": ci_covers[method][sys],
            })

    # Ranking metrics (per-method, appended as extra rows with system="__rank__")
    true_rank_arr = np.array([true_ranks[s] for s in systems])
    for method in methods:
        est_arr  = np.array([estimates[method][s] for s in systems])
        est_rank = pd.Series(est_arr).rank(method="average").values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = float(spearmanr(est_rank, true_rank_arr).statistic)
            kt = float(kendalltau(est_arr, [true_means[s] for s in systems]).statistic)
        mae = float(np.mean(np.abs(est_rank - true_rank_arr)))
        rows.append({
            "system":    "__rank__",
            "method":    method,
            "true_mean": np.nan,
            "estimate":  np.nan,
            "dr_error":  np.nan,
            "ci_covers": np.nan,
            "spearman":  sp,
            "kendall":   kt,
            "rank_mae":  mae,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results_df: pd.DataFrame, budgets: list[float]) -> dict:
    methods = list(MODEL_LABELS.keys())
    agg: dict[str, dict] = {m: {} for m in methods}

    sys_df  = results_df[results_df["system"] != "__rank__"].copy()
    rank_df = results_df[results_df["system"] == "__rank__"].copy()

    for method in methods:
        for eta in budgets:
            se = sys_df[(sys_df["method"] == method) & (sys_df["eta"] == eta)]
            errors   = se["dr_error"].dropna().values
            coverage = se["ci_covers"].dropna().values.astype(float)
            n        = len(errors)
            bias     = float(np.mean(errors))
            sd       = float(np.std(errors, ddof=1))
            rmse     = float(np.sqrt(np.mean(errors ** 2)))

            sr = rank_df[(rank_df["method"] == method) & (rank_df["eta"] == eta)]
            sp_v  = sr["spearman"].dropna().values
            kt_v  = sr["kendall"].dropna().values
            mae_v = sr["rank_mae"].dropna().values

            def _se(v):
                return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

            agg[method][eta] = {
                "bias":        bias,
                "bias_se":     float(sd / np.sqrt(n)),
                "sd":          sd,
                "sd_se":       float(sd / np.sqrt(2 * max(n - 1, 1))),
                "dr_rmse":     rmse,
                "dr_rmse_se":  float(np.std(errors ** 2) / (2 * max(rmse, 1e-12) * np.sqrt(n))),
                "coverage":    float(np.mean(coverage)),
                "coverage_se": _se(coverage),
                "spearman":    float(np.mean(sp_v)) if len(sp_v) else np.nan,
                "spearman_se": _se(sp_v),
                "kendall":     float(np.mean(kt_v))  if len(kt_v)  else np.nan,
                "kendall_se":  _se(kt_v),
                "rank_mae":    float(np.mean(mae_v)) if len(mae_v)  else np.nan,
                "rank_mae_se": _se(mae_v),
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


def make_plots(results_df: pd.DataFrame, budgets: list[float], out_dir: Path, n_trials: int, suffix: str = "") -> None:
    agg     = aggregate(results_df, budgets)
    methods = list(MODEL_LABELS.keys())
    xs      = budgets

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Figure 1: DR estimator quality + CI coverage (4 panels) ────────────
    fig1, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    m1 = [
        ("bias",    "bias_se",    "Bias"),
        ("sd",      "sd_se",      "Standard Deviation"),
        ("dr_rmse", "dr_rmse_se", "RMSE"),
    ]
    for ax, (key, sek, ylabel) in zip(axes[:3], m1):
        for method in methods:
            ys  = np.array([agg[method][e][key] for e in xs])
            ses = np.array([agg[method][e][sek] for e in xs])
            c   = MODEL_COLORS[method]
            ax.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
        if key == "bias":
            ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)

    # 4th panel: CI coverage
    ax_cov = axes[3]
    target_cov = 1.0 - ALPHA
    ax_cov.axhline(target_cov, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Target {target_cov:.0%}")
    for method in methods:
        ys  = np.array([agg[method][e]["coverage"]    for e in xs])
        ses = np.array([agg[method][e]["coverage_se"] for e in xs])
        c   = MODEL_COLORS[method]
        ax_cov.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
        ax_cov.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
    ax_cov.set_ylim(0.925, 1.0)
    _style_ax(ax_cov, "Sampling budget η", "CI Coverage")
    ax_cov.set_xticks(xs)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig1.legend(handles, labels, loc="upper center", ncol=4,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig1.tight_layout()
    p1 = out_dir / f"uniform_dr_ranking_bias_sd_rmse{suffix}.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight"); plt.close(fig1)
    print(f"Saved: {p1.name}")

    # ── Figure 3: Ranking quality ───────────────────────────────────────────
    fig3, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    m3 = [
        ("spearman",  "spearman_se",  "Spearman ρ"),
        ("kendall",   "kendall_se",   "Kendall τ"),
        ("rank_mae",  "rank_mae_se",  "MAE (ranks)"),
    ]
    for ax, (key, sek, ylabel) in zip(axes, m3):
        for method in methods:
            ys  = np.array([agg[method][e][key] for e in xs])
            ses = np.array([agg[method][e][sek] for e in xs])
            c   = MODEL_COLORS[method]
            ax.plot(xs, ys, marker="o", markersize=7, label=MODEL_LABELS[method],
                    color=c, linewidth=2.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=c, alpha=0.15)
        _style_ax(ax, "Sampling budget η", ylabel)
        ax.set_xticks(xs)
    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig3.legend(handles, labels, loc="upper center", ncol=4,
                      bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=11)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig3.tight_layout()
    p3 = out_dir / f"uniform_dr_ranking_quality{suffix}.png"
    fig3.savefig(p3, dpi=200, bbox_inches="tight"); plt.close(fig3)
    print(f"Saved: {p3.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Parallel worker helpers (module-level so pickle can find them)
# ─────────────────────────────────────────────────────────────────────────────

# Set before Pool creation so fork() inherits them — zero IPC overhead
_worker_data:        dict | None = None
_worker_true_means:  dict | None = None
_worker_true_ranks:  dict | None = None
_worker_crossfit:    bool        = False


def _run_trial_job(eta_seed: tuple[float, int]) -> list[dict]:
    eta, seed = eta_seed
    trial_rng = np.random.default_rng(seed)
    rows = run_trial(_worker_data, _worker_true_means, _worker_true_ranks, eta, trial_rng,
                     use_crossfit=_worker_crossfit)
    for row in rows:
        row["eta"] = eta
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int,  default=DEFAULT_TRIALS)
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--seed",    type=int,  default=DEFAULT_SEED)
    parser.add_argument("--emb-pca", type=int,  default=DEFAULT_EMB_PCA)
    parser.add_argument("--workers",   type=int,  default=min(os.cpu_count() or 4, 12))
    parser.add_argument("--crossfit",  action="store_true", help="Use 5-fold cross-fitting for DR residuals")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; reload saved CSV and regenerate plots.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional subset of LLM model names to use (default: all)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory for results and figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        suffix = "_crossfit" if args.crossfit else ""
        csv_path = out_dir / "uniform_dr_ranking_trials.csv"
        print(f"Loading saved trials from {csv_path.name} ...")
        results_df = pd.read_csv(csv_path)
        budgets = sorted(results_df["eta"].unique().tolist())
        make_plots(results_df, budgets, out_dir, args.trials, suffix=suffix)
        return

    print("Loading data...")
    data = load_all_data(emb_pca=args.emb_pca, seed=args.seed, models=args.models)

    systems    = list(data.keys())
    true_means = {s: float(data[s]["y"].mean()) for s in systems}
    # Rank 1 = lowest MQM (best quality)
    sorted_sys = sorted(systems, key=lambda s: true_means[s])
    true_ranks = {s: i + 1 for i, s in enumerate(sorted_sys)}

    print("\nTrue system ranking (rank 1 = lowest MQM = best):")
    for s in sorted_sys:
        print(f"  {true_ranks[s]:2d}. {s:<35s}  mean MQM = {true_means[s]:.4f}")
    print(f"\nBonferroni z = {Z_BONF:.4f}  (α={ALPHA}, K={N_SYSTEMS} systems)")

    rng         = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(1, 10_000_000, size=args.trials).tolist()

    jobs = [(eta, s) for eta in args.budgets for s in trial_seeds]
    n_jobs = len(jobs)
    print(f"\nRunning {n_jobs} jobs ({args.trials} trials × {len(args.budgets)} budgets) "
          f"on {args.workers} workers...")

    # Assign globals before Pool so fork() inherits them — no pickle IPC per worker
    global _worker_data, _worker_true_means, _worker_true_ranks, _worker_crossfit
    _worker_data       = data
    _worker_true_means = true_means
    _worker_true_ranks = true_ranks
    _worker_crossfit   = args.crossfit

    all_rows: list[dict] = []
    with Pool(processes=args.workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_run_trial_job, jobs, chunksize=4), 1):
            all_rows.extend(result)
            if done % 100 == 0 or done == n_jobs:
                print(f"  {done}/{n_jobs} jobs done")

    results_df = pd.DataFrame(all_rows)
    csv_path   = out_dir / "uniform_dr_ranking_trials.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved trial data: {csv_path.name}")

    # Summary table
    agg = aggregate(results_df, args.budgets)
    print("\n=== Summary (aggregated across 10 systems and 200 trials) ===")
    rows_summary = []
    for eta in args.budgets:
        for method in MODEL_LABELS:
            a = agg[method][eta]
            rows_summary.append({
                "eta": eta, "model": method,
                "dr_bias": a["bias"], "dr_sd": a["sd"], "dr_rmse": a["dr_rmse"],
                "coverage": a["coverage"],
                "spearman": a["spearman"], "kendall": a["kendall"],
                "rank_mae": a["rank_mae"],
            })
    summary_df = pd.DataFrame(rows_summary)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_df.to_csv(out_dir / "uniform_dr_ranking_summary.csv", index=False)

    suffix = "_crossfit" if args.crossfit else ""
    make_plots(results_df, args.budgets, out_dir, args.trials, suffix=suffix)


if __name__ == "__main__":
    main()
