"""
train.py — Two-stage DR experiment runner.

This file is edited directly by the autoresearch loop. Each version implements
a specific hypothesis about how to improve the two-stage DR estimator.

Outputs a JSON object to stdout with keys:
  dr_rmse, dr_bias, dr_sd,
  baseline2_rmse, baseline2_bias,
  dr_vs_b2_ratio,
  avg_sigma_mean, avg_sigma_q10, avg_sigma_q90, avg_sample_rate,
  n_trials, label, notes

Usage:
  python datasets/mqm/train.py [--trials N] [--n-jobs K] [--seed S]

Current hypothesis: Direction C+E combined — softmax temperature allocation
using ALL X_full features (88-dim) for sigma model. Temperature T=5.0 gives
smooth, budget-exact sigma distribution. Better than plain normalization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from datasets.mqm.two_stage_dr_simulation import (
    build_feature_table,
    fit_outcome,
    fit_sigma_model,
)

# ── Experiment label and notes (edited by autoresearch) ──────────────────────
LABEL = "softmax_T5_xfull_pi015_eta020"
NOTES = (
    "Same softmax T=5 + X_full sigma model as best previous config, "
    "but with pi=0.15 (larger pilot) to get better residual estimates "
    "for sigma fitting. Stage-2 budget is eta-pi=0.05. "
    "Hypothesis: larger pilot reduces sigma estimation noise, "
    "improving adaptive allocation quality."
)

# ── Hyperparameters (edited by autoresearch) ──────────────────────────────────
PI = 0.15               # stage-1 sampling rate (increased from 0.10)
ETA = 0.20              # total budget fraction
OUTCOME_MODEL = "hurdle"  # "ridge" | "hurdle"
VAR_TRANSFORM = "sqrt"  # "sqrt" | "log"
JUDGE_PCA = 64
SIGMA_PCA = 16          # used for X_sigma build (kept for compatibility)

SIGMA_L2_BETA = 1e-3
SIGMA_LR = 5e-2
SIGMA_OUTER_ITERS = 25
SIGMA_INNER_STEPS = 150
SIGMA_RESTARTS = 2
SIGMA_RHO0 = 1.0
SIGMA_TOL = 1e-4

RESID_CLIP_Q = 0.99     # quantile for clipping residuals before sigma fitting
Q_MIN_CLIP = 1e-6       # minimum propensity in DR formula
S1_MIN = 30             # minimum stage-1 sample size guardrail
SIGMA_MIN = 0.08        # floor for sigma (budget-neutral)

# Softmax temperature: higher T → more uniform allocation
SOFTMAX_TEMPERATURE = 5.0


# ── Patchable functions (edited by autoresearch) ──────────────────────────────

def compute_dr(f_all: np.ndarray, s: np.ndarray, y: np.ndarray,
               q_all: np.ndarray) -> float:
    """Doubly-robust estimate of the population mean.

    Self-normalized (Hajek) DR estimator:
      mu_hat = sum_i [ f_i + s_i * (y_i - f_i) / q_i ] / sum_i [ s_i / q_i ]
    """
    q = np.clip(q_all, Q_MIN_CLIP, None)
    numer = np.sum(f_all + s * (y - f_all) / q)
    denom = np.sum(s / q)
    # If denom is too small, fall back to standard mean (shouldn't happen)
    if denom < 1e-8:
        return float(np.mean(f_all))
    return float(numer / denom)


def transform_residuals(resid_sq: np.ndarray) -> np.ndarray:
    """Transform squared residuals into weights for the sigma model.

    Default: clip at RESID_CLIP_Q-th quantile.
    """
    clip = np.quantile(resid_sq, RESID_CLIP_Q) if resid_sq.size > 10 else float(np.max(resid_sq))
    return np.clip(resid_sq, 0.0, clip).astype(np.float32)


def fit_sigma_softmax_ridge(
    X_sigma_feat: np.ndarray,
    s1_idx: np.ndarray,
    resid_sq_s1: np.ndarray,
    target_sigma_mean: float,
    sigma_min: float,
    temperature: float,
) -> np.ndarray:
    """Fit sigma using RidgeCV + softmax temperature allocation.

    Steps:
    1. Fit RidgeCV on sqrt(residuals) using X_sigma_feat[s1_idx].
    2. Predict scores for all items.
    3. Apply softmax with temperature to allocate stage-2 budget exactly:
       sigma_i = softmax((scores - mean) / (std * T)) * stage2_budget_n
    4. Clip to [0, 1].
    5. Apply budget-neutral floor at sigma_min.

    Args:
        X_sigma_feat: Feature matrix (n x d) for sigma model.
        s1_idx: Indices of stage-1 sampled items.
        resid_sq_s1: Squared (and clipped) residuals for stage-1 items.
        target_sigma_mean: Target mean for sigma (budget constraint).
        sigma_min: Floor value for sigma.
        temperature: Softmax temperature (higher = more uniform).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    n = X_sigma_feat.shape[0]
    stage2_budget_n = target_sigma_mean * n  # total stage-2 "expected samples"

    # sqrt of clipped residuals as target
    sqrt_r = np.sqrt(np.maximum(resid_sq_s1, 0.0))

    # Fit ridge regression
    ridge = Pipeline([
        ("sc", StandardScaler()),
        ("r", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
    ])
    ridge.fit(X_sigma_feat[s1_idx], sqrt_r)
    raw_scores = ridge.predict(X_sigma_feat)

    # Softmax temperature allocation
    scores_mean = raw_scores.mean()
    scores_std = raw_scores.std()
    if scores_std < 1e-9:
        # Degenerate case: uniform allocation
        sigma_alloc = np.full(n, target_sigma_mean)
    else:
        # Normalize and apply temperature
        z = (raw_scores - scores_mean) / (scores_std * temperature)
        # Softmax
        z_shifted = z - z.max()  # numerical stability
        exp_z = np.exp(z_shifted)
        softmax_vals = exp_z / exp_z.sum()
        # Allocate budget
        sigma_alloc = softmax_vals * stage2_budget_n
        # Clip to [0, 1]
        sigma_alloc = np.clip(sigma_alloc, 0.0, 1.0)

    # Budget-neutral floor at sigma_min
    sigma_floored = np.clip(sigma_alloc, sigma_min, 1.0)
    excess = sigma_floored.mean() - target_sigma_mean
    if excess > 1e-6:
        free_mask = sigma_alloc >= sigma_min
        if free_mask.any():
            adjustment = excess * len(sigma_floored) / free_mask.sum()
            sigma_floored[free_mask] = np.clip(
                sigma_floored[free_mask] - adjustment,
                0.0, 1.0
            )
        # Re-floor after adjustment
        sigma_floored = np.clip(sigma_floored, sigma_min, 1.0)

    # Verify budget constraint — if badly off, fall back to uniform
    if abs(sigma_floored.mean() - target_sigma_mean) > 0.02:
        return np.full(n, target_sigma_mean)

    return sigma_floored


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(y: np.ndarray, X_full: np.ndarray, X_sigma: np.ndarray,
              rng: np.random.Generator) -> dict:
    n = y.shape[0]
    idx = np.arange(n)
    m_budget = max(1, int(round(ETA * n)))

    # Baseline B1: uniform direct sample mean
    s_base = np.zeros(n, dtype=bool)
    s_base[rng.choice(idx, size=m_budget, replace=False)] = True
    baseline1_mean = float(np.mean(y[s_base]))

    # Baseline B2: uniform sample + outcome model + DR
    model_base = fit_outcome(X_full[s_base], y[s_base], OUTCOME_MODEL)
    f_base = model_base.predict(X_full).astype(np.float32)
    q_base = np.full(n, float(m_budget) / n, dtype=np.float32)
    baseline2_dr = compute_dr(f_base, s_base.astype(np.float32), y, q_base)

    # Stage 1
    s1 = rng.binomial(1, PI, size=n).astype(bool)
    if s1.sum() < S1_MIN:
        add_n = min(S1_MIN - s1.sum(), (~s1).sum())
        if add_n > 0:
            s1[rng.choice(idx[~s1], size=add_n, replace=False)] = True

    model1 = fit_outcome(X_full[s1], y[s1], OUTCOME_MODEL)
    f1_all = model1.predict(X_full).astype(np.float32)
    resid_sq_s1 = (y[s1] - f1_all[s1]) ** 2
    weights_s1 = transform_residuals(resid_sq_s1)

    # Target sigma mean for stage-2 budget
    target_sigma_mean = (ETA - PI) / (1.0 - PI)  # ≈ 0.111

    # Use ALL X_full features (88-dim: judge scores + uncertainty + PCA embeddings)
    # for sigma model — captures maximum hardness signal
    X_sigma_feat = X_full  # shape (n, 88)

    # Sigma model: softmax temperature allocation with Ridge scores
    sigma_all = fit_sigma_softmax_ridge(
        X_sigma_feat=X_sigma_feat,
        s1_idx=np.where(s1)[0],
        resid_sq_s1=weights_s1,
        target_sigma_mean=target_sigma_mean,
        sigma_min=SIGMA_MIN,
        temperature=SOFTMAX_TEMPERATURE,
    )

    q_all = PI + (1.0 - PI) * sigma_all

    # Stage 2
    s2 = np.zeros(n, dtype=bool)
    remain = ~s1
    s2[remain] = rng.binomial(1, sigma_all[remain]).astype(bool)
    s = s1 | s2

    # Final model + DR
    model_final = fit_outcome(X_full[s], y[s], OUTCOME_MODEL)
    f_all = model_final.predict(X_full).astype(np.float32)
    dr = compute_dr(f_all, s.astype(np.float32), y, q_all)

    true_mean = float(np.mean(y))
    return {
        "dr_error":         float(dr) - true_mean,
        "baseline1_error":  baseline1_mean - true_mean,
        "baseline2_error":  float(baseline2_dr) - true_mean,
        "sigma_mean":       float(sigma_all.mean()),
        "sigma_q10":        float(np.quantile(sigma_all, 0.10)),
        "sigma_q90":        float(np.quantile(sigma_all, 0.90)),
        "sample_rate":      float(s.mean()),
    }


# ── Parallel worker ───────────────────────────────────────────────────────────

_Y = _X_FULL = _X_SIGMA = None

def _init(y, xf, xs):
    global _Y, _X_FULL, _X_SIGMA
    _Y, _X_FULL, _X_SIGMA = y, xf, xs

def _worker(seed: int) -> dict:
    return run_trial(_Y, _X_FULL, _X_SIGMA, np.random.default_rng(seed))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/mqm"))
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    emb_path = results_dir / "text_embedding_3_large_mqm_embeddings_separate.npz"
    tsv_path = data_dir / "mqm_newstest2020_ende.tsv"

    _, y, X_full, X_sigma = build_feature_table(
        tsv_path=tsv_path, data_dir=data_dir, emb_path=emb_path,
        judge_pca_components=JUDGE_PCA, sigma_pca_components=SIGMA_PCA,
        var_transform=VAR_TRANSFORM, system_filter=None,
    )

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, 10_000_000, size=args.trials).tolist()
    rows: list[dict] = []

    if args.n_jobs <= 1:
        for s in seeds:
            rows.append(run_trial(y, X_full, X_sigma, np.random.default_rng(s)))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs,
                                 initializer=_init, initargs=(y, X_full, X_sigma)) as ex:
            for r in as_completed([ex.submit(_worker, int(s)) for s in seeds]):
                rows.append(r.result())

    df = pd.DataFrame(rows)

    def rmse(col): return float(np.sqrt(np.mean(col ** 2)))

    result = {
        "label":            LABEL,
        "notes":            NOTES,
        "n_trials":         args.trials,
        "dr_bias":          float(df["dr_error"].mean()),
        "dr_sd":            float(df["dr_error"].std(ddof=1)),
        "dr_rmse":          rmse(df["dr_error"]),
        "baseline2_bias":   float(df["baseline2_error"].mean()),
        "baseline2_rmse":   rmse(df["baseline2_error"]),
        "baseline1_rmse":   rmse(df["baseline1_error"]),
        "dr_vs_b2_ratio":   rmse(df["dr_error"]) / max(rmse(df["baseline2_error"]), 1e-9),
        "avg_sigma_mean":   float(df["sigma_mean"].mean()),
        "avg_sigma_q10":    float(df["sigma_q10"].mean()),
        "avg_sigma_q90":    float(df["sigma_q90"].mean()),
        "avg_sample_rate":  float(df["sample_rate"].mean()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()