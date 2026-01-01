"""
Two-stage adaptive sampling simulation for mean MQM score estimation.

Design:
1) Stage-1 label sampling: S1_i ~ Bernoulli(pi)
2) Fit outcome model f using labeled stage-1 segments
3) Fit adaptive stage-2 propensity model sigma(x) under budget eta
4) Stage-2 sampling on unlabeled pool: S2_i ~ Bernoulli(sigma_i) if S1_i=0
5) Refit outcome model on all sampled segments and estimate mean via DR:

    mu_hat = mean_i [ f_i + S_i * (Y_i - f_i) / q_i ]
    with q_i = pi + (1-pi)*sigma_i

Baselines per trial:
    (B1) Uniformly sample exactly round(eta * N) segments, report sample mean.
    (B2) Same uniform sample size, fit judge model and compute DR with q_i = m/N.

Supports both Ridge and Hurdle outcome models, and both pooled and
per-system operation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from datasets.mqm.MRR import (
    TARGET,
    build_feature_table as mrr_build_feature_table,
    load_embeddings,
    load_human_mqm,
    load_llm_mqm,
)

_W_Y: np.ndarray | None = None
_W_X_FULL: np.ndarray | None = None
_W_X_SIGMA: np.ndarray | None = None
_W_PI: float | None = None
_W_ETA: float | None = None
_W_SIGMA_L2_BETA: float | None = None
_W_SIGMA_DEVICE: str | None = None
_W_OUTCOME_MODEL: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def build_feature_table(
    tsv_path: Path,
    data_dir: Path,
    emb_path: Path,
    judge_pca_components: int,
    sigma_pca_components: int,
    var_transform: str = "sqrt",
    system_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MQM data and return (ids, y, X_full, X_sigma).

    X_full: judge features = scaled non-emb cols + PCA(judge_pca) on embeddings
    X_sigma: sigma features = PCA(sigma_pca) on embeddings only
    """
    human = load_human_mqm(tsv_path)
    llm_agg = load_llm_mqm(data_dir)
    emb = load_embeddings(emb_path)

    df, feat_cols = mrr_build_feature_table(
        human, llm_agg, emb,
        use_uncertainty=True,
        system_filter=system_filter,
        var_transform=var_transform,
    )

    y = df[TARGET].astype(np.float32).values
    ids = df["id"].astype(str).values

    emb_cols = [c for c in feat_cols if c.startswith("emb__")]
    non_emb_cols = [c for c in feat_cols if c not in emb_cols]

    X_non_emb = df[non_emb_cols].values.astype(np.float32) if non_emb_cols else np.empty((len(df), 0), dtype=np.float32)
    X_emb_raw = df[emb_cols].values.astype(np.float32) if emb_cols else np.empty((len(df), 0), dtype=np.float32)

    imp = SimpleImputer(strategy="mean")
    X_emb_raw = imp.fit_transform(X_emb_raw)

    scaler_emb = StandardScaler()
    X_emb_z = scaler_emb.fit_transform(X_emb_raw)

    # Judge features: scaled non-emb + PCA(judge_pca) on embeddings
    if non_emb_cols:
        imp_ne = SimpleImputer(strategy="mean")
        sc_ne = StandardScaler()
        X_ne_z = sc_ne.fit_transform(imp_ne.fit_transform(X_non_emb))
    else:
        X_ne_z = np.empty((len(df), 0), dtype=np.float32)

    if emb_cols:
        judge_pca_n = min(judge_pca_components, X_emb_z.shape[1], X_emb_z.shape[0] - 1)
        judge_pca = PCA(n_components=judge_pca_n, random_state=42)
        X_emb_judge = judge_pca.fit_transform(X_emb_z).astype(np.float32)
    else:
        X_emb_judge = np.empty((len(df), 0), dtype=np.float32)

    X_full = np.hstack([X_ne_z, X_emb_judge]).astype(np.float32)

    # Sigma features: PCA(sigma_pca) on embeddings only
    if emb_cols:
        sigma_pca_n = min(sigma_pca_components, X_emb_z.shape[1], X_emb_z.shape[0] - 1)
        sigma_pca = PCA(n_components=sigma_pca_n, random_state=42)
        X_sigma = sigma_pca.fit_transform(X_emb_z).astype(np.float32)
    else:
        X_sigma = np.empty((len(df), 0), dtype=np.float32)

    return ids, y, X_full, X_sigma


# ─────────────────────────────────────────────────────────────────────────
# Outcome models
# ─────────────────────────────────────────────────────────────────────────

RIDGE_ALPHAS = np.logspace(-3, 4, 20)


def fit_outcome_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])
    model.fit(X_train, y_train)
    return model


class HurdleModel:
    """Logistic gate (P(y>0)) + Ridge on positives. Predictions are non-negative."""

    def __init__(self):
        self.gate = None
        self.amount = None
        self.pos_mean = 0.0

    def fit(self, X, y):
        z = (y > 0).astype(int)
        self.gate = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegressionCV(Cs=10, cv=3, max_iter=2000, random_state=42)),
        ])
        self.gate.fit(X, z)
        pos = y > 0
        self.pos_mean = float(y[pos].mean()) if pos.any() else 0.0
        if pos.sum() >= 10:
            self.amount = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
            ])
            self.amount.fit(X[pos], y[pos])
        else:
            self.amount = None
        return self

    def predict(self, X):
        p_pos = self.gate.predict_proba(X)[:, 1]
        if self.amount is not None:
            e_pos = self.amount.predict(X).clip(min=0)
        else:
            e_pos = np.full(X.shape[0], self.pos_mean)
        return p_pos * e_pos


def fit_outcome_hurdle(X_train: np.ndarray, y_train: np.ndarray) -> HurdleModel:
    model = HurdleModel()
    model.fit(X_train, y_train)
    return model


def fit_outcome(X_train, y_train, model_type="hurdle"):
    if model_type == "ridge":
        return fit_outcome_ridge(X_train, y_train)
    elif model_type == "hurdle":
        return fit_outcome_hurdle(X_train, y_train)
    else:
        raise ValueError(f"Unknown outcome model: {model_type}")


# ─────────────────────────────────────────────────────────────────────────
# Sigma model (identical to PERSUADE)
# ─────────────────────────────────────────────────────────────────────────

def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def fit_sigma_model(
    X_sigma_all: np.ndarray,
    train_idx: np.ndarray,
    resid_sq_train: np.ndarray,
    pi: float,
    eta: float,
    rng: np.random.Generator,
    *,
    l2_beta: float = 1e-3,
    lr: float = 5e-2,
    outer_iters: int = 25,
    inner_steps: int = 150,
    rho0: float = 1.0,
    tol: float = 1e-4,
    restarts: int = 2,
    device_preference: str = "auto",
) -> np.ndarray:
    if train_idx.size == 0:
        return np.full(X_sigma_all.shape[0], (eta - pi) / (1.0 - pi), dtype=np.float32)

    if device_preference == "cpu":
        device = torch.device("cpu")
    elif device_preference == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xall = torch.tensor(X_sigma_all, dtype=torch.float32, device=device)
    Xtr = torch.tensor(X_sigma_all[train_idx], dtype=torch.float32, device=device)
    wtr = torch.tensor(resid_sq_train, dtype=torch.float32, device=device)
    d = X_sigma_all.shape[1]

    best = None
    for r in range(restarts):
        target_mean = (eta - pi) / (1.0 - pi)
        b0 = logit(target_mean)
        b = torch.tensor([b0], dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.zeros(d, dtype=torch.float32, device=device, requires_grad=True)
        if r > 0:
            seed_noise = rng.normal(0.0, 0.01, size=d).astype(np.float32)
            beta.data.copy_(torch.tensor(seed_noise, device=device))

        lam = torch.tensor(0.0, dtype=torch.float32, device=device)
        rho = rho0
        opt = torch.optim.Adam([b, beta], lr=lr)

        for _ in range(outer_iters):
            for _ in range(inner_steps):
                opt.zero_grad()
                z_tr = torch.clamp(b + Xtr @ beta, -25.0, 25.0)
                sig_tr = torch.sigmoid(z_tr)
                q_tr = pi + (1.0 - pi) * sig_tr
                obj = torch.mean(wtr / q_tr)

                z_all = torch.clamp(b + Xall @ beta, -25.0, 25.0)
                q_all = pi + (1.0 - pi) * torch.sigmoid(z_all)
                g = torch.mean(q_all) - eta
                penalty = lam * g + 0.5 * rho * torch.relu(g) ** 2
                loss = obj + penalty + l2_beta * torch.sum(beta * beta)
                loss.backward()
                opt.step()

            with torch.no_grad():
                z_all = torch.clamp(b + Xall @ beta, -25.0, 25.0)
                q_all = pi + (1.0 - pi) * torch.sigmoid(z_all)
                g_val = (torch.mean(q_all) - eta).item()
                obj_val = obj.item()
                lam = torch.clamp(lam + rho * g_val, min=0.0)
                if g_val > 5 * tol:
                    rho *= 2.0
            if abs(g_val) <= tol:
                break

        with torch.no_grad():
            z_all = torch.clamp(b + Xall @ beta, -25.0, 25.0)
            sig_all = torch.sigmoid(z_all)
            q_all = pi + (1.0 - pi) * sig_all
            g_val = (torch.mean(q_all) - eta).item()
            obj_val = torch.mean(wtr / (pi + (1.0 - pi) * torch.sigmoid(torch.clamp(b + Xtr @ beta, -25.0, 25.0)))).item()
            cand = {"sigma": sig_all.detach().cpu().numpy(), "g": g_val, "obj": obj_val}
            if best is None:
                best = cand
            else:
                best_feas = best["g"] <= tol
                cand_feas = cand["g"] <= tol
                if cand_feas and (not best_feas or cand["obj"] < best["obj"]):
                    best = cand
                elif (not best_feas) and (not cand_feas) and cand["g"] < best["g"]:
                    best = cand

    sigma = np.clip(best["sigma"], 0.0, 1.0).astype(np.float32)
    return sigma


# ─────────────────────────────────────────────────────────────────────────
# Trial
# ─────────────────────────────────────────────────────────────────────────

def run_trial(
    y: np.ndarray,
    X_full: np.ndarray,
    X_sigma: np.ndarray,
    pi: float,
    eta: float,
    rng: np.random.Generator,
    sigma_l2_beta: float,
    sigma_device: str = "auto",
    outcome_model: str = "hurdle",
) -> dict:
    n = y.shape[0]
    idx = np.arange(n)
    m_budget = max(1, int(round(eta * n)))

    # Baseline B1: direct uniform sample mean
    s_base = np.zeros(n, dtype=bool)
    base_idx = rng.choice(idx, size=m_budget, replace=False)
    s_base[base_idx] = True
    baseline1_mean = float(np.mean(y[s_base]))

    # Baseline B2: uniform sample + judge model + DR
    model_base = fit_outcome(X_full[s_base], y[s_base], outcome_model)
    f_base_all = model_base.predict(X_full).astype(np.float32)
    q_base = float(m_budget / n)
    baseline2_dr = float(np.mean(
        f_base_all + s_base.astype(np.float32) * (y - f_base_all) / np.clip(q_base, 1e-6, None)
    ))

    # Stage 1
    s1 = rng.binomial(1, pi, size=n).astype(bool)
    if s1.sum() < 30:
        add = rng.choice(idx[~s1], size=min(30 - s1.sum(), (~s1).sum()), replace=False)
        s1[add] = True

    model1 = fit_outcome(X_full[s1], y[s1], outcome_model)
    f1_all = model1.predict(X_full).astype(np.float32)
    resid_sq_s1 = (y[s1] - f1_all[s1]) ** 2
    clip = np.quantile(resid_sq_s1, 0.99) if resid_sq_s1.size > 10 else np.max(resid_sq_s1)
    resid_sq_s1 = np.clip(resid_sq_s1, 0.0, clip)

    # Sigma model
    sigma_all = fit_sigma_model(
        X_sigma_all=X_sigma,
        train_idx=np.where(s1)[0],
        resid_sq_train=resid_sq_s1.astype(np.float32),
        pi=pi, eta=eta, rng=rng,
        l2_beta=sigma_l2_beta,
        device_preference=sigma_device,
    )
    q_all = pi + (1.0 - pi) * sigma_all

    # Stage 2
    s2 = np.zeros(n, dtype=bool)
    remain = ~s1
    s2[remain] = rng.binomial(1, sigma_all[remain]).astype(bool)
    s = s1 | s2

    # Final model + DR
    model_final = fit_outcome(X_full[s], y[s], outcome_model)
    f_all = model_final.predict(X_full).astype(np.float32)

    dr = np.mean(f_all + s.astype(np.float32) * (y - f_all) / np.clip(q_all, 1e-6, None))
    true_mean = float(np.mean(y))
    naive_sample_mean = float(np.mean(y[s])) if s.any() else np.nan
    ht = float(np.mean(s.astype(np.float32) * y / np.clip(q_all, 1e-6, None)))

    return {
        "n_total": int(n),
        "m_budget_direct": int(m_budget),
        "n_stage1": int(s1.sum()),
        "n_stage2": int(s2.sum()),
        "n_sampled_total": int(s.sum()),
        "sample_rate_total": float(s.mean()),
        "target_eta": float(eta),
        "mean_q": float(q_all.mean()),
        "dr_estimate": float(dr),
        "ht_estimate": float(ht),
        "naive_sample_mean": naive_sample_mean,
        "baseline1_direct_mean_estimate": baseline1_mean,
        "baseline2_direct_dr_estimate": baseline2_dr,
        "true_mean": true_mean,
        "dr_error": float(dr - true_mean),
        "ht_error": float(ht - true_mean),
        "naive_error": float(naive_sample_mean - true_mean),
        "baseline1_direct_mean_error": float(baseline1_mean - true_mean),
        "baseline2_direct_dr_error": float(baseline2_dr - true_mean),
        "sigma_mean": float(sigma_all.mean()),
        "sigma_q10": float(np.quantile(sigma_all, 0.10)),
        "sigma_q50": float(np.quantile(sigma_all, 0.50)),
        "sigma_q90": float(np.quantile(sigma_all, 0.90)),
    }


# ─────────────────────────────────────────────────────────────────────────
# Parallel worker
# ─────────────────────────────────────────────────────────────────────────

def _init_worker(
    y, x_full, x_sigma, pi, eta, sigma_l2_beta, sigma_device, outcome_model, threads_per_worker,
):
    global _W_Y, _W_X_FULL, _W_X_SIGMA, _W_PI, _W_ETA
    global _W_SIGMA_L2_BETA, _W_SIGMA_DEVICE, _W_OUTCOME_MODEL
    if threads_per_worker and threads_per_worker > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
        if threadpool_limits is not None:
            threadpool_limits(limits=threads_per_worker)
    _W_Y = y
    _W_X_FULL = x_full
    _W_X_SIGMA = x_sigma
    _W_PI = pi
    _W_ETA = eta
    _W_SIGMA_L2_BETA = sigma_l2_beta
    _W_SIGMA_DEVICE = sigma_device
    _W_OUTCOME_MODEL = outcome_model


def _run_trial_worker(seed: int) -> dict:
    if _W_Y is None or _W_X_FULL is None or _W_X_SIGMA is None:
        raise RuntimeError("Worker not initialized.")
    trial_rng = np.random.default_rng(seed)
    return run_trial(
        y=_W_Y, X_full=_W_X_FULL, X_sigma=_W_X_SIGMA,
        pi=float(_W_PI), eta=float(_W_ETA), rng=trial_rng,
        sigma_l2_beta=float(_W_SIGMA_L2_BETA),
        sigma_device=str(_W_SIGMA_DEVICE),
        outcome_model=str(_W_OUTCOME_MODEL),
    )


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────

def summarize_trials(df: pd.DataFrame) -> dict:
    def rmse(x):
        return float(np.sqrt(np.mean(np.square(x))))

    return {
        "n_trials": int(len(df)),
        "avg_true_mean": float(df["true_mean"].mean()),
        "avg_sample_rate_total": float(df["sample_rate_total"].mean()),
        "avg_direct_budget_m": float(df["m_budget_direct"].mean()),
        "avg_n_stage1": float(df["n_stage1"].mean()),
        "avg_n_stage2": float(df["n_stage2"].mean()),
        "dr_bias": float(df["dr_error"].mean()),
        "dr_sd": float(df["dr_error"].std(ddof=1)),
        "dr_rmse": rmse(df["dr_error"].to_numpy()),
        "ht_bias": float(df["ht_error"].mean()),
        "ht_sd": float(df["ht_error"].std(ddof=1)),
        "ht_rmse": rmse(df["ht_error"].to_numpy()),
        "naive_bias": float(df["naive_error"].mean()),
        "naive_sd": float(df["naive_error"].std(ddof=1)),
        "naive_rmse": rmse(df["naive_error"].to_numpy()),
        "baseline1_direct_mean_bias": float(df["baseline1_direct_mean_error"].mean()),
        "baseline1_direct_mean_sd": float(df["baseline1_direct_mean_error"].std(ddof=1)),
        "baseline1_direct_mean_rmse": rmse(df["baseline1_direct_mean_error"].to_numpy()),
        "baseline2_direct_dr_bias": float(df["baseline2_direct_dr_error"].mean()),
        "baseline2_direct_dr_sd": float(df["baseline2_direct_dr_error"].std(ddof=1)),
        "baseline2_direct_dr_rmse": rmse(df["baseline2_direct_dr_error"].to_numpy()),
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage DR simulation on MQM.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/mqm"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--emb-path", type=Path, default=None)
    parser.add_argument("--system", type=str, default=None,
                        help="MT system to filter on. Omit for pooled.")
    parser.add_argument("--var-transform", choices=["log", "sqrt"], default="sqrt")
    parser.add_argument("--outcome-model", choices=["ridge", "hurdle"], default="hurdle")
    parser.add_argument("--pi", type=float, default=0.10)
    parser.add_argument("--eta", type=float, default=0.20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--judge-pca-components", type=int, default=None)
    parser.add_argument("--sigma-pca-components", type=int, default=None)
    parser.add_argument("--sigma-l2-beta", type=float, default=1e-3)
    parser.add_argument("--sigma-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out-prefix", type=str, default="mqm_two_stage_dr_sim")
    args = parser.parse_args()

    if not (0.0 < args.pi < 1.0):
        raise ValueError("pi must be in (0, 1)")
    if not (args.pi < args.eta < 1.0):
        raise ValueError("eta must satisfy pi < eta < 1")

    base_dir = args.base_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else (base_dir / "data")
    results_dir = args.results_dir.resolve() if args.results_dir else (base_dir / "results")
    emb_path = args.emb_path.resolve() if args.emb_path else (
        results_dir / "text_embedding_3_large_mqm_embeddings_separate.npz"
    )
    tsv_path = data_dir / "mqm_newstest2020_ende.tsv"

    judge_pca = args.judge_pca_components if args.judge_pca_components is not None else args.pca_components
    sigma_pca = args.sigma_pca_components if args.sigma_pca_components is not None else args.pca_components

    system_label = args.system or "pooled"
    print(f"System: {system_label}")
    print(f"Outcome model: {args.outcome_model}")
    print(f"Var transform: {args.var_transform}")
    print(f"pi={args.pi}, eta={args.eta}, trials={args.trials}")

    ids, y, X_full, X_sigma = build_feature_table(
        tsv_path=tsv_path,
        data_dir=data_dir,
        emb_path=emb_path,
        judge_pca_components=judge_pca,
        sigma_pca_components=sigma_pca,
        var_transform=args.var_transform,
        system_filter=args.system,
    )
    print(f"Loaded: n={len(y)}, X_full={X_full.shape}, X_sigma={X_sigma.shape}")
    print(f"y: mean={y.mean():.3f}, std={y.std():.3f}, zeros={(y==0).mean():.1%}")

    rng = np.random.default_rng(args.seed)
    trial_seeds = rng.integers(1, 10_000_000, size=args.trials).tolist()
    rows: list[dict] = []

    sigma_device = args.sigma_device
    if args.n_jobs > 1 and sigma_device == "auto" and torch.cuda.is_available():
        sigma_device = "cpu"
        print("Parallel mode: switching sigma device to CPU to avoid GPU contention.")

    if args.n_jobs <= 1:
        if args.threads_per_worker and args.threads_per_worker > 0:
            os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads_per_worker)
            if threadpool_limits is not None:
                threadpool_limits(limits=args.threads_per_worker)
        iterator = trial_seeds
        if (not args.no_progress) and (tqdm is not None):
            iterator = tqdm(iterator, total=args.trials, desc="Trials", unit="trial")
        for s in iterator:
            trial_rng = np.random.default_rng(s)
            rows.append(run_trial(
                y=y, X_full=X_full, X_sigma=X_sigma,
                pi=args.pi, eta=args.eta, rng=trial_rng,
                sigma_l2_beta=args.sigma_l2_beta,
                sigma_device=sigma_device,
                outcome_model=args.outcome_model,
            ))
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        progress = None
        if (not args.no_progress) and (tqdm is not None):
            progress = tqdm(total=args.trials, desc="Trials", unit="trial")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                y, X_full, X_sigma,
                args.pi, args.eta,
                args.sigma_l2_beta, sigma_device,
                args.outcome_model, args.threads_per_worker,
            ),
        ) as ex:
            futures = [ex.submit(_run_trial_worker, int(s)) for s in trial_seeds]
            for fut in as_completed(futures):
                rows.append(fut.result())
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys_tag = system_label.replace(".", "_")
    tag = f"{args.out_prefix}_{sys_tag}_{args.outcome_model}_pi{args.pi}_eta{args.eta}_T{args.trials}"
    trials_csv = out_dir / f"{tag}_trials.csv"
    summary_json = out_dir / f"{tag}_summary.json"

    df = pd.DataFrame(rows)
    df.to_csv(trials_csv, index=False)
    summary = summarize_trials(df)
    summary.update({
        "system": system_label,
        "outcome_model": args.outcome_model,
        "var_transform": args.var_transform,
        "pi": args.pi,
        "eta": args.eta,
        "pca_components": args.pca_components,
        "judge_pca_components": judge_pca,
        "sigma_pca_components": sigma_pca,
        "sigma_l2_beta": args.sigma_l2_beta,
        "sigma_device": sigma_device,
        "n_jobs": args.n_jobs,
        "threads_per_worker": args.threads_per_worker,
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "n_segments": int(len(y)),
        "trials_csv": str(trials_csv),
    })
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
