"""
Two-stage adaptive sampling simulation for mean outcome estimation on WebDesign.

Design per outcome:
1) Stage-1 label sampling: S1_i ~ Bernoulli(pi)
2) Fit outcome model f using labeled stage-1 items
3) Fit adaptive stage-2 propensity model sigma(x) under budget eta
4) Stage-2 sampling on unlabeled pool: S2_i ~ Bernoulli(sigma_i) if S1_i=0
5) Refit outcome model on all sampled items and estimate mean via DR:

    mu_hat = mean_i [ f_i + S_i * (Y_i - f_i) / q_i ]
    with q_i = pi + (1-pi)*sigma_i

Baselines per trial:
    (B1) Uniformly sample exactly round(eta * N) items, report sample mean.
    (B2) Same uniform sample size, fit outcome model and compute DR with q_i = m/N.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


OUTCOMES = ["AE", "TRU", "TYP", "EXMPL", "AVG", "US"]
ALPHAS = np.logspace(-3, 4, 20)

DOMAIN_CONFIG = {
    "unis": {
        "folders": ["Universitites", "Universities"],
        "ratings_avg": "ratings.avg.unis.txt",
        "vlm_scores": "vlm_scores_unis.jsonl",
        "file_prefix": "webdesign_unis_two_stage_dr_sim",
    },
    "banks": {
        "folders": ["Commercial Banks"],
        "ratings_avg": "ratings.avg.banks.txt",
        "vlm_scores": "vlm_scores_banks.jsonl",
        "file_prefix": "webdesign_banks_two_stage_dr_sim",
    },
    "fashion": {
        "folders": ["eCommerce"],
        "ratings_avg": "ratings.avg.fashion.txt",
        "vlm_scores": "vlm_scores_fashion.jsonl",
        "file_prefix": "webdesign_fashion_two_stage_dr_sim",
    },
    "homeware": {
        "folders": ["eCommerce"],
        "ratings_avg": "ratings.avg.homeware.txt",
        "vlm_scores": "vlm_scores_homeware.jsonl",
        "file_prefix": "webdesign_homeware_two_stage_dr_sim",
    },
}


_W_DATA: dict[str, dict[str, np.ndarray]] | None = None
_W_PI: float | None = None
_W_ETA: float | None = None
_W_SIGMA_L2_BETA: float | None = None
_W_SIGMA_DEVICE: str | None = None


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def resolve_domain_dir(base_dir: Path, folders: list[str]) -> Path:
    for name in folders:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find domain folder under {base_dir} among: {folders}")


def load_human_avg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    req = {"stimulusId", *OUTCOMES}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in human ratings: {sorted(missing)}")
    out = df[["stimulusId", *OUTCOMES]].copy()
    out["stimulusId"] = out["stimulusId"].astype(str)
    return out


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
    df["stimulusId"] = df["stimulusId"].astype(str)
    return df.reset_index(drop=True)


def load_siglip_embeddings(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    paths = arr["paths"]
    emb = arr["embeddings"]
    stimulus = [Path(str(p)).name for p in paths]
    cols = [f"siglip__{i}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=cols)
    out.insert(0, "stimulusId", stimulus)
    out["stimulusId"] = out["stimulusId"].astype(str)
    out = out.drop_duplicates("stimulusId")
    return out


def make_ridge_pipe() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )


def fit_outcome_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = make_ridge_pipe()
    model.fit(X_train, y_train)
    return model


def build_outcome_table(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    siglip_df: pd.DataFrame,
    outcome: str,
    include_sigma_vlm_features: bool,
    judge_siglip_pca_components: int,
    sigma_siglip_pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_col = f"score_{outcome}"
    pivot_score = llm_df.pivot_table(index="stimulusId", columns="model", values=score_col, aggfunc="mean")
    model_names = sorted(pivot_score.columns.tolist())
    pivot_score.columns = [f"s__{m}" for m in model_names]
    pivot_score = pivot_score.reset_index()

    pivot_entropy = llm_df.pivot_table(index="stimulusId", columns="model", values="mean_entropy", aggfunc="mean")
    pivot_entropy.columns = [f"entropy__{m}" for m in pivot_entropy.columns]
    pivot_entropy = pivot_entropy.reset_index()

    pivot_ppl = llm_df.pivot_table(index="stimulusId", columns="model", values="perplexity", aggfunc="mean")
    pivot_ppl.columns = [f"ppl__{m}" for m in pivot_ppl.columns]
    pivot_ppl = pivot_ppl.reset_index()

    merged = (
        human_df[["stimulusId", outcome]]
        .rename(columns={outcome: "y"})
        .merge(pivot_score, on="stimulusId", how="inner")
        .merge(pivot_entropy, on="stimulusId", how="left")
        .merge(pivot_ppl, on="stimulusId", how="left")
        .merge(siglip_df, on="stimulusId", how="inner")
    )

    score_cols = [f"s__{m}" for m in model_names]
    judge_features: list[str] = []
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
        judge_features.extend([s_col, sxe_col, sxlp_col])

    siglip_cols = [c for c in merged.columns if c.startswith("siglip__")]
    Xsig = merged[siglip_cols].to_numpy(dtype=np.float32)
    Xsig = np.nan_to_num(Xsig, nan=0.0, posinf=0.0, neginf=0.0)
    Xsig_z = StandardScaler().fit_transform(Xsig)

    judge_pca_n = min(judge_siglip_pca_components, Xsig_z.shape[1], Xsig_z.shape[0] - 1)
    judge_pca = PCA(n_components=judge_pca_n, random_state=seed)
    Xsig_judge = judge_pca.fit_transform(Xsig_z).astype(np.float32)
    judge_sig_cols = [f"sigj__{i}" for i in range(Xsig_judge.shape[1])]
    for i, c in enumerate(judge_sig_cols):
        merged[c] = Xsig_judge[:, i]
    judge_features.extend(judge_sig_cols)

    sigma_pca_n = min(sigma_siglip_pca_components, Xsig_z.shape[1], Xsig_z.shape[0] - 1)
    sigma_pca = PCA(n_components=sigma_pca_n, random_state=seed)
    Xsig_sigma = sigma_pca.fit_transform(Xsig_z).astype(np.float32)
    sigma_features = [f"sigs__{i}" for i in range(Xsig_sigma.shape[1])]
    for i, c in enumerate(sigma_features):
        merged[c] = Xsig_sigma[:, i]

    if include_sigma_vlm_features:
        sigma_features.extend(score_cols)

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged = merged.dropna(subset=["y"]).copy()

    y = merged["y"].astype(np.float32).to_numpy()
    X_full = merged[judge_features].to_numpy(dtype=np.float32)
    X_sigma = merged[sigma_features].to_numpy(dtype=np.float32)

    return merged["stimulusId"].astype(str).to_numpy(), y, X_full, X_sigma


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
            noise = rng.normal(0.0, 0.01, size=d).astype(np.float32)
            beta.data.copy_(torch.tensor(noise, device=device))

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
                lam = torch.clamp(lam + rho * g_val, min=0.0)
                if g_val > 5 * tol:
                    rho *= 2.0
            if abs(g_val) <= tol:
                break

        with torch.no_grad():
            z_all = torch.clamp(b + Xall @ beta, -25.0, 25.0)
            sig_all = torch.sigmoid(z_all)
            g_val = (torch.mean(pi + (1.0 - pi) * sig_all) - eta).item()
            obj_val = torch.mean(
                wtr / (pi + (1.0 - pi) * torch.sigmoid(torch.clamp(b + Xtr @ beta, -25.0, 25.0)))
            ).item()
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

    return np.clip(best["sigma"], 0.0, 1.0).astype(np.float32)


def run_trial(
    y: np.ndarray,
    X_full: np.ndarray,
    X_sigma: np.ndarray,
    pi: float,
    eta: float,
    rng: np.random.Generator,
    sigma_l2_beta: float,
    sigma_device: str = "auto",
) -> dict:
    n = y.shape[0]
    idx = np.arange(n)
    m_budget = max(1, int(round(eta * n)))

    s_base = np.zeros(n, dtype=bool)
    base_idx = rng.choice(idx, size=m_budget, replace=False)
    s_base[base_idx] = True
    baseline1_mean = float(np.mean(y[s_base]))

    model_base = fit_outcome_ridge(X_full[s_base], y[s_base])
    f_base_all = model_base.predict(X_full).astype(np.float32)
    q_base = float(m_budget / n)
    baseline2_dr = float(
        np.mean(f_base_all + s_base.astype(np.float32) * (y - f_base_all) / np.clip(q_base, 1e-6, None))
    )

    s1 = rng.binomial(1, pi, size=n).astype(bool)
    if s1.sum() < 20:
        add = rng.choice(idx[~s1], size=min(20 - s1.sum(), (~s1).sum()), replace=False)
        s1[add] = True

    model1 = fit_outcome_ridge(X_full[s1], y[s1])
    f1_all = model1.predict(X_full).astype(np.float32)
    resid_sq_s1 = (y[s1] - f1_all[s1]) ** 2
    clip = np.quantile(resid_sq_s1, 0.99) if resid_sq_s1.size > 10 else np.max(resid_sq_s1)
    resid_sq_s1 = np.clip(resid_sq_s1, 0.0, clip)

    sigma_all = fit_sigma_model(
        X_sigma_all=X_sigma,
        train_idx=np.where(s1)[0],
        resid_sq_train=resid_sq_s1.astype(np.float32),
        pi=pi,
        eta=eta,
        rng=rng,
        l2_beta=sigma_l2_beta,
        device_preference=sigma_device,
    )
    q_all = pi + (1.0 - pi) * sigma_all

    s2 = np.zeros(n, dtype=bool)
    remain = ~s1
    s2[remain] = rng.binomial(1, sigma_all[remain]).astype(bool)
    s = s1 | s2

    model_final = fit_outcome_ridge(X_full[s], y[s])
    f_all = model_final.predict(X_full).astype(np.float32)

    dr = float(np.mean(f_all + s.astype(np.float32) * (y - f_all) / np.clip(q_all, 1e-6, None)))
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
        "dr_estimate": dr,
        "ht_estimate": ht,
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


def _init_worker(
    data_by_outcome: dict[str, dict[str, np.ndarray]],
    pi: float,
    eta: float,
    sigma_l2_beta: float,
    sigma_device: str,
    threads_per_worker: int,
) -> None:
    global _W_DATA, _W_PI, _W_ETA, _W_SIGMA_L2_BETA, _W_SIGMA_DEVICE
    if threads_per_worker and threads_per_worker > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
        if threadpool_limits is not None:
            threadpool_limits(limits=threads_per_worker)

    _W_DATA = data_by_outcome
    _W_PI = pi
    _W_ETA = eta
    _W_SIGMA_L2_BETA = sigma_l2_beta
    _W_SIGMA_DEVICE = sigma_device


def _run_trial_worker(seed: int, outcome: str) -> dict:
    if _W_DATA is None:
        raise RuntimeError("Worker not initialized.")
    d = _W_DATA[outcome]
    rng = np.random.default_rng(seed)
    row = run_trial(
        y=d["y"],
        X_full=d["X_full"],
        X_sigma=d["X_sigma"],
        pi=float(_W_PI),
        eta=float(_W_ETA),
        rng=rng,
        sigma_l2_beta=float(_W_SIGMA_L2_BETA),
        sigma_device=str(_W_SIGMA_DEVICE),
    )
    row["outcome"] = outcome
    return row


def summarize_trials(df: pd.DataFrame) -> dict:
    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    out = {
        "n_trials_total": int(len(df)),
        "dr_rmse": rmse(df["dr_error"].to_numpy()),
        "ht_rmse": rmse(df["ht_error"].to_numpy()),
        "naive_rmse": rmse(df["naive_error"].to_numpy()),
        "baseline1_direct_mean_rmse": rmse(df["baseline1_direct_mean_error"].to_numpy()),
        "baseline2_direct_dr_rmse": rmse(df["baseline2_direct_dr_error"].to_numpy()),
        "avg_sample_rate_total": float(df["sample_rate_total"].mean()),
        "avg_n_stage1": float(df["n_stage1"].mean()),
        "avg_n_stage2": float(df["n_stage2"].mean()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage DR simulation on WebDesign.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/WebDesign"))
    parser.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()), default="unis")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--pi", type=float, default=0.10)
    parser.add_argument("--eta", type=float, default=0.40)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-siglip-pca-components", type=int, default=64)
    parser.add_argument("--sigma-siglip-pca-components", type=int, default=32)
    parser.add_argument("--include-sigma-vlm-features", action="store_true")
    parser.add_argument("--sigma-l2-beta", type=float, default=1e-3)
    parser.add_argument("--sigma-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for Monte Carlo tasks.")
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="BLAS/OpenMP thread cap per process (recommended 1 for parallel mode).",
    )
    parser.add_argument("--out-prefix", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.pi < 1.0):
        raise ValueError("pi must be in (0, 1)")
    if not (args.pi < args.eta < 1.0):
        raise ValueError("eta must satisfy pi < eta < 1")

    cfg = DOMAIN_CONFIG[args.domain]
    base_dir = args.base_dir.resolve()
    domain_dir = resolve_domain_dir(base_dir, cfg["folders"])
    data_dir = domain_dir / "data"
    results_dir = domain_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    human = load_human_avg(data_dir / cfg["ratings_avg"])
    vlm = load_vlm_scores(results_dir / cfg["vlm_scores"], models=args.models)
    siglip = load_siglip_embeddings(results_dir / "siglip_google_siglip_base_patch16_224_embeddings.npz")

    data_by_outcome: dict[str, dict[str, np.ndarray]] = {}
    n_by_outcome: dict[str, int] = {}
    for outcome in OUTCOMES:
        ids, y, X_full, X_sigma = build_outcome_table(
            human_df=human,
            llm_df=vlm,
            siglip_df=siglip,
            outcome=outcome,
            include_sigma_vlm_features=args.include_sigma_vlm_features,
            judge_siglip_pca_components=args.judge_siglip_pca_components,
            sigma_siglip_pca_components=args.sigma_siglip_pca_components,
            seed=args.seed,
        )
        _ = ids
        data_by_outcome[outcome] = {"y": y, "X_full": X_full, "X_sigma": X_sigma}
        n_by_outcome[outcome] = int(len(y))

    sigma_device = args.sigma_device
    if args.n_jobs > 1 and sigma_device == "auto" and torch.cuda.is_available():
        sigma_device = "cpu"
        print("Parallel mode detected: switching sigma device to CPU to avoid GPU contention.")

    seed_rng = np.random.default_rng(args.seed)
    jobs: list[tuple[int, str]] = []
    for outcome in OUTCOMES:
        seeds = seed_rng.integers(1, 10_000_000, size=args.trials).tolist()
        jobs.extend([(int(s), outcome) for s in seeds])

    rows: list[dict] = []
    if args.n_jobs <= 1:
        if args.threads_per_worker and args.threads_per_worker > 0:
            os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["MKL_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_worker)
            os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads_per_worker)
            if threadpool_limits is not None:
                threadpool_limits(limits=args.threads_per_worker)

        iterator = jobs
        if (not args.no_progress) and (tqdm is not None):
            iterator = tqdm(iterator, total=len(jobs), desc="Simulation tasks", unit="task")

        _init_worker(data_by_outcome, args.pi, args.eta, args.sigma_l2_beta, sigma_device, args.threads_per_worker)
        for seed, outcome in iterator:
            rows.append(_run_trial_worker(seed, outcome))
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        progress = None
        if (not args.no_progress) and (tqdm is not None):
            progress = tqdm(total=len(jobs), desc="Simulation tasks", unit="task")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                data_by_outcome,
                args.pi,
                args.eta,
                args.sigma_l2_beta,
                sigma_device,
                args.threads_per_worker,
            ),
        ) as ex:
            futures = [ex.submit(_run_trial_worker, seed, outcome) for seed, outcome in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    df = pd.DataFrame(rows)
    per_outcome = []
    for outcome in OUTCOMES:
        dfo = df[df["outcome"] == outcome].copy()
        s = summarize_trials(dfo)
        s["outcome"] = outcome
        s["n_items"] = n_by_outcome[outcome]
        per_outcome.append(s)
    per_outcome_df = pd.DataFrame(per_outcome)

    overall = summarize_trials(df)
    overall.update(
        {
            "domain": args.domain,
            "resolved_domain_dir": str(domain_dir),
            "pi": args.pi,
            "eta": args.eta,
            "trials_per_outcome": args.trials,
            "seed": args.seed,
            "judge_siglip_pca_components": args.judge_siglip_pca_components,
            "sigma_siglip_pca_components": args.sigma_siglip_pca_components,
            "include_sigma_vlm_features": bool(args.include_sigma_vlm_features),
            "sigma_l2_beta": args.sigma_l2_beta,
            "sigma_device": sigma_device,
            "n_jobs": args.n_jobs,
            "threads_per_worker": args.threads_per_worker,
            "models_used": sorted(vlm["model"].dropna().unique().tolist()),
            "outcomes": OUTCOMES,
        }
    )

    prefix = args.out_prefix if args.out_prefix else cfg["file_prefix"]
    tag = f"{prefix}_pi{args.pi}_eta{args.eta}_T{args.trials}"
    trials_csv = results_dir / f"{tag}_trials.csv"
    per_outcome_csv = results_dir / f"{tag}_summary_by_outcome.csv"
    summary_json = results_dir / f"{tag}_summary.json"
    df.to_csv(trials_csv, index=False)
    per_outcome_df.to_csv(per_outcome_csv, index=False)
    summary_json.write_text(json.dumps(overall, indent=2))

    print(json.dumps(overall, indent=2))
    print(f"Saved: {trials_csv}")
    print(f"Saved: {per_outcome_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
