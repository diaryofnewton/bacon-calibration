"""
Heteroskedastic synthetic outcome simulation on PERSUADE features.

Synthetic outcome:
    Y_i ~ Normal(mu_i, var_i)
    mu_i = sum(PC1..PC5)_i + avg_llm_score_i
    var_i = 0.1 * exp(lambda * PC1_i)

For varying lambda, compare:
  (1) Two-stage adaptive DR
  (2) Baseline 1: direct simple mean under budget eta
  (3) Baseline 2: one-stage DR with uniform direct sample
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


ALPHAS = np.logspace(-3, 4, 20)
_W_DATA: dict[float, dict[str, np.ndarray]] | None = None
_W_PI: float | None = None
_W_ETA: float | None = None
_W_SIGMA_L2_BETA: float | None = None
_W_SIGMA_DEVICE: str | None = None


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def parse_lambda_values(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("No lambda values provided.")
    return vals


def load_avg_llm_score(llm_path: Path) -> pd.DataFrame:
    rows = []
    with llm_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            score = rec.get("predicted_score")
            if score is None:
                continue
            rows.append({"id": str(rec["essay_id_comp"]), "score": float(score)})

    if not rows:
        raise RuntimeError(f"No valid predicted scores found in {llm_path}")
    df = pd.DataFrame(rows)
    out = df.groupby("id", as_index=False)["score"].mean().rename(columns={"score": "avg_llm_score"})
    return out


def build_feature_tables(
    data_dir: Path,
    results_dir: Path,
    judge_pca_components: int,
    sigma_pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    emb_path = results_dir / "openai_text_embedding_3_large_essay_embeddings.npz"
    llm_path = results_dir / "llm_scores.jsonl"
    if not emb_path.exists():
        alt = data_dir / "openai_text_embedding_3_large_essay_embeddings.npz"
        if alt.exists():
            emb_path = alt
    if not llm_path.exists():
        alt = data_dir / "llm_scores.jsonl"
        if alt.exists():
            llm_path = alt

    emb = np.load(emb_path, allow_pickle=True)
    emb_ids = np.array([str(x) for x in emb["essay_ids"]], dtype=object)
    Xemb = emb["embeddings"].astype(np.float32)
    emb_df = pd.DataFrame(Xemb)
    emb_df.insert(0, "id", emb_ids.tolist())

    llm_df = load_avg_llm_score(llm_path)
    merged = emb_df.merge(llm_df, on="id", how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between embeddings and llm scores.")

    ids = merged["id"].astype(str).to_numpy()
    avg_llm = merged["avg_llm_score"].to_numpy(dtype=np.float32)
    emb_cols = [c for c in merged.columns if isinstance(c, int)]
    X = merged[emb_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xz = StandardScaler().fit_transform(X).astype(np.float32)

    judge_n = min(judge_pca_components, Xz.shape[1], Xz.shape[0] - 1)
    sigma_n = min(sigma_pca_components, Xz.shape[1], Xz.shape[0] - 1)
    if judge_n < 1 or sigma_n < 1:
        raise RuntimeError("PCA components are too small for available data.")

    X_judge = PCA(n_components=judge_n, random_state=seed).fit_transform(Xz)
    X_judge = StandardScaler().fit_transform(X_judge).astype(np.float32)
    X_sigma = PCA(n_components=sigma_n, random_state=seed + 1).fit_transform(Xz)
    X_sigma = StandardScaler().fit_transform(X_sigma).astype(np.float32)
    return ids, avg_llm, X_judge, X_sigma


def simulate_outcome(avg_llm: np.ndarray, X_judge: np.ndarray, lam: float, seed: int) -> np.ndarray:
    pc_count = min(5, X_judge.shape[1])
    mu = X_judge[:, :pc_count].sum(axis=1, dtype=np.float64) + avg_llm.astype(np.float64)
    pc1 = X_judge[:, 0].astype(np.float64)
    exp_arg = np.clip(lam * pc1, -50.0, 50.0)
    var = 0.1 * np.exp(exp_arg)
    sd = np.sqrt(np.clip(var, 1e-10, 1e8))
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=mu, scale=sd).astype(np.float32)
    return y


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
        for _ in range(outer_iters):
            for _ in range(inner_steps):
                if b.grad is not None:
                    b.grad.zero_()
                if beta.grad is not None:
                    beta.grad.zero_()
                z_tr = torch.clamp(b + Xtr @ beta, -25.0, 25.0)
                q_tr = pi + (1.0 - pi) * torch.sigmoid(z_tr)
                obj = torch.mean(wtr / q_tr)

                z_all = torch.clamp(b + Xall @ beta, -25.0, 25.0)
                q_all = pi + (1.0 - pi) * torch.sigmoid(z_all)
                g = torch.mean(q_all) - eta
                penalty = lam * g + 0.5 * rho * torch.relu(g) ** 2
                loss = obj + penalty + l2_beta * torch.sum(beta * beta)
                loss.backward()
                with torch.no_grad():
                    b -= lr * b.grad
                    beta -= lr * beta.grad

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

    # Baseline 1: simple direct estimator.
    s_base = np.zeros(n, dtype=bool)
    base_idx = rng.choice(idx, size=m_budget, replace=False)
    s_base[base_idx] = True
    baseline1_mean = float(np.mean(y[s_base]))

    # Baseline 2: one-stage DR with uniform propensity q = m / N.
    model_base = fit_outcome_ridge(X_full[s_base], y[s_base])
    f_base = model_base.predict(X_full).astype(np.float32)
    q_base = float(m_budget / n)
    baseline2_dr = float(
        np.mean(f_base + s_base.astype(np.float32) * (y - f_base) / np.clip(q_base, 1e-6, None))
    )

    # Two-stage DR.
    s1 = rng.binomial(1, pi, size=n).astype(bool)
    if s1.sum() < 30:
        add = rng.choice(idx[~s1], size=min(30 - s1.sum(), (~s1).sum()), replace=False)
        s1[add] = True

    model1 = fit_outcome_ridge(X_full[s1], y[s1])
    f1 = model1.predict(X_full).astype(np.float32)
    resid_sq = (y[s1] - f1[s1]) ** 2
    if resid_sq.size > 10:
        resid_sq = np.clip(resid_sq, 0.0, np.quantile(resid_sq, 0.99))

    sigma_all = fit_sigma_model(
        X_sigma_all=X_sigma,
        train_idx=np.where(s1)[0],
        resid_sq_train=resid_sq.astype(np.float32),
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
    f = model_final.predict(X_full).astype(np.float32)
    dr = float(np.mean(f + s.astype(np.float32) * (y - f) / np.clip(q_all, 1e-6, None)))

    true_mean = float(np.mean(y))
    return {
        "true_mean": true_mean,
        "two_stage_dr_estimate": dr,
        "baseline1_direct_estimate": baseline1_mean,
        "baseline2_one_stage_dr_estimate": baseline2_dr,
        "two_stage_dr_error": float(dr - true_mean),
        "baseline1_direct_error": float(baseline1_mean - true_mean),
        "baseline2_one_stage_dr_error": float(baseline2_dr - true_mean),
        "n_stage1": int(s1.sum()),
        "n_stage2": int(s2.sum()),
        "sample_rate_total": float(s.mean()),
        "mean_q": float(np.mean(q_all)),
    }


def _init_worker(
    data_by_lambda: dict[float, dict[str, np.ndarray]],
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
    _W_DATA = data_by_lambda
    _W_PI = pi
    _W_ETA = eta
    _W_SIGMA_L2_BETA = sigma_l2_beta
    _W_SIGMA_DEVICE = sigma_device


def _run_trial_worker(seed: int, lam: float) -> dict:
    if _W_DATA is None:
        raise RuntimeError("Worker not initialized.")
    d = _W_DATA[lam]
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
    row["lambda"] = float(lam)
    return row


def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    def _rmse(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        return float(np.sqrt(np.mean(np.square(arr))))

    rows = []
    for lam, d in df.groupby("lambda"):
        rows.append(
            {
                "lambda": float(lam),
                "n_trials": int(len(d)),
                "two_stage_dr_bias": float(d["two_stage_dr_error"].mean()),
                "two_stage_dr_sd": float(d["two_stage_dr_error"].std(ddof=1)),
                "two_stage_dr_rmse": _rmse(d["two_stage_dr_error"]),
                "baseline1_direct_bias": float(d["baseline1_direct_error"].mean()),
                "baseline1_direct_sd": float(d["baseline1_direct_error"].std(ddof=1)),
                "baseline1_direct_rmse": _rmse(d["baseline1_direct_error"]),
                "baseline2_one_stage_dr_bias": float(d["baseline2_one_stage_dr_error"].mean()),
                "baseline2_one_stage_dr_sd": float(d["baseline2_one_stage_dr_error"].std(ddof=1)),
                "baseline2_one_stage_dr_rmse": _rmse(d["baseline2_one_stage_dr_error"]),
                "avg_sample_rate_total": float(d["sample_rate_total"].mean()),
                "avg_n_stage1": float(d["n_stage1"].mean()),
                "avg_n_stage2": float(d["n_stage2"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic heteroskedastic DR simulation on PERSUADE.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/persuade"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--pi", type=float, default=0.15)
    parser.add_argument("--eta", type=float, default=0.40)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-pca-components", type=int, default=32)
    parser.add_argument("--sigma-pca-components", type=int, default=16)
    parser.add_argument("--lambda-values", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--sigma-l2-beta", type=float, default=1e-2)
    parser.add_argument("--sigma-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--out-prefix", type=str, default="synthetic_pc_llm_var_exp_pc1")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.pi < 1.0):
        raise ValueError("pi must be in (0, 1)")
    if not (args.pi < args.eta < 1.0):
        raise ValueError("eta must satisfy pi < eta < 1")

    lambdas = parse_lambda_values(args.lambda_values)
    base_dir = args.base_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else (base_dir / "data")
    results_dir = args.results_dir.resolve() if args.results_dir else (base_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ids, avg_llm, X_judge, X_sigma = build_feature_tables(
        data_dir=data_dir,
        results_dir=results_dir,
        judge_pca_components=args.judge_pca_components,
        sigma_pca_components=args.sigma_pca_components,
        seed=args.seed,
    )
    _ = ids

    # Outcome model uses judge PCs plus average LLM score.
    X_full = np.concatenate([X_judge, avg_llm.reshape(-1, 1)], axis=1).astype(np.float32)

    data_by_lambda: dict[float, dict[str, np.ndarray]] = {}
    for i, lam in enumerate(lambdas):
        y = simulate_outcome(avg_llm=avg_llm, X_judge=X_judge, lam=lam, seed=args.seed + 1000 + i)
        data_by_lambda[lam] = {"y": y, "X_full": X_full, "X_sigma": X_sigma}

    sigma_device = args.sigma_device
    if args.n_jobs > 1 and sigma_device == "auto" and torch.cuda.is_available():
        sigma_device = "cpu"
        print("Parallel mode detected: switching sigma device to CPU to avoid GPU contention.")

    seed_rng = np.random.default_rng(args.seed)
    jobs: list[tuple[int, float]] = []
    for lam in lambdas:
        seeds = seed_rng.integers(1, 10_000_000, size=args.trials).tolist()
        jobs.extend([(int(s), float(lam)) for s in seeds])

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
            iterator = tqdm(iterator, total=len(jobs), desc="Synthetic trials", unit="trial")
        _init_worker(data_by_lambda, args.pi, args.eta, args.sigma_l2_beta, sigma_device, args.threads_per_worker)
        for seed, lam in iterator:
            rows.append(_run_trial_worker(seed, lam))
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        progress = None
        if (not args.no_progress) and (tqdm is not None):
            progress = tqdm(total=len(jobs), desc="Synthetic trials", unit="trial")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                data_by_lambda,
                args.pi,
                args.eta,
                args.sigma_l2_beta,
                sigma_device,
                args.threads_per_worker,
            ),
        ) as ex:
            futures = [ex.submit(_run_trial_worker, seed, lam) for seed, lam in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    trials_df = pd.DataFrame(rows)
    summary_df = summarize_trials(trials_df)

    tag = (
        f"{args.out_prefix}_pi{args.pi}_eta{args.eta}_"
        f"j{args.judge_pca_components}_s{args.sigma_pca_components}_T{args.trials}"
    )
    trials_csv = results_dir / f"{tag}_trials.csv"
    summary_csv = results_dir / f"{tag}_summary_by_lambda.csv"
    summary_json = results_dir / f"{tag}_summary.json"
    trials_df.to_csv(trials_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    summary = {
        "pi": args.pi,
        "eta": args.eta,
        "trials_per_lambda": args.trials,
        "seed": args.seed,
        "judge_pca_components": args.judge_pca_components,
        "sigma_pca_components": args.sigma_pca_components,
        "lambda_values": lambdas,
        "sigma_l2_beta": args.sigma_l2_beta,
        "sigma_device": sigma_device,
        "n_jobs": args.n_jobs,
        "threads_per_worker": args.threads_per_worker,
        "n_items": int(X_full.shape[0]),
        "summary_by_lambda": summary_df.to_dict(orient="records"),
        "trials_csv": str(trials_csv),
        "summary_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved: {trials_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()

