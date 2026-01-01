"""
Constrained optimization on PERSUADE residual squares.

Objective:
    minimize_theta  sum_i w_i / q_i(theta)
where
    w_i = residual_i^2
    q_i(theta) = pi + (1 - pi) * sigmoid(b + x_i^T beta)

Constraint:
    mean_i q_i(theta) <= eta

Solved via augmented Lagrangian with a hinge-squared penalty for inequality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def load_data(
    residuals_csv: Path,
    embeddings_npz: Path,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    res = pd.read_csv(residuals_csv)
    if "id" not in res.columns:
        raise RuntimeError(f"'id' column missing in {residuals_csv}")

    if "residual_oof" in res.columns:
        res["residual_sq"] = res["residual_oof"] ** 2
    elif "residual_sq_oof" in res.columns:
        res["residual_sq"] = res["residual_sq_oof"]
    else:
        raise RuntimeError(
            f"Need residual column in {residuals_csv}: residual_oof or residual_sq_oof"
        )

    arr = np.load(embeddings_npz, allow_pickle=True)
    essay_ids = [str(x) for x in arr["essay_ids"]]
    emb = arr["embeddings"]

    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "id", essay_ids)

    res["id"] = res["id"].astype(str)
    merged = res[["id", "residual_sq"]].merge(emb_df, on="id", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping ids between residuals and embeddings.")

    X = merged[emb_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca_n = min(pca_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=pca_n, random_state=42)
    Xp = pca.fit_transform(X).astype(np.float32)

    w = merged["residual_sq"].values.astype(np.float32)
    ids = merged["id"].tolist()
    return Xp, w, ids


def solve_aug_lagrangian(
    X: np.ndarray,
    w: np.ndarray,
    pi: float,
    eta: float,
    *,
    l2_beta: float = 1e-4,
    lr: float = 5e-2,
    outer_iters: int = 40,
    inner_steps: int = 300,
    rho0: float = 1.0,
    tol: float = 1e-4,
    restarts: int = 3,
) -> dict:
    if not (0.0 < pi < 1.0):
        raise ValueError("pi must be in (0, 1)")
    if not (pi < eta < 1.0):
        raise ValueError("eta must satisfy pi < eta < 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    d = X.shape[1]
    best = None

    for r in range(restarts):
        target_mean = (eta - pi) / (1.0 - pi)
        b0 = logit(target_mean)
        b = torch.tensor([b0], dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.zeros(d, dtype=torch.float32, device=device, requires_grad=True)
        if r > 0:
            beta.data.normal_(mean=0.0, std=0.01)

        lam = torch.tensor(0.0, dtype=torch.float32, device=device)
        rho = rho0
        opt = torch.optim.Adam([b, beta], lr=lr)

        for _ in range(outer_iters):
            for _ in range(inner_steps):
                opt.zero_grad()
                z = b + Xt @ beta
                z = torch.clamp(z, -25.0, 25.0)
                q = pi + (1.0 - pi) * torch.sigmoid(z)
                g = torch.mean(q) - eta
                f = torch.mean(wt / q)
                penalty = lam * g + 0.5 * rho * torch.relu(g) ** 2
                loss = f + penalty + l2_beta * torch.sum(beta * beta)
                loss.backward()
                opt.step()

            with torch.no_grad():
                z = torch.clamp(b + Xt @ beta, -25.0, 25.0)
                q = pi + (1.0 - pi) * torch.sigmoid(z)
                g_val = (torch.mean(q) - eta).item()
                f_val = torch.mean(wt / q).item()
                lam = torch.clamp(lam + rho * g_val, min=0.0)
                if g_val > 5 * tol:
                    rho *= 2.0
            if abs(g_val) <= tol:
                break

        with torch.no_grad():
            z = torch.clamp(b + Xt @ beta, -25.0, 25.0)
            q = pi + (1.0 - pi) * torch.sigmoid(z)
            g_val = (torch.mean(q) - eta).item()
            f_val = torch.mean(wt / q).item()

            cand = {
                "objective_mean": float(f_val),
                "constraint_violation": float(g_val),
                "b": float(b.item()),
                "beta": beta.detach().cpu().numpy(),
                "q": q.detach().cpu().numpy(),
                "rho_final": float(rho),
                "lambda_final": float(lam.item()),
                "restart": r,
            }
            if best is None:
                best = cand
            else:
                best_feasible = best["constraint_violation"] <= tol
                cand_feasible = cand["constraint_violation"] <= tol
                if cand_feasible and (not best_feasible or cand["objective_mean"] < best["objective_mean"]):
                    best = cand
                elif (not best_feasible) and (not cand_feasible) and cand["constraint_violation"] < best["constraint_violation"]:
                    best = cand

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained weighting optimization on PERSUADE residuals.")
    parser.add_argument(
        "--residuals-csv",
        type=Path,
        default=Path("datasets/persuade/mrr_persuade_judge_oof_scores_uncertainty_emb_intercept.csv"),
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=Path("datasets/persuade/openai_text_embedding_3_large_essay_embeddings.npz"),
    )
    parser.add_argument("--pi", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--l2-beta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--outer-iters", type=int, default=40)
    parser.add_argument("--inner-steps", type=int, default=300)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="persuade_constrained_weighting",
    )
    args = parser.parse_args()

    X, w, ids = load_data(
        residuals_csv=args.residuals_csv.resolve(),
        embeddings_npz=args.embeddings_npz.resolve(),
        pca_components=args.pca_components,
    )
    best = solve_aug_lagrangian(
        X=X,
        w=w,
        pi=args.pi,
        eta=args.eta,
        l2_beta=args.l2_beta,
        lr=args.lr,
        outer_iters=args.outer_iters,
        inner_steps=args.inner_steps,
        rho0=args.rho0,
        tol=args.tol,
        restarts=args.restarts,
    )

    out_dir = Path("datasets/persuade")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.out_prefix}_pi{args.pi}_eta{args.eta}"
    sample_csv = out_dir / f"{prefix}_sample_weights.csv"
    summary_json = out_dir / f"{prefix}_summary.json"
    beta_npy = out_dir / f"{prefix}_beta.npy"

    out = pd.DataFrame(
        {
            "id": ids,
            "residual_sq": w,
            "q_hat": best["q"],
            "inv_weight_1_over_q": 1.0 / np.clip(best["q"], 1e-8, None),
        }
    )
    out.to_csv(sample_csv, index=False)
    np.save(beta_npy, best["beta"])

    summary = {
        "pi": args.pi,
        "eta": args.eta,
        "n_samples": int(len(ids)),
        "pca_components": int(X.shape[1]),
        "objective_mean": best["objective_mean"],
        "constraint_violation_mean_q_minus_eta": best["constraint_violation"],
        "mean_q_hat": float(np.mean(best["q"])),
        "min_q_hat": float(np.min(best["q"])),
        "max_q_hat": float(np.max(best["q"])),
        "b": best["b"],
        "rho_final": best["rho_final"],
        "lambda_final": best["lambda_final"],
        "restart_selected": best["restart"],
        "sample_weights_csv": str(sample_csv),
        "beta_npy": str(beta_npy),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print("Optimization done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
