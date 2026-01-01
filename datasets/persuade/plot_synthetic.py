"""
Generate synthetic simulation plots (bias + MSE with bootstrap CIs on MSE).

Produces three figures:
  1. Bias & MSE vs lambda  (heteroskedasticity sweep)
  2. Bias & MSE vs pi      (pilot-rate sweep)
  3. Bias & MSE vs eta     (budget sweep)
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = [
    ("two_stage_dr", "Two-Stage DR", "#1f77b4", "o"),
    ("baseline2_one_stage_dr", "One-Stage DR", "#ff7f0e", "s"),
    ("baseline1_direct", "Direct Sample Mean", "#2ca02c", "^"),
]


def load_trials(prefix_pattern: str, param_name: str, param_values: list[float]) -> pd.DataFrame:
    all_rows = []
    for val in param_values:
        for f in glob.glob(prefix_pattern + "*_summary.json"):
            with open(f) as fh:
                d = json.load(fh)
            if abs(d[param_name] - val) < 0.001:
                df = pd.read_csv(d["trials_csv"])
                df["_param"] = val
                all_rows.append(df)
                break
    return pd.concat(all_rows, ignore_index=True)


def compute_stats(
    df: pd.DataFrame, param_col: str, n_boot: int = 2000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for val, grp in df.groupby(param_col):
        stats = {}
        for key, _, _, _ in METHODS:
            errors = grp[f"{key}_error"].values
            n = len(errors)
            mse = (errors**2).mean()
            boot_mse = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_mse[b] = (errors[idx] ** 2).mean()
            stats[key] = {
                "bias": errors.mean(),
                "mse": mse,
                "mse_lo": np.percentile(boot_mse, 2.5),
                "mse_hi": np.percentile(boot_mse, 97.5),
            }
        results[val] = stats
    return results


def make_plot(
    param_vals: list[float], stats: dict, xlabel: str, out_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xs = np.array(param_vals)

    ax = axes[0]
    for key, label, color, marker in METHODS:
        bias = np.array([stats[v][key]["bias"] for v in param_vals])
        ax.plot(xs, bias, marker=marker, color=color, label=label, linewidth=1.8, markersize=6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Bias", fontsize=12)
    ax.set_title("Bias", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="best")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    ax = axes[1]
    for key, label, color, marker in METHODS:
        mse = np.array([stats[v][key]["mse"] for v in param_vals])
        lo = np.array([stats[v][key]["mse_lo"] for v in param_vals])
        hi = np.array([stats[v][key]["mse_hi"] for v in param_vals])
        ax.plot(xs, mse, marker=marker, color=color, label=label, linewidth=1.8, markersize=6)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("MSE", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8.5, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    results_dir = "datasets/persuade/results/"

    # 1. Lambda sweep
    lam_csv = results_dir + "synthetic_pc_llm_var_exp_pc1_pi0.15_eta0.4_j32_s2_T400_trials.csv"
    lam_df = pd.read_csv(lam_csv)
    lam_df["_param"] = lam_df["lambda"]
    lam_vals = sorted(lam_df["lambda"].unique())
    lam_stats = compute_stats(lam_df, "_param")
    make_plot(lam_vals, lam_stats, r"$\lambda$",
              results_dir + "synthetic_bias_variance_mse_vs_lambda.png")

    # 2. Pi sweep
    pi_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    pi_df = load_trials(results_dir + "sweep_pi_*", "pi", pi_vals)
    pi_stats = compute_stats(pi_df, "_param")
    make_plot(pi_vals, pi_stats, r"Pilot rate $\pi$",
              results_dir + "synthetic_sweep_pi_bias_mse.png")

    # 3. Eta sweep
    eta_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    eta_df = load_trials(results_dir + "sweep_eta_*", "eta", eta_vals)
    eta_stats = compute_stats(eta_df, "_param")
    make_plot(eta_vals, eta_stats, r"Total budget $\eta$",
              results_dir + "synthetic_sweep_eta_bias_mse.png")
