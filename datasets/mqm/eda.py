"""
EDA for LLM-as-MQM-Judge annotations on WMT newstest2020 en-de.

Data: LLM-generated MQM error annotations for the 10 MT systems in
      newstest2020 English->German, following the annotation scheme from
      https://github.com/google/wmt-mqm-human-evaluation

MQM weighting (Freitag et al., 2021):
  - Minor         -> 1
  - Major         -> 5
  - Non-translation -> 25
  - Minor Fluency/Punctuation -> 0.1
"""

import json, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

HUMAN_MQM_SCORES = {
    "Human-B.0": 0.75,
    "Human-A.0": 0.91,
    "Human-P.0": 1.41,
    "Tohoku-AIP-NTT.396": 2.02,
    "OPPO.722": 2.25,
    "eTranslation.737": 2.33,
    "Tencent_Translation.684": 2.35,
    "Huoshan_Translate.832": 2.45,
    "Online-B.1590": 2.48,
    "Online-A.1576": 2.99,
}

# ─────────────────────────────────────────────────────────────────────────
# 1. Loading
# ─────────────────────────────────────────────────────────────────────────

def load_all_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for fpath in sorted(data_dir.glob("*.jsonl")):
        t0 = time.time()
        records = [json.loads(line) for line in open(fpath)]
        df = pd.DataFrame(records)
        df["source_file"] = fpath.name
        frames.append(df)
        print(f"  loaded {fpath.name} ({len(df):,} rows) in {time.time()-t0:.1f}s")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────
# 2. Vectorized MQM helpers
# ─────────────────────────────────────────────────────────────────────────

def _extract_errors_list(mqm_col: pd.Series) -> pd.Series:
    """Extract the errors list from the mqm dict column."""
    return mqm_col.apply(
        lambda x: x.get("errors", []) if isinstance(x, dict) else []
    )


def compute_mqm_scores_vectorized(df: pd.DataFrame) -> pd.Series:
    """Compute MQM score per row using vectorized explode + merge."""
    errors_lists = _extract_errors_list(df["mqm"])

    has_errors = errors_lists.apply(len) > 0
    if not has_errors.any():
        return pd.Series(0.0, index=df.index)

    err_series = errors_lists[has_errors].explode().dropna()
    if err_series.empty:
        return pd.Series(0.0, index=df.index)

    err_df = pd.DataFrame(err_series.tolist(), index=err_series.index)
    err_df.columns = [c.lower() for c in err_df.columns]

    if "category" not in err_df.columns or "severity" not in err_df.columns:
        return pd.Series(0.0, index=df.index)

    cat_lower = err_df["category"].str.strip().str.lower()
    sev_lower = err_df["severity"].str.strip().str.lower()

    weight = pd.Series(0.0, index=err_df.index)
    weight[sev_lower == "minor"] = 1.0
    weight[sev_lower == "major"] = 5.0
    weight[(sev_lower == "minor") & cat_lower.str.startswith("fluency/punctuation")] = 0.1
    weight[cat_lower.str.contains("non-translation", na=False)] = 25.0

    scores = weight.groupby(level=0).sum()
    result = pd.Series(0.0, index=df.index)
    result.loc[scores.index] = scores.values
    return result


def extract_error_rows_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Explode mqm.errors using vectorized operations."""
    errors_lists = _extract_errors_list(df["mqm"])

    keep_cols = ["model", "system", "seg_id", "doc_id", "rater"]
    base = df[keep_cols].copy()
    base["_errors"] = errors_lists

    has_errors = base["_errors"].apply(len) > 0
    base = base[has_errors].copy()

    if base.empty:
        return pd.DataFrame(columns=keep_cols + ["category", "severity", "span"])

    base = base.explode("_errors").reset_index(drop=True)
    err_detail = pd.DataFrame(base["_errors"].tolist())
    err_detail.columns = [c.lower() for c in err_detail.columns]

    for col in ["category", "severity", "span"]:
        if col not in err_detail.columns:
            err_detail[col] = "Unknown"

    result = pd.concat([base[keep_cols].reset_index(drop=True),
                        err_detail[["category", "severity", "span"]]], axis=1)
    return result


# ─────────────────────────────────────────────────────────────────────────
# 3. EDA sections
# ─────────────────────────────────────────────────────────────────────────

def section_basic_stats(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 1 - Basic data overview")
    print("=" * 70)
    print(f"  Total rows              : {len(df):,}")
    print(f"  Source files             : {df['source_file'].nunique()}")

    file_counts = df["source_file"].value_counts()
    for f in sorted(file_counts.index):
        print(f"    {f:55s} {file_counts[f]:>7,} rows")

    model_counts = df["model"].value_counts()
    print(f"  Unique LLM models       : {len(model_counts)}")
    for m in sorted(model_counts.index):
        print(f"    {m:30s} ({model_counts[m]:>7,} rows)")

    print(f"  Unique MT systems        : {df['system'].nunique()}")
    print(f"  Unique segment IDs       : {df['seg_id'].nunique()}")
    print(f"  Unique raters            : {df['rater'].nunique()}  {sorted(df['rater'].unique())}")
    print(f"  Unique (system, seg_id)  : {df.groupby(['system', 'seg_id']).ngroups:,}")

    reps = df.groupby(["segment_id", "model"]).size()
    print(f"\n  Rows per (segment_id, model):")
    print(f"    mean={reps.mean():.2f}  median={reps.median():.0f}  "
          f"min={reps.min()}  max={reps.max()}")

    n_errors = df["mqm"].apply(
        lambda x: len(x.get("errors", [])) if isinstance(x, dict) else 0
    )
    print(f"\n  Errors per annotation row:")
    print(f"    mean={n_errors.mean():.2f}  median={n_errors.median():.0f}  max={n_errors.max()}")
    print(f"    fraction with 0 errors : {(n_errors == 0).mean():.1%}")

    for c in ["prompt_tokens", "completion_tokens"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                print(f"  {c}: mean={vals.mean():.0f}  median={vals.median():.0f}  "
                      f"min={vals.min():.0f}  max={vals.max():.0f}")
    print()


def section_error_distribution(err_df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 2 - Error distribution")
    print("=" * 70)

    n = len(err_df)
    sev_counts = err_df["severity"].value_counts()
    print("\n  (a) By severity:")
    for sev, cnt in sev_counts.items():
        print(f"    {sev:15s} {cnt:>8,}  ({cnt / n:.1%})")

    err_df = err_df.copy()
    err_df["top_category"] = err_df["category"].str.split("/").str[0]

    print("\n  (b) By top-level category:")
    for cat, cnt in err_df["top_category"].value_counts().items():
        print(f"    {cat:25s} {cnt:>8,}  ({cnt / n:.1%})")

    print("\n  (c) Full category breakdown (top 20):")
    for cat, cnt in err_df["category"].value_counts().head(20).items():
        print(f"    {cat:45s} {cnt:>8,}  ({cnt / n:.1%})")

    print("\n  (d) Severity x Top-category crosstab:")
    ct = pd.crosstab(err_df["top_category"], err_df["severity"])
    print(ct.to_string())

    print("\n  (e) Error counts by model:")
    print(err_df.groupby(["model", "severity"]).size().unstack(fill_value=0).to_string())
    print()

    # ── plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sev_order = [s for s in ["Minor", "Major", "Neutral"] if s in sev_counts.index]
    sns.countplot(data=err_df, y="severity", order=sev_order, ax=axes[0],
                  hue="severity", palette="Set2", legend=False)
    axes[0].set_title("Error severity distribution")
    axes[0].set_xlabel("Count")

    top_cats = err_df["top_category"].value_counts().index[:8]
    sns.countplot(data=err_df[err_df["top_category"].isin(top_cats)],
                  y="top_category", order=top_cats, ax=axes[1],
                  hue="top_category", palette="Set2", legend=False)
    axes[1].set_title("Error category distribution (top-level)")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'error_distribution.png'}")

    n_models = err_df["model"].nunique()
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), squeeze=False)
    for i, (model, grp) in enumerate(err_df.groupby("model")):
        ct_m = pd.crosstab(grp["top_category"], grp["severity"])
        ct_m = ct_m.reindex(columns=[c for c in sev_order if c in ct_m.columns])
        sns.heatmap(ct_m, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0, i])
        axes[0, i].set_title(model)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "error_heatmaps_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'error_heatmaps_by_model.png'}")
    print()


def section_mqm_scores(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 3 - MQM scores & system ranking")
    print("=" * 70)

    sys_scores = (
        df.groupby(["model", "system"])["mqm_score"]
        .mean()
        .reset_index()
        .rename(columns={"mqm_score": "avg_mqm"})
    )

    pivot = sys_scores.pivot(index="system", columns="model", values="avg_mqm")
    pivot["Human (gold)"] = pivot.index.map(lambda s: HUMAN_MQM_SCORES.get(s, np.nan))
    pivot = pivot.sort_values("Human (gold)")

    print("\n  (a) System-level avg MQM by model (lower = better):\n")
    print(pivot.round(3).to_string())
    print()

    # rank correlation
    print("  (b) Rank correlation with human gold MQM scores:")
    models = [c for c in pivot.columns if c != "Human (gold)"]
    valid = pivot.dropna(subset=["Human (gold)"])
    human_ranks = valid["Human (gold)"].rank()
    for m in models:
        if m in valid.columns and valid[m].notna().sum() >= 3:
            mr = valid[m].rank()
            tau, p_tau = stats.kendalltau(human_ranks, mr)
            rho, p_rho = stats.spearmanr(human_ranks, mr)
            r, p_r = stats.pearsonr(valid["Human (gold)"], valid[m])
            print(f"    {m:30s}  tau={tau:.3f} (p={p_tau:.3f})  "
                  f"rho={rho:.3f} (p={p_rho:.3f})  r={r:.3f} (p={p_r:.3f})")
    print()

    # bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = pivot.reset_index().melt(id_vars="system", var_name="judge", value_name="mqm")
    plot_df = plot_df.dropna(subset=["mqm"])
    sns.barplot(data=plot_df, x="system", y="mqm", hue="judge", ax=ax)
    ax.set_title("System-level MQM scores: LLM judges vs Human gold")
    ax.set_ylabel("Average MQM score (lower = better)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "system_mqm_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'system_mqm_scores.png'}")

    # scatter
    n_models = len(models)
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
        for i, m in enumerate(models):
            ax = axes[0, i]
            sub = valid[["Human (gold)", m]].dropna()
            ax.scatter(sub["Human (gold)"], sub[m], s=60, zorder=5)
            for sys_name, row in sub.iterrows():
                ax.annotate(sys_name.split(".")[0], (row["Human (gold)"], row[m]),
                            fontsize=7, ha="center", va="bottom")
            lo, hi = min(sub.min()), max(sub.max())
            margin = (hi - lo) * 0.1
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", alpha=0.3, label="y=x")
            ax.set_xlabel("Human MQM")
            ax.set_ylabel(f"{m} MQM")
            ax.set_title(m)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "system_scatter_vs_human.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved {OUT_DIR / 'system_scatter_vs_human.png'}")
    print()


def section_segment_level(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 4 - Segment-level analysis")
    print("=" * 70)

    print("\n  (a) Segment-level MQM score distribution per model:")
    for model, grp in df.groupby("model"):
        d = grp["mqm_score"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
        print(f"    {model}: mean={d['mean']:.3f}  std={d['std']:.3f}  "
              f"median={d['50%']:.3f}  p90={d['90%']:.3f}  p95={d['95%']:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, grp in df.groupby("model"):
        scores = grp["mqm_score"].clip(upper=grp["mqm_score"].quantile(0.99))
        ax.hist(scores, bins=50, alpha=0.5, label=model, density=True)
    ax.set_xlabel("MQM score (per annotation)")
    ax.set_ylabel("Density")
    ax.set_title("Segment-level MQM score distribution by model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "segment_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved {OUT_DIR / 'segment_score_distribution.png'}")

    df["_no_error"] = (df["mqm_score"] == 0)
    no_err = df.groupby(["model", "system"])["_no_error"].mean().unstack()
    print("\n  (b) Fraction of 'no error' annotations (model x system):")
    print(no_err.round(3).to_string())

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        clip_val = sub["mqm_score"].quantile(0.99)
        plot_data = sub.assign(mqm_clipped=sub["mqm_score"].clip(upper=clip_val))
        order = sub.groupby("system")["mqm_score"].mean().sort_values().index.tolist()
        sns.boxplot(data=plot_data, x="system", y="mqm_clipped",
                    order=order, ax=axes[0, i])
        axes[0, i].set_title(f"MQM by system - {model}")
        axes[0, i].tick_params(axis="x", rotation=45)
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("MQM score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "segment_boxplots_by_system.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'segment_boxplots_by_system.png'}")
    print()


def section_inter_judge_agreement(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 5 - Inter-judge (model) agreement")
    print("=" * 70)

    seg_mean = (
        df.groupby(["model", "system", "seg_id", "rater"])["mqm_score"]
        .mean()
        .reset_index()
    )
    pivot_seg = seg_mean.pivot_table(
        index=["system", "seg_id", "rater"], columns="model", values="mqm_score"
    )
    models = list(pivot_seg.columns)
    if len(models) < 2:
        print("  Only one model found - skipping inter-judge analysis.\n")
        return

    print(f"\n  Models compared: {models}")

    corr_p = pivot_seg.corr(method="pearson")
    corr_s = pivot_seg.corr(method="spearman")
    print("\n  Pairwise Pearson r (segment-level MQM scores):")
    print(corr_p.round(3).to_string())
    print("\n  Pairwise Spearman rho:")
    print(corr_s.round(3).to_string())
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(corr_p, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title("Pearson r (segment MQM)")
    sns.heatmap(corr_s, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Spearman rho (segment MQM)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "inter_judge_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'inter_judge_correlation.png'}")

    print("\n  Binary agreement (has-error vs no-error):")
    binary = (pivot_seg > 0).astype(int)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mask = binary[[models[i], models[j]]].dropna().index
            agree = (binary.loc[mask, models[i]] == binary.loc[mask, models[j]]).mean()
            print(f"    {models[i]:30s} vs {models[j]:30s}  agreement={agree:.1%}")
    print()


def section_sampling_variability(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 6 - Sampling variability (within-model)")
    print("=" * 70)

    reps = df.groupby(["segment_id", "model"]).size()
    multi = reps[reps > 1]
    if multi.empty:
        print("  No repeated samples found - skipping.\n")
        return

    print(f"\n  Segments with >1 sample: {len(multi):,} "
          f"(mean reps={multi.mean():.1f}, max={multi.max()})")

    var_df = (
        df.groupby(["segment_id", "model"])["mqm_score"]
        .agg(["size", "mean", "std"])
        .rename(columns={"size": "n", "mean": "mean_score", "std": "std_score"})
        .reset_index()
    )
    var_df = var_df[var_df["n"] > 1]

    print("\n  Within-segment MQM score std (across repeated samples):")
    for model, grp in var_df.groupby("model"):
        s = grp["std_score"]
        print(f"    {model:30s}  mean_std={s.mean():.3f}  median_std={s.median():.3f}")

    # binary consistency: vectorized via groupby
    multi_df = df.merge(multi.reset_index()[["segment_id", "model"]],
                        on=["segment_id", "model"])
    multi_df["has_error"] = (multi_df["mqm_score"] > 0).astype(int)
    consistency = (
        multi_df.groupby(["segment_id", "model"])["has_error"]
        .agg(["nunique"])
        .rename(columns={"nunique": "n_unique"})
        .reset_index()
    )
    consistency["all_agree"] = consistency["n_unique"] == 1

    print("\n  Binary consistency (all samples agree on error/no-error):")
    for model, grp in consistency.groupby("model"):
        print(f"    {model:30s}  consistency={grp['all_agree'].mean():.1%}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for model, grp in var_df.groupby("model"):
        clip = grp["std_score"].clip(upper=grp["std_score"].quantile(0.99))
        ax.hist(clip, bins=40, alpha=0.5, label=model, density=True)
    ax.set_xlabel("Within-segment std of MQM score")
    ax.set_ylabel("Density")
    ax.set_title("Sampling variability (within-model, across repeated runs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sampling_variability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Saved {OUT_DIR / 'sampling_variability.png'}")
    print()


def section_rater_analysis(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 7 - Rater persona analysis")
    print("=" * 70)

    rater_stats = df.groupby(["model", "rater"]).agg(
        n=("mqm_score", "size"),
        mean_mqm=("mqm_score", "mean"),
        std_mqm=("mqm_score", "std"),
        frac_no_error=("mqm_score", lambda x: (x == 0).mean()),
    ).reset_index()

    print("\n  Mean MQM score by (model, rater):")
    print(rater_stats.pivot(index="rater", columns="model", values="mean_mqm")
          .round(3).to_string())

    print("\n  Fraction of no-error annotations by (model, rater):")
    print(rater_stats.pivot(index="rater", columns="model", values="frac_no_error")
          .round(3).to_string())
    print()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=rater_stats, x="rater", y="mean_mqm", hue="model", ax=ax)
    ax.set_title("Mean MQM score by rater persona and LLM model")
    ax.set_ylabel("Mean MQM score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rater_persona_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'rater_persona_analysis.png'}")
    print()


def load_human_mqm(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the original human MQM TSV, compute per-row error weights,
    and return a segment-level score table."""
    tsv_path = data_dir / "mqm_newstest2020_ende.tsv"
    hdf = pd.read_csv(tsv_path, sep="\t")

    sev = hdf["severity"].str.strip().str.lower()
    cat = hdf["category"].str.strip().str.lower()

    weight = pd.Series(0.0, index=hdf.index)
    weight[sev == "minor"] = 1.0
    weight[sev == "major"] = 5.0
    weight[(sev == "minor") & cat.str.startswith("fluency/punctuation")] = 0.1
    weight[cat.str.contains("non-translation", na=False)] = 25.0

    hdf["weight"] = weight

    # sum weights per (system, seg_id, rater) to get per-rater segment score
    human_seg = (
        hdf.groupby(["system", "seg_id", "rater"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "human_mqm"})
    )
    return human_seg


def section_human_model_alignment(df: pd.DataFrame):
    """Per-segment alignment between human MQM and each LLM judge."""
    print("=" * 70)
    print("SECTION 9 - Human-model alignment (segment-level)")
    print("=" * 70)

    t0 = time.time()
    human_seg = load_human_mqm()
    print(f"\n  Loaded human MQM scores: {len(human_seg):,} (system, seg_id, rater) entries")

    # human average across raters per (system, seg_id)
    human_avg = (
        human_seg.groupby(["system", "seg_id"])["human_mqm"]
        .mean()
        .reset_index()
        .rename(columns={"human_mqm": "human_avg_mqm"})
    )
    print(f"  Unique (system, seg_id) with human scores: {len(human_avg):,}")

    # LLM average across raters + repeated samples per (model, system, seg_id)
    llm_avg = (
        df.groupby(["model", "system", "seg_id"])["mqm_score"]
        .mean()
        .reset_index()
        .rename(columns={"mqm_score": "llm_avg_mqm"})
    )

    llm_avg["seg_id"] = llm_avg["seg_id"].astype(str)
    human_avg["seg_id"] = human_avg["seg_id"].astype(str)
    merged = llm_avg.merge(human_avg, on=["system", "seg_id"], how="inner")
    print(f"  Matched entries (model x system x seg_id): {len(merged):,}")
    models = sorted(merged["model"].unique())
    print(f"  Models: {models}")

    # ── (a) summary stats ──
    print("\n  (a) Per-model correlation with human segment-level scores:")
    corr_rows = []
    for model in models:
        sub = merged[merged["model"] == model]
        tau, p_tau = stats.kendalltau(sub["human_avg_mqm"], sub["llm_avg_mqm"])
        rho, p_rho = stats.spearmanr(sub["human_avg_mqm"], sub["llm_avg_mqm"])
        r, p_r = stats.pearsonr(sub["human_avg_mqm"], sub["llm_avg_mqm"])
        mae = (sub["human_avg_mqm"] - sub["llm_avg_mqm"]).abs().mean()
        bias = (sub["llm_avg_mqm"] - sub["human_avg_mqm"]).mean()
        print(f"    {model:20s}  tau={tau:.3f}  rho={rho:.3f}  r={r:.3f}  "
              f"MAE={mae:.3f}  bias={bias:+.3f}")
        corr_rows.append(dict(model=model, tau=tau, rho=rho, pearson_r=r,
                               mae=mae, bias=bias))
    corr_df = pd.DataFrame(corr_rows)
    print()

    # ── (b) overlaid histograms: human vs each LLM ──
    n_models = len(models)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(7 * ((n_models + 1) // 2), 10),
                             squeeze=False)
    axes_flat = axes.flatten()

    clip_hi = merged["human_avg_mqm"].quantile(0.99)
    bins = np.linspace(0, max(clip_hi, 15), 60)

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = merged[merged["model"] == model]
        ax.hist(sub["human_avg_mqm"].clip(upper=clip_hi), bins=bins,
                alpha=0.5, density=True, label="Human", color="steelblue")
        llm_clip = sub["llm_avg_mqm"].clip(upper=sub["llm_avg_mqm"].quantile(0.99))
        ax.hist(llm_clip, bins=bins,
                alpha=0.5, density=True, label=model, color="coral")
        ax.set_title(f"Human vs {model}")
        ax.set_xlabel("Avg MQM score per segment")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # hide unused subplots
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Segment-level MQM score distributions: Human vs LLM", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "human_model_alignment_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'human_model_alignment_histograms.png'}")

    # ── (c) scatter plots: human vs LLM per segment ──
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(6 * ((n_models + 1) // 2), 10),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = merged[merged["model"] == model]
        ax.scatter(sub["human_avg_mqm"], sub["llm_avg_mqm"], s=3, alpha=0.15, rasterized=True)

        lo = 0
        hi = max(sub["human_avg_mqm"].quantile(0.99), sub["llm_avg_mqm"].quantile(0.99))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="y=x")

        r_val = corr_df.loc[corr_df["model"] == model, "pearson_r"].values[0]
        rho_val = corr_df.loc[corr_df["model"] == model, "rho"].values[0]
        ax.set_title(f"{model}\nr={r_val:.3f}, rho={rho_val:.3f}", fontsize=10)
        ax.set_xlabel("Human avg MQM")
        ax.set_ylabel("LLM avg MQM")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi * 1.5)
        ax.legend(fontsize=8)

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Segment-level: Human vs LLM MQM scores", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "human_model_alignment_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'human_model_alignment_scatter.png'}")

    # ── (d) residual (LLM - Human) distribution ──
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(7 * ((n_models + 1) // 2), 10),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = merged[merged["model"] == model]
        residual = sub["llm_avg_mqm"] - sub["human_avg_mqm"]
        clip_r = residual.clip(lower=residual.quantile(0.01), upper=residual.quantile(0.99))
        ax.hist(clip_r, bins=60, alpha=0.7, color="mediumpurple", density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(residual.mean(), color="red", linestyle="-", alpha=0.7,
                   label=f"mean={residual.mean():.2f}")
        ax.set_title(f"{model}")
        ax.set_xlabel("LLM - Human (MQM)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Residual distribution (LLM - Human) per segment", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "human_model_alignment_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {OUT_DIR / 'human_model_alignment_residuals.png'}")

    print(f"\n  Section 9 done in {time.time()-t0:.1f}s\n")


def section_token_cost(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION 8 - Token usage summary")
    print("=" * 70)

    prompt_tok = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    comp_tok = pd.to_numeric(df["completion_tokens"], errors="coerce")
    total = prompt_tok + comp_tok

    valid = df.assign(_prompt=prompt_tok, _comp=comp_tok, _total=total).dropna(subset=["_prompt", "_comp"])
    tok_stats = valid.groupby("model").agg(
        n=("_total", "size"),
        mean_prompt=("_prompt", "mean"),
        mean_completion=("_comp", "mean"),
        mean_total=("_total", "mean"),
        total_prompt=("_prompt", "sum"),
        total_completion=("_comp", "sum"),
    ).round(1)
    print("\n  Token usage per model:")
    print(tok_stats.to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Loading data from", DATA_DIR)
    df = load_all_data(DATA_DIR)
    print(f"Loaded {len(df):,} rows in {time.time()-t_start:.1f}s.\n")

    error_rows = df[df["error"] != "NA"]
    if len(error_rows):
        print(f"WARNING: {len(error_rows):,} rows have non-NA 'error' field "
              f"(parse / API failures).\n")

    section_basic_stats(df)

    t0 = time.time()
    print("Computing MQM scores (vectorized)...")
    df["mqm_score"] = compute_mqm_scores_vectorized(df)
    print(f"  done in {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print("Extracting individual errors...")
    err_df = extract_error_rows_fast(df)
    print(f"  {len(err_df):,} individual errors extracted in {time.time()-t0:.1f}s.\n")

    section_error_distribution(err_df)
    section_mqm_scores(df)
    section_segment_level(df)
    section_inter_judge_agreement(df)
    section_sampling_variability(df)
    section_rater_analysis(df)
    section_token_cost(df)
    section_human_model_alignment(df)

    print("=" * 70)
    print(f"EDA complete in {time.time()-t_start:.1f}s. Figures saved to {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
