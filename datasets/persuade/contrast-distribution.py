from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PERSUADE_HUMAN_PATH = Path("datasets/persuade/data/persuade_corpus_2.0_train.csv")
PERSUADE_LLM_PATH = Path("datasets/persuade/results/llm_scores.jsonl")
PERSUADE_OUT_PATH = Path("datasets/persuade/results/persuade_grade10_human_vs_llm_score_distribution.png")

WEB_HUMAN_PATH = Path("datasets/WebDesign/Universitites/data/ratings.avg.unis.txt")
WEB_LLM_PATH = Path("datasets/WebDesign/Universitites/results/vlm_scores_unis.jsonl")
WEB_OUT_PATH = Path("datasets/WebDesign/Universitites/results/unis_human_vs_vlm_score_distribution.png")
MQM_TSV_PATH = Path("datasets/mqm/data/mqm_newstest2020_ende.tsv")
MQM_LLM_DIR  = Path("datasets/mqm/data")
MQM_OUT_PATH = Path("datasets/mqm/results/mqm_human_model_alignment_histograms.png")

WEB_OUTCOMES = ["AE", "TRU", "TYP", "EXMPL", "AVG", "US"]
WEB_REQUESTED_MODELS = ["gemini-2.5-flash-lite", "gpt-4.1", "gpt-4o", "llama-3-1-8b"]
WEB_MODEL_ALIASES = {
    "gemini-2.5-flash": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash-lite": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "gpt-4.1": ["gpt-4.1"],
    "gpt-4o": ["gpt-4o"],
    "llama-3-1-8b": ["llama-3-1-8b"],
}


def _distribution_table(df: pd.DataFrame, source_col: str, score_col: str, score_values: list[int]) -> pd.DataFrame:
    out = []
    for source, sub in df.groupby(source_col):
        vc = sub[score_col].value_counts(normalize=True).reindex(score_values, fill_value=0.0)
        for score, frac in vc.items():
            out.append({"source": source, "score": int(score), "fraction": float(frac)})
    return pd.DataFrame(out)


def make_persuade_plot() -> Path:
    human = pd.read_csv(PERSUADE_HUMAN_PATH, low_memory=False)
    human = human.drop_duplicates("essay_id_comp")[["essay_id_comp", "grade_level", "holistic_essay_score"]]
    human = human[human["grade_level"] == 10].copy()
    human = human.rename(columns={"essay_id_comp": "id", "holistic_essay_score": "score"})
    human["source"] = "Human"

    llm = pd.read_json(PERSUADE_LLM_PATH, lines=True)
    llm = llm.rename(columns={"essay_id_comp": "id", "predicted_score": "score", "model": "source"})
    llm = llm[["id", "source", "score"]].copy()
    llm["score"] = pd.to_numeric(llm["score"], errors="coerce")
    llm = llm.dropna(subset=["score"])
    llm = llm[llm["score"].between(1, 6)]
    llm["score"] = llm["score"].astype(int)

    g10_ids = set(human["id"].tolist())
    llm = llm[llm["id"].isin(g10_ids)].copy()
    all_models = sorted(llm["source"].dropna().unique().tolist())
    plot_df = pd.concat([human[["id", "source", "score"]], llm], ignore_index=True)
    all_sources = ["Human", *all_models]

    # Compute proportion at each score value per source
    score_values = [1, 2, 3, 4, 5, 6]
    prop_rows = []
    for src in all_sources:
        sub = plot_df[plot_df["source"] == src]["score"]
        vc = sub.value_counts(normalize=True).reindex(score_values, fill_value=0.0)
        for s, p in vc.items():
            prop_rows.append({"source": src, "score": s, "proportion": p})
    prop_df = pd.DataFrame(prop_rows)

    palette = sns.color_palette("tab10", n_colors=len(all_sources))
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for i, src in enumerate(all_sources):
        sub = prop_df[prop_df["source"] == src]
        lw = 2.8 if src == "Human" else 1.8
        ls = "-" if src == "Human" else "--"
        ax.plot(sub["score"], sub["proportion"], marker=markers[i % len(markers)],
                linewidth=lw, linestyle=ls, markersize=6, label=src, color=palette[i])

    ax.set_xlabel("Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=13, fontweight="bold")
    ax.set_xticks(score_values)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")

    legend = ax.legend(title="Source", loc="upper right", ncol=1, fontsize=9,
                       title_fontsize=10, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    PERSUADE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PERSUADE_OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return PERSUADE_OUT_PATH


def make_webdesign_plot() -> Path:
    human = pd.read_csv(WEB_HUMAN_PATH, sep="\t")
    human = human[["stimulusId", *WEB_OUTCOMES]].copy()

    llm = pd.read_json(WEB_LLM_PATH, lines=True)
    llm = llm.rename(columns={"stimulus_id": "stimulusId"})
    need_cols = ["stimulusId", "model", *[f"score_{o}" for o in WEB_OUTCOMES]]
    llm = llm[need_cols].copy()

    selected_sources = sorted(llm["model"].dropna().unique().tolist())

    score_values = [-3, -2, -1, 0, 1, 2, 3]
    markers = ["s", "D", "^", "v", "P", "X"]
    human_color = "#1f77b4"
    model_palette = sns.color_palette("tab10", n_colors=len(selected_sources))

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
    axes = axes.ravel()

    for i, outcome in enumerate(WEB_OUTCOMES):
        ax = axes[i]

        # Human: KDE on continuous averaged ratings
        h_scores = pd.to_numeric(human[outcome], errors="coerce").dropna()
        h_scores = h_scores[h_scores.between(-3, 3)]
        sns.kdeplot(x=h_scores, ax=ax, color=human_color, linewidth=2.5,
                    bw_adjust=0.8, clip=(-3, 3), fill=False, label="Human")

        # Models: proportion lines on discrete integer scores
        for j, model in enumerate(selected_sources):
            m_scores = pd.to_numeric(
                llm.loc[llm["model"] == model, f"score_{outcome}"], errors="coerce"
            ).dropna()
            m_scores = m_scores[m_scores.between(-3, 3)].astype(int)
            props = m_scores.value_counts(normalize=True).reindex(score_values, fill_value=0.0)
            ax.plot(score_values, props.values, marker=markers[j % len(markers)],
                    linewidth=1.6, linestyle="--", markersize=5,
                    label=model, color=model_palette[j])

        ax.set_title(outcome, fontsize=12, fontweight="bold")
        ax.set_xticks(score_values)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("Score" if i >= 3 else "", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density / Prop." if i % 3 == 0 else "", fontsize=10, fontweight="bold")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(9)
            lbl.set_fontweight("bold")
        if ax.get_legend():
            ax.get_legend().remove()

    # Shared legend on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, title="Source", loc="center left",
                     bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=10, title_fontsize=11)
    leg.get_title().set_fontweight("bold")
    for text in leg.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 1])
    WEB_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(WEB_OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return WEB_OUT_PATH


def make_mqm_plot() -> Path:
    from datasets.mqm.MRR import load_human_mqm, load_llm_mqm, TARGET

    human_df = load_human_mqm(MQM_TSV_PATH)
    llm_df   = load_llm_mqm(MQM_LLM_DIR)

    human_scores = human_df[TARGET].values
    # Exclude degenerate models (≥95% zero predictions)
    zero_frac = (
        llm_df.groupby(["model", "seg_id"])["llm_mean"]
        .mean()
        .reset_index()
        .groupby("model")["llm_mean"]
        .apply(lambda s: (s == 0).mean())
    )
    excluded = set(zero_frac[zero_frac >= 0.95].index.tolist())
    if excluded:
        llm_df = llm_df[~llm_df["model"].isin(excluded)]
    models = sorted(llm_df["model"].unique().tolist())
    llm_mean = (
        llm_df.groupby(["model", "seg_id"])["llm_mean"]
        .mean()
        .reset_index()
    )

    all_sources = ["Human"] + models
    palette = sns.color_palette("tab10", n_colors=len(all_sources))

    # Use raw per-(model, system, segment) scores so the zero bin reflects individual
    # per-system predictions (not averaged across systems).
    source_raw = {"Human": human_scores}
    for model in models:
        source_raw[model] = llm_df.loc[llm_df["model"] == model, "llm_mean"].values

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    clip_hi = float(np.percentile(human_scores, 99))
    binwidth = 1.0

    sns.histplot(x=np.clip(human_scores, 0, clip_hi), ax=ax,
                 color=palette[0], linewidth=2.2, binwidth=binwidth,
                 stat="density", element="step", fill=False, label="Human")

    for j, model in enumerate(models):
        scores = source_raw[model]
        s_hi = float(np.percentile(scores, 99))
        sns.histplot(x=np.clip(scores, 0, s_hi), ax=ax, color=palette[j + 1],
                     linewidth=1.6, binwidth=binwidth,
                     stat="density", element="step", fill=False, label=model,
                     linestyle="--")

    ax.set_xlabel("Avg MQM score per segment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Density", fontsize=13, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(11)
        lbl.set_fontweight("bold")

    legend = ax.legend(title="Source", loc="upper right", ncol=1, fontsize=9,
                       title_fontsize=10, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    MQM_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MQM_OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return MQM_OUT_PATH


def main() -> None:
    p_out = make_persuade_plot()
    w_out = make_webdesign_plot()
    m_out = make_mqm_plot()
    print(f"Saved: {p_out.resolve()}")
    print(f"Saved: {w_out.resolve()}")
    print(f"Saved: {m_out.resolve()}")


if __name__ == "__main__":
    main()
