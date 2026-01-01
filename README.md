# BACON: Budgeted Human Calibration for Modeling and Evaluation with Multiple AI Judges

This repository contains the code to replicate the experiments in the paper. It covers three datasets: **PERSUADE** (essay scoring), **MQM** (machine translation evaluation), and **WebDesign** (web-design perception scoring).

---

## Repository Structure

```
datasets/
├── persuade/          # Essay scoring (PERSUADE corpus)
│   ├── llm_scoring.py          # LLM judge scoring of essays
│   ├── embedder.py             # Text embedding extraction
│   ├── uniform_dr_simulation.py  # DR simulation (Table 1 / Figure 3)
│   └── two_stage_dr_simulation.py
├── mqm/               # Machine translation (WMT 2020 MQM)
│   ├── data/
│   │   └── run_llm_mqm_new_models.py  # LLM MQM annotation script
│   ├── embedder.py             # Segment embedding extraction
│   ├── train.py                # Outcome model training
│   ├── uniform_dr_ranking_simulation.py  # DR simulation (Table 2 / Figure 4)
│   └── two_stage_dr_simulation.py
└── WebDesign/         # Web-design perception scoring
    ├── vlm_scoring.py          # VLM judge scoring of screenshots
    ├── embedder.py             # Image embedding extraction (SigLIP)
    ├── uniform_dr_simulation_unis.py  # DR simulation (Table 3 / Figure 5)
    └── two_stage_dr_simulation.py
```

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai pandas numpy scipy scikit-learn tqdm python-dotenv matplotlib
```

Create a `.env` file in the project root with your API credentials:

```
API_KEY=<your-key>
PLUTO_ENDPOINT=<openai-compatible-endpoint>
```

All scoring and embedding scripts read credentials from this file automatically.

---

## Dataset Access

Raw data files are **not included** in this repository. Follow the instructions below to obtain each dataset and place it in the expected path.

### 1. PERSUADE (Essay Scoring)

**Source:** Crossley et al. (2024), *Assessing Writing* — [PERSUADE Corpus 2.0](https://github.com/scrosseye/PERSUADE_corpus_2.0)

25,000+ argumentative essays from US students (grades 6–12); each essay carries a holistic human score from 1–6. Publicly released for research.

```bash
git clone https://github.com/scrosseye/PERSUADE_corpus_2.0
cp PERSUADE_corpus_2.0/persuade_2.0_human_scores_demo_id_github.csv \
   datasets/persuade/data/
```

The main columns used are `holistic_essay_score` (1–6), `full_text`, `prompt_name`, and `grade_level`.

### 2. MQM (Machine Translation — WMT 2020 en-de)

**Source:** Freitag et al. (2021), *TACL*; WMT 2020 shared task — [wmt-mqm-human-evaluation](https://github.com/google/wmt-mqm-human-evaluation)

Expert human MQM (Multidimensional Quality Metrics) annotations for 10 MT systems on the English–German language pair; scores are non-negative weighted error sums (zero = no errors detected). Publicly released.

```bash
wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2020/ende/mqm_newstest2020_ende.tsv \
  -O datasets/mqm/data/mqm_newstest2020_ende.tsv
```

Expected columns: `system`, `doc`, `doc_id`, `seg_id`, `rater`, `source`, `target`, `category`, `severity`.

### 3. WebDesign (Web-Design Perception Scoring)

**Source:** [Harvard Dataverse — doi:10.7910/DVN/9FKSQI](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/9FKSQI)

Perceptual ratings of university websites on 6 dimensions (Aesthetics, Trustworthiness, Typicality, Exemplar Goodness, Family Resemblance, Usability) on a −3 to +3 scale, collected via human raters. Download from the Harvard Dataverse link above and place the Universities split as:

```
datasets/WebDesign/Universitites/data/   # screenshot images (.png)
datasets/WebDesign/Universitites/data/ratings.csv  # human ratings
```

---

## Replication Steps

### Step 1: LLM / VLM Judge Scoring

Generate AI judge annotations for all items in each dataset.

**PERSUADE:**
```bash
python datasets/persuade/llm_scoring.py \
  --data datasets/persuade/data/persuade_2.0_human_scores_demo_id_github.csv \
  --out datasets/persuade/results/llm_scores.jsonl
```

**MQM:**
```bash
python datasets/mqm/data/run_llm_mqm_new_models.py \
  --tsv datasets/mqm/data/mqm_newstest2020_ende.tsv \
  --out datasets/mqm/data/llm_mqm_annotations.jsonl \
  --max-concurrent 32
```

The MQM script supports resuming: if the output file already exists, already-completed `(segment_id, model)` pairs are skipped automatically.

**WebDesign:**
```bash
python datasets/WebDesign/vlm_scoring.py \
  --data-dir datasets/WebDesign/Universitites/data \
  --out datasets/WebDesign/Universitites/results/vlm_scores.jsonl
```

### Step 2: Embedding Extraction

Extract contextual embeddings for all items (used as features in the outcome model).

**PERSUADE / MQM** (text embeddings via `text-embedding-3-large`):
```bash
python datasets/persuade/embedder.py \
  --data datasets/persuade/data/persuade_2.0_human_scores_demo_id_github.csv \
  --out datasets/persuade/results/embeddings.npz

python datasets/mqm/embedder.py \
  --tsv datasets/mqm/data/mqm_newstest2020_ende.tsv \
  --out datasets/mqm/results/embeddings.npz
```

**WebDesign** (image embeddings via SigLIP):
```bash
python datasets/WebDesign/embedder.py \
  --data-dir datasets/WebDesign/Universitites/data \
  --out datasets/WebDesign/Universitites/results/embeddings.npz
```

### Step 3: DR Simulation (Main Results)

Run the doubly robust estimation simulation across outcome model configurations and sampling budgets.

**PERSUADE** (reproduces Table 1 / Figure 3):
```bash
python datasets/persuade/uniform_dr_simulation.py \
  --scores  datasets/persuade/results/llm_scores.jsonl \
  --embeddings datasets/persuade/results/embeddings.npz \
  --human-data datasets/persuade/data/persuade_2.0_human_scores_demo_id_github.csv \
  --out-dir datasets/persuade/results/
```

**MQM** (reproduces Table 2 / Figure 4):
```bash
python datasets/mqm/uniform_dr_ranking_simulation.py \
  --annotations datasets/mqm/data/llm_mqm_annotations.jsonl \
  --embeddings datasets/mqm/results/embeddings.npz \
  --tsv datasets/mqm/data/mqm_newstest2020_ende.tsv \
  --out-dir datasets/mqm/results/
```

**WebDesign** (reproduces Table 3 / Figure 5):
```bash
python datasets/WebDesign/uniform_dr_simulation_unis.py \
  --scores  datasets/WebDesign/Universitites/results/vlm_scores.jsonl \
  --embeddings datasets/WebDesign/Universitites/results/embeddings.npz \
  --out-dir datasets/WebDesign/Universitites/results/
```

---

## Notes

- All simulation scripts accept a `--n-trials` argument (default 200) and `--seed` for reproducibility.
- Rate-limit handling and resume support are built into all scoring scripts. If a run is interrupted, simply re-run the same command.
- The `datasets/mqm/data/` directory also contains `Untitled1.ipynb`, the original notebook used for the DeepSeek/Kimi/Qwen annotation runs, for reference.
- Figures are saved as `.png` files in the respective `results/` or `figures/` directories.

---

## Citation

If you use this code or the BACON framework, please cite our paper:

```bibtex
@inproceedings{bacon2026,
  title     = {{BACON}: Budgeted Human Calibration for Modeling and Evaluation with Multiple {AI} Judges},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
