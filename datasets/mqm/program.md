# Two-Stage DR Estimator — Research Program

## Goal
Make the two-stage adaptive doubly-robust (DR) estimator beat baseline2 on the MQM dataset (pooled, N=14,180).

**Target**: `dr_rmse < baseline2_rmse` (≈ 0.034 from 100-trial ground truth)

## Dataset
- MQM newstest2020 en-de, N=14,180 segments
- Target: MQM score (mean ≈ 2.0, std ≈ 2.3, ~13% zeros, zero-inflated)
- Features:
  - `X_full` (88-dim): scaled LLM judge scores + uncertainty stats + 64 PCA dims of embeddings
  - `X_sigma` (64-dim): PCA of text embeddings only — currently used to fit sigma model
  - Raw embeddings: 3072-dim (text-embedding-3-large), PCA reduced

## Current Algorithm (train.py)

### Stage 1
Sample `s1 ~ Bernoulli(pi=0.10)` uniformly. Fit hurdle outcome model `f`.

### Sigma model (current best: sqrt-Ridge)
Compute residuals `r_i = (y_i - f(x_i))^2` for `i ∈ s1`. Fit RidgeCV on X_sigma[s1, :16 PCA dims]
to predict `sqrt(r_i)`. Normalize predictions to target mean, apply budget-neutral floor:
```python
target_sigma_mean = (ETA - PI) / (1.0 - PI)  # ≈ 0.111
sigma_raw = ridge.predict(X_sigma)  # predicted sqrt-residuals, normalized
sigma_floored = np.clip(sigma_raw, SIGMA_MIN, 1.0)
excess = sigma_floored.mean() - target_sigma_mean
if excess > 1e-6:
    free_mask = sigma_raw >= SIGMA_MIN
    sigma_floored[free_mask] -= excess * len(sigma_floored) / free_mask.sum()
    sigma_floored = np.clip(sigma_floored, SIGMA_MIN, 1.0)
sigma_all = sigma_floored
```

### Stage 2
Sample `s2_i ~ Bernoulli(sigma_i)` for unlabeled items. Total `s = s1 ∪ s2`.

### DR estimator (self-normalized / Hajek)
```
mu_hat = sum_i [ f_i + s_i * (y_i - f_i) / q_i ] / sum_i [ s_i / q_i ]
where q_i = PI + (1 - PI) * sigma_i
```

## Performance History (100-trial validated)
| Config                            | RMSE   | Bias    | SD     | sigma q10 | sigma q90 | sample_rate |
|-----------------------------------|--------|---------|--------|-----------|-----------|-------------|
| Standard DR (original)            | 0.0606 | -0.0452 | 0.0406 | 0.003     | 0.362     | 0.187       |
| Self-norm DR only                 | 0.0641 | +0.0004 | 0.0644 | 0.003     | 0.362     | 0.187       |
| **sqrt-Ridge PCA16 + floor 0.08** | **0.0403** | **-0.008** | **0.040** | **0.083** | **0.140** | **0.200** |
| **Baseline2 (target)**            | **0.0338** | +0.005 | **0.031** | —     | —         | 0.200       |

**Current gap**: dr_rmse=0.0403 vs baseline2=0.0338. SD is the bottleneck: 0.040 vs 0.031.
sigma spread is now only 1.7× (q10=0.083, q90=0.140) — budget correctly satisfied.

## Root Cause of Remaining Gap
The DR correction term `s_i*(y_i-f_i)/q_i` still adds variance even with controlled sigma.
With sigma near-uniform (spread 1.7×), we're essentially doing near-uniform DR — which is baseline2.
The adaptive sampling needs to add VALUE (reduce variance vs baseline2) by concentrating on items
where the *outcome model residuals are large AND predictable from features*.

The hurdle outcome model achieves R²≈0.35 on the full dataset (from MRR.py results).
Items where the model is reliably WRONG are the valuable targets for stage-2 sampling.

## Research Agenda (Next Directions)

### Direction A: Better outcome model for residual prediction → better sigma targets
Current: hurdle model trained on ~1,418 stage-1 points. With R²=0.35 total, residuals are noisy.
Try: use RIDGE (not hurdle) for stage-1 fitting — ridge is more stable with small N.
Or: use the FULL X_full features (judge scores + embeddings) for sigma rather than embeddings only.
The judge scores are directly predictive of hardness.

### Direction B: Two-step sigma — first predict hardness, then allocate
Instead of predicting sqrt(residual) directly, predict a "hardness score" using judge disagreement
(uncertainty features in X_full), then use that to set sigma.

Concretely: sigma_raw_i ∝ (judge_uncertainty_i)^alpha, normalized to budget.
The uncertainty features are columns in X_full. Access them via the pre-loaded features:
```python
# X_full columns: judge scores, uncertainty stats (var, std across judges), then PCA dims
# Use only the uncertainty/variance columns as hardness proxy — no ML needed
# Uncertainty cols are the features after the first ~3 per judge (mean, var, std)
```

### Direction C: Softmax temperature allocation (exact budget, bounded spread)
Replace Ridge with a temperature-controlled softmax that satisfies the budget exactly:
```python
stage2_budget_n = (ETA - PI) * n  # target number of stage-2 samples
# Learn scores on X_sigma via Ridge, then allocate via temperature softmax:
raw_scores = ridge.predict(X_sigma)  # any real-valued scores
temperature = T  # hyperparameter: higher T → more uniform
exp_scores = np.exp((raw_scores - raw_scores.mean()) / (raw_scores.std() * temperature + 1e-8))
sigma_all = np.clip(exp_scores / exp_scores.sum() * stage2_budget_n, 0.0, 1.0)
# Budget: sum(sigma_all) ≈ stage2_budget_n (exactly if no clipping)
```
Try temperature T in {1.0, 2.0, 5.0, 10.0}. Higher T = more uniform = lower variance DR.

### Direction D: Fewer PCA dims — try 4 and 8
Current best is SIGMA_PCA=16. Try 4 and 8 to further reduce overfitting.
With ~1,418 training points, 4-8 PCA dims may generalize better.

### Direction E: Use X_full for sigma (judge features + embeddings)
Current sigma uses only embedding PCA. The judge scores (first ~30 cols of X_full) contain
hardness information — items where judges disagree have high variance = hard items.
Try using X_full[:, :30] (judge-only cols) or all of X_full for the RidgeCV sigma fit.

### Direction F: Outcome model regularization for stage-1
The stage-1 outcome model trained on only 10% of data has high variance.
Try ridge (instead of hurdle) for stage-1 ONLY, then refit with hurdle on full s for final estimate.
Setting OUTCOME_MODEL_STAGE1 = "ridge" and OUTCOME_MODEL_FINAL = "hurdle".

## ⚠️ Budget Constraint Rule
`avg_sample_rate` in results MUST be 0.19–0.21. If outside this range, the experiment is invalid.

## Critical Rules
1. ALWAYS use self-normalized (Hajek) DR — keep the formula unchanged
2. ALWAYS verify budget: avg_sample_rate must be 0.19–0.21
3. Try Direction C (softmax) and Direction D (fewer PCA) first — cleanest changes
4. Do NOT use naive `clip(sigma, SIGMA_MIN, 1.0)` without budget rebalancing
5. Do NOT cross-fit (tensor shape errors with multiprocessing)
6. Do NOT repeat experiments already in the log

## Experiment Log
| # | Label | dr_rmse | dr_bias | sample_rate | ratio | Notes |
|---|-------|---------|---------|-------------|-------|-------|
| 0 | self_normalized_dr | 0.0641 | +0.0004 | 0.187 | 1.89× | Baseline: sigma too extreme |
| 3 | sqrt_ridge_budget_neutral_floor | 0.0404 | -0.0081 | ~1.25× | — | INVALID: sample_rate too high |
| 6 | sqrt_ridge_pca16_budget_neutral_floor | 0.0403 | -0.008 | 0.200 | 1.19× | **Current best (100-trial validated)** |

## Key Constants (current train.py)
```
PI = 0.10, ETA = 0.20
OUTCOME_MODEL = "hurdle"
SIGMA_PCA = 16          ← try 4, 8
SIGMA_MIN = 0.08        ← budget-neutral floor already applied correctly
Q_MIN_CLIP = 1e-6
target_sigma_mean = (ETA - PI) / (1 - PI) ≈ 0.111
```

| 1 | sqrt_ridge_judge_features_sigma | 0.0383 | -0.0031 | 1.15× | ## Hypothesis

The current sigma model uses only 16 PCA dims… ✗ |

| 2 | softmax_temp_alloc_xfull_T5 | 0.0362 | -0.0090 | 1.08× | ## Hypothesis

The current sigma model uses only the first 3… ✗ |

| 3 | sqrt_ridge_judge30_budget_neutral_floor | 0.0386 | -0.0040 | 1.19× | ## Hypothesis

The softmax allocation with T=5.0 using all 8… ✗ |

| 4 | sqrt_ridge_xfull32_floor006 | 0.0466 | -0.0009 | 1.48× | ## Hypothesis

The experiment log shows that using X_full (8… ✗ |

| 5 | sqrt_ridge_pca8_budget_neutral_floor | 0.0537 | -0.0227 | 1.75× | ## Hypothesis

The experiment log shows that experiments usi… ✗ |

| 6 | sqrt_ridge_pca8_floor006_robust_budget | 0.0405 | -0.0025 | 1.37× | ## Hypothesis

The experiment log shows that experiments #1 … ✗ |

| 7 | sqrt_ridge_judge30_floor008_stable_budget | 0.0404 | -0.0070 | 1.11× | ## Hypothesis

The experiment log shows that experiment #1 (… ✗ |

| 8 | sqrt_ridge_judge30_iterative_budget | 0.0376 | -0.0013 | 1.17× | ## Hypothesis

Looking at the experiment log, the best valid… ✗ |

| 9 | gbm_judge30_iterative_budget | 0.0633 | -0.0447 | 2.45× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |

| 10 | sqrt_ridge_judge30_bisection_budget_floor006 | 0.0524 | -0.0150 | 1.40× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |

| 11 | linear_alloc_judge30_bisection_floor006 | 0.0481 | -0.0017 | 1.46× | ## Hypothesis

Looking at the experiment log, the best valid… ✗ |

| 12 | sqrt_ridge_judge30_bisection_exact_floor008 | 0.0448 | -0.0015 | 1.51× | ## Hypothesis

Looking at the experiment log, experiment #8 … ✗ |

| 13 | sqrt_ridge_judge30_bisection_floor006_robust2 | 0.0493 | -0.0113 | 1.93× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |

| 14 | sqrt_ridge_judge30_normalized_bisection_strict | 0.0417 | -0.0013 | 1.22× | ## Hypothesis

Looking at the experiment log, the best valid… ✗ |

| 15 | sqrt_ridge_judge30_normalized_floor008_v2 | 0.0451 | -0.0016 | 1.74× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |

| 16 | sqrt_ridge_judge30_iterative_floor006_v2 | 0.0363 | 0.0022 | 1.07× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |

| 17 | sqrt_ridge_judge30_iterative_floor006_v3 | 0.0439 | 0.0045 | 1.58× | ## Hypothesis

Looking at the experiment log, the best perfo… ✗ |
