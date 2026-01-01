"""
MRR for PERSUADE (single outcome: holistic_essay_score) + two-stage hardness.

Judge model:
    y = b + L w [+ uncertainty interactions] [+ SigLIP/OpenAI text embedding intercept]

    Sigmoid-weighted variant (SigmoidWeightedJudge):
        y = intercept + emb_pca @ alpha
              + sum_m [ s_m * beta0_m * sigmoid(beta1_m * ent_m + beta2_m * logppl_m) ]
        Uncertainty inputs are standardised per-model on training data.
        Fitted end-to-end via L-BFGS-B (scipy).

Hardness model (nested cross-fit):
    target = log(residual^2 + eps), features = embeddings only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from mord import LogisticAT
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "holistic_essay_score"
ALPHAS = np.logspace(-4, 4, 40)


def load_human(csv_path: Path, grade: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    essay_df = df.drop_duplicates("essay_id_comp")[
        ["essay_id_comp", TARGET, "grade_level"]
    ].copy()
    if grade is not None:
        essay_df = essay_df[essay_df["grade_level"] == grade]
    essay_df = essay_df.rename(columns={"essay_id_comp": "id"})
    return essay_df.reset_index(drop=True)


def load_llm_scores(path: Path, models: list[str] | None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df.rename(columns={"essay_id_comp": "id"})
    if models:
        df = df[df["model"].isin(models)]
    return df.reset_index(drop=True)


def load_embeddings(npz_path: Path) -> pd.DataFrame:
    arr = np.load(npz_path, allow_pickle=True)
    ids = [str(x) for x in arr["essay_ids"]]
    emb = arr["embeddings"]
    cols = [f"emb__{i}" for i in range(emb.shape[1])]
    e = pd.DataFrame(emb, columns=cols)
    e.insert(0, "id", ids)
    return e.drop_duplicates("id")


def build_feature_table(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    emb_df: pd.DataFrame | None,
    use_uncertainty: bool,
) -> tuple[pd.DataFrame, list[str]]:
    pivot_score = llm_df.pivot_table(index="id", columns="model", values="predicted_score", aggfunc="first")
    model_names = list(pivot_score.columns)
    pivot_score.columns = [f"s__{m}" for m in model_names]
    base = human_df.merge(pivot_score.reset_index(), on="id", how="inner").copy()
    feature_cols = [f"s__{m}" for m in model_names]

    if use_uncertainty:
        pivot_entropy = llm_df.pivot_table(index="id", columns="model", values="mean_entropy", aggfunc="first")
        pivot_ppl = llm_df.pivot_table(index="id", columns="model", values="perplexity", aggfunc="first")
        pivot_entropy.columns = [f"entropy__{m}" for m in pivot_entropy.columns]
        pivot_ppl.columns = [f"ppl__{m}" for m in pivot_ppl.columns]
        base = base.merge(pivot_entropy.reset_index(), on="id", how="left")
        base = base.merge(pivot_ppl.reset_index(), on="id", how="left")
        for m in model_names:
            s = f"s__{m}"
            e = f"entropy__{m}"
            p = f"ppl__{m}"
            lp = f"logppl__{m}"
            sxe = f"sxe__{m}"
            sxlp = f"sxlp__{m}"
            base[lp] = np.log(base[p].clip(lower=1e-8))
            base[sxe] = base[s] * base[e]
            base[sxlp] = base[s] * base[lp]
            feature_cols.extend([sxe, sxlp])

    if emb_df is not None:
        base = base.merge(emb_df, on="id", how="left")
        emb_cols = [c for c in base.columns if c.startswith("emb__")]
        feature_cols.extend(emb_cols)

    base.replace([np.inf, -np.inf], np.nan, inplace=True)
    return base, feature_cols


def make_pipe(feature_cols: list[str], seed: int, emb_pca: int | None) -> Pipeline:
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]
    if emb_cols:
        pca_n = min(emb_pca or 64, len(emb_cols))
        pre = ColumnTransformer(
            transformers=[
                (
                    "base",
                    Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                    non_emb,
                ),
                (
                    "emb",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=pca_n, random_state=seed)),
                        ]
                    ),
                    emb_cols,
                ),
            ]
        )
    else:
        pre = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

    return Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=ALPHAS, fit_intercept=True))])


MN_CS = np.logspace(-4, 4, 20)


def make_multinomial_pipe(feature_cols: list[str], seed: int, emb_pca: int | None) -> Pipeline:
    """Same preprocessor as make_pipe, but with multinomial logistic regression as head."""
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]
    if emb_cols:
        pca_n = min(emb_pca or 64, len(emb_cols))
        pre = ColumnTransformer(
            transformers=[
                (
                    "base",
                    Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                    non_emb,
                ),
                (
                    "emb",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=pca_n, random_state=seed)),
                        ]
                    ),
                    emb_cols,
                ),
            ]
        )
    else:
        pre = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

    clf = LogisticRegressionCV(
        Cs=MN_CS,
        cv=3,
        solver="saga",
        max_iter=3000,
        tol=1e-3,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


ORDINAL_ALPHAS = np.logspace(-4, 4, 20)


def _make_preprocessor(feature_cols: list[str], seed: int, emb_pca: int | None):
    """Standalone preprocessor (impute + scale + PCA on embeddings) reused by ordinal CV."""
    emb_cols = [c for c in feature_cols if c.startswith("emb__")]
    non_emb = [c for c in feature_cols if c not in emb_cols]
    if emb_cols:
        pca_n = min(emb_pca or 64, len(emb_cols))
        return ColumnTransformer(
            transformers=[
                (
                    "base",
                    Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                    non_emb,
                ),
                (
                    "emb",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=pca_n, random_state=seed)),
                        ]
                    ),
                    emb_cols,
                ),
            ]
        )
    return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])


def _cv_alpha_ordinal(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> float:
    """3-fold inner CV to select LogisticAT alpha by MSE of E[Y]."""
    classes = np.sort(np.unique(ytr)).astype(float)
    best_alpha, best_mse = ORDINAL_ALPHAS[0], np.inf
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed + 1)
    for a in ORDINAL_ALPHAS:
        fold_mses = []
        for itr, ite in inner_kf.split(Xtr):
            try:
                m = LogisticAT(alpha=a, max_iter=2000)
                m.fit(Xtr[itr], ytr[itr])
                ev = m.predict_proba(Xtr[ite]) @ classes
                fold_mses.append(float(mean_squared_error(ytr[ite], ev)))
            except Exception:
                fold_mses.append(np.inf)
        mse = float(np.mean(fold_mses))
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    return best_alpha


def crossfit_predict_ordinal(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    folds: int,
    seed: int,
    emb_pca: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """K-fold OOF for proportional-odds ordinal regression (mord.LogisticAT).

    Returns
    -------
    oof_ev   : expected-value predictions E[Y] = sum_k k * P(Y=k)
    oof_hard : argmax class predictions
    metrics  : dict with cv_rmse, cv_r2, cv_spearman, cv_accuracy, mean_alpha
    """
    X = df[features]
    y = df[target].astype(int).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_ev = np.zeros(len(y), dtype=float)
    oof_hard = np.zeros(len(y), dtype=int)
    best_alphas = []

    for tr, te in kf.split(X):
        pre = _make_preprocessor(features, seed=seed, emb_pca=emb_pca)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        y_tr = y[tr]

        alpha = _cv_alpha_ordinal(Xtr, y_tr, seed)
        best_alphas.append(alpha)

        mdl = LogisticAT(alpha=alpha, max_iter=2000)
        mdl.fit(Xtr, y_tr)

        classes = mdl.classes_.astype(float)
        proba = mdl.predict_proba(Xte)
        oof_ev[te] = proba @ classes
        oof_hard[te] = mdl.predict(Xte)

    return oof_ev, oof_hard, {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof_ev))),
        "cv_r2": float(r2_score(y, oof_ev)),
        "cv_spearman": float(pd.Series(y).corr(pd.Series(oof_ev), method="spearman")),
        "cv_accuracy": float(accuracy_score(y, oof_hard)),
        "mean_alpha_cv_folds": float(np.mean(best_alphas)),
    }


def run_judge_ordinal(
    df: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
    emb_pca: int | None,
    out_dir: Path,
    exp: str,
) -> pd.DataFrame:
    y = df[TARGET].astype(int).values
    oof_ev, oof_hard, m = crossfit_predict_ordinal(df, features, TARGET, folds, seed, emb_pca)

    metrics = pd.DataFrame(
        [
            {
                "experiment": exp,
                "n_samples": int(len(df)),
                "n_features_raw": int(len(features)),
                "cv_rmse": m["cv_rmse"],
                "cv_r2": m["cv_r2"],
                "cv_spearman": m["cv_spearman"],
                "cv_accuracy": m["cv_accuracy"],
                "mean_C_cv_folds": np.nan,
                "mean_alpha_cv_folds": m["mean_alpha_cv_folds"],
                "std_alpha_cv_folds": np.nan,
            }
        ]
    )
    preds = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "y_pred_ev_oof": oof_ev,
            "y_pred_hard_oof": oof_hard,
            "residual_oof": y - oof_ev,
            "residual_sq_oof": (y - oof_ev) ** 2,
        }
    )
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    return metrics


def crossfit_predict_multinomial(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    folds: int,
    seed: int,
    emb_pca: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """K-fold OOF for multinomial logistic regression.

    Returns
    -------
    oof_ev   : expected-value predictions (float), for regression metrics
    oof_hard : argmax class predictions (int), for accuracy
    metrics  : dict with cv_rmse, cv_r2, cv_spearman, cv_accuracy
    """
    X = df[features]
    y = df[target].astype(int).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_ev = np.zeros(len(y), dtype=float)
    oof_hard = np.zeros(len(y), dtype=int)
    best_Cs = []

    for tr, te in kf.split(X):
        pipe = make_multinomial_pipe(features, seed=seed, emb_pca=emb_pca)
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])          # (n_te, n_classes)
        pipe_classes = pipe.named_steps["clf"].classes_
        oof_ev[te] = proba @ pipe_classes.astype(float)  # E[Y] = sum_k k * P(Y=k)
        oof_hard[te] = pipe_classes[np.argmax(proba, axis=1)]
        best_Cs.append(float(np.mean(pipe.named_steps["clf"].C_)))  # mean over classes

    return oof_ev, oof_hard, {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof_ev))),
        "cv_r2": float(r2_score(y, oof_ev)),
        "cv_spearman": float(pd.Series(y).corr(pd.Series(oof_ev), method="spearman")),
        "cv_accuracy": float(accuracy_score(y, oof_hard)),
        "mean_C_cv_folds": float(np.mean(best_Cs)),
    }


def run_judge_multinomial(
    df: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
    emb_pca: int | None,
    out_dir: Path,
    exp: str,
) -> pd.DataFrame:
    y = df[TARGET].astype(int).values
    oof_ev, oof_hard, m = crossfit_predict_multinomial(df, features, TARGET, folds, seed, emb_pca)

    metrics = pd.DataFrame(
        [
            {
                "experiment": exp,
                "n_samples": int(len(df)),
                "n_features_raw": int(len(features)),
                "cv_rmse": m["cv_rmse"],
                "cv_r2": m["cv_r2"],
                "cv_spearman": m["cv_spearman"],
                "cv_accuracy": m["cv_accuracy"],
                "mean_C_cv_folds": m["mean_C_cv_folds"],
                "mean_alpha_cv_folds": np.nan,
                "std_alpha_cv_folds": np.nan,
            }
        ]
    )
    preds = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "y_pred_ev_oof": oof_ev,
            "y_pred_hard_oof": oof_hard,
            "residual_oof": y - oof_ev,
            "residual_sq_oof": (y - oof_ev) ** 2,
        }
    )
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    return metrics


class SigmoidWeightedJudge:
    """
    Y = intercept + emb_pca @ alpha
          + sum_m [ s_m * beta0_m * sigmoid(beta1_m * ent_m_std + beta2_m * logppl_m_std) ]

    Per-model (beta0, beta1, beta2) triples.  Uncertainty inputs are z-scored
    per-model on training data so that beta1/beta2 are on a comparable scale.
    Fitted end-to-end with L-BFGS-B + L2 regularisation.
    """

    def __init__(
        self,
        model_names: list[str],
        n_pca: int = 64,
        l2: float = 1e-3,
        seed: int = 42,
        max_iter: int = 3000,
    ) -> None:
        self.model_names = list(model_names)
        self.n_pca = n_pca
        self.l2 = l2
        self.seed = seed
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    def _split_arrays(self, df: pd.DataFrame):
        scores = np.column_stack([df[f"s__{m}"].values for m in self.model_names]).astype(float)
        entropies = np.column_stack([df[f"entropy__{m}"].values for m in self.model_names]).astype(float)
        ppls = np.column_stack([df[f"ppl__{m}"].values for m in self.model_names]).astype(float)
        emb_cols = sorted(c for c in df.columns if c.startswith("emb__"))
        emb = df[emb_cols].values.astype(float) if emb_cols else None
        return scores, entropies, ppls, emb

    @staticmethod
    def _impute_cols(arr: np.ndarray, fill: np.ndarray) -> np.ndarray:
        """Replace non-finite values with per-column fill (broadcast)."""
        return np.where(np.isfinite(arr), arr, fill[np.newaxis, :])

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "SigmoidWeightedJudge":
        scores, entropies, ppls, emb = self._split_arrays(df)
        N, M = scores.shape

        # --- impute scores --------------------------------------------
        self.score_fill_ = np.where(np.isnan(np.nanmean(scores, 0)), 0.0, np.nanmean(scores, 0))
        scores = self._impute_cols(scores, self.score_fill_)

        # --- entropy: impute, then z-score ----------------------------
        self.ent_fill_ = np.where(np.isnan(np.nanmean(entropies, 0)), 0.0, np.nanmean(entropies, 0))
        entropies = self._impute_cols(entropies, self.ent_fill_)
        self.ent_mean_ = entropies.mean(0)
        self.ent_std_ = entropies.std(0) + 1e-8
        ent_s = (entropies - self.ent_mean_) / self.ent_std_

        # --- ppl: impute, log, z-score --------------------------------
        self.ppl_fill_ = np.where(np.isnan(np.nanmean(ppls, 0)), 1.0, np.nanmean(ppls, 0))
        ppls = np.clip(self._impute_cols(ppls, self.ppl_fill_), 1e-8, None)
        log_ppls = np.log(ppls)
        self.logppl_mean_ = log_ppls.mean(0)
        self.logppl_std_ = log_ppls.std(0) + 1e-8
        logppl_s = (log_ppls - self.logppl_mean_) / self.logppl_std_

        # --- PCA on embeddings ----------------------------------------
        if emb is not None and self.n_pca > 0:
            emb = np.where(np.isfinite(emb), emb, 0.0)
            n_pca = min(self.n_pca, emb.shape[1], N - 1)
            self.pca_ = PCA(n_components=n_pca, random_state=self.seed)
            self.emb_scaler_ = StandardScaler()
            emb_pca = self.emb_scaler_.fit_transform(self.pca_.fit_transform(emb))
        else:
            self.pca_ = None
            self.emb_scaler_ = None
            emb_pca = np.zeros((N, 0))

        n_emb = emb_pca.shape[1]
        self.n_emb_ = n_emb

        # --- parameter layout -----------------------------------------
        # index 0            : intercept
        # index 1..n_emb     : alpha  (PCA embedding coefficients)
        # index 1+n_emb ..   : [beta0_m, beta1_m, beta2_m] for m=0..M-1
        n_params = 1 + n_emb + 3 * M

        def _predict(p: np.ndarray) -> np.ndarray:
            ic = p[0]
            alpha = p[1 : 1 + n_emb]
            betas = p[1 + n_emb :].reshape(M, 3)
            yhat = np.full(N, ic)
            if n_emb > 0:
                yhat = yhat + emb_pca @ alpha
            for mi in range(M):
                b0, b1, b2 = betas[mi]
                w = b0 * expit(b1 * ent_s[:, mi] + b2 * logppl_s[:, mi])
                yhat = yhat + scores[:, mi] * w
            return yhat

        def loss(p: np.ndarray) -> float:
            resid = y - _predict(p)
            mse = float(np.dot(resid, resid)) / N
            reg = self.l2 * float(np.dot(p[1:], p[1:]))  # intercept not penalised
            return mse + reg

        # Initialise: intercept = mean(y), beta0 = 1, beta1 = beta2 = 0
        p0 = np.zeros(n_params)
        p0[0] = float(y.mean())
        for mi in range(M):
            p0[1 + n_emb + 3 * mi] = 1.0  # beta0

        result = minimize(
            loss, p0, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-7},
        )
        self.params_ = result.x
        self.opt_success_ = result.success
        self.opt_message_ = result.message
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores, entropies, ppls, emb = self._split_arrays(df)
        N, M = scores.shape

        scores = self._impute_cols(scores, self.score_fill_)
        entropies = self._impute_cols(entropies, self.ent_fill_)
        ent_s = (entropies - self.ent_mean_) / self.ent_std_

        ppls = np.clip(self._impute_cols(ppls, self.ppl_fill_), 1e-8, None)
        logppl_s = (np.log(ppls) - self.logppl_mean_) / self.logppl_std_

        if self.pca_ is not None and emb is not None:
            emb = np.where(np.isfinite(emb), emb, 0.0)
            emb_pca = self.emb_scaler_.transform(self.pca_.transform(emb))
        else:
            emb_pca = np.zeros((N, 0))

        p = self.params_
        n_emb = self.n_emb_
        ic = p[0]
        alpha = p[1 : 1 + n_emb]
        betas = p[1 + n_emb :].reshape(M, 3)

        yhat = np.full(N, ic)
        if n_emb > 0:
            yhat = yhat + emb_pca @ alpha
        for mi in range(M):
            b0, b1, b2 = betas[mi]
            w = b0 * expit(b1 * ent_s[:, mi] + b2 * logppl_s[:, mi])
            yhat = yhat + scores[:, mi] * w
        return yhat

    # ------------------------------------------------------------------
    def beta_summary(self) -> pd.DataFrame:
        """Per-model learned betas as a DataFrame."""
        M = len(self.model_names)
        betas = self.params_[1 + self.n_emb_ :].reshape(M, 3)
        return pd.DataFrame(
            {
                "model": self.model_names,
                "beta0": betas[:, 0],
                "beta1_entropy": betas[:, 1],
                "beta2_logppl": betas[:, 2],
            }
        )


def crossfit_predict_sigmoid(
    df: pd.DataFrame,
    model_names: list[str],
    target: str,
    folds: int,
    seed: int,
    n_pca: int,
    l2: float,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    y = df[target].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    beta_frames: list[pd.DataFrame] = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(df)):
        mdl = SigmoidWeightedJudge(model_names=model_names, n_pca=n_pca, l2=l2, seed=seed)
        mdl.fit(df.iloc[tr_idx].reset_index(drop=True), y[tr_idx])
        oof[te_idx] = mdl.predict(df.iloc[te_idx].reset_index(drop=True))
        bdf = mdl.beta_summary()
        bdf["fold"] = fold_i
        beta_frames.append(bdf)

    betas_cv = pd.concat(beta_frames, ignore_index=True)
    return oof, {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "cv_r2": float(r2_score(y, oof)),
    }, betas_cv


def run_judge_sigmoid(
    df: pd.DataFrame,
    model_names: list[str],
    folds: int,
    seed: int,
    n_pca: int,
    l2: float,
    out_dir: Path,
    exp: str,
) -> pd.DataFrame:
    y = df[TARGET].astype(float).values
    oof, m, betas_cv = crossfit_predict_sigmoid(df, model_names, TARGET, folds, seed, n_pca, l2)
    cv_spearman = float(pd.Series(y).corr(pd.Series(oof), method="spearman"))

    metrics = pd.DataFrame(
        [
            {
                "experiment": exp,
                "n_samples": int(len(df)),
                "n_features_raw": len(model_names) * 3 + n_pca,
                "cv_rmse": m["cv_rmse"],
                "cv_r2": m["cv_r2"],
                "cv_spearman": cv_spearman,
                "mean_alpha_cv_folds": np.nan,
                "std_alpha_cv_folds": np.nan,
            }
        ]
    )
    preds = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "y_pred_oof": oof,
            "residual_oof": y - oof,
            "residual_sq_oof": (y - oof) ** 2,
        }
    )
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    betas_cv.to_csv(out_dir / f"mrr_persuade_judge_sigmoid_betas_{exp}.csv", index=False)

    # Print beta summary (mean across folds)
    beta_mean = betas_cv.groupby("model")[["beta0", "beta1_entropy", "beta2_logppl"]].mean()
    print(f"\n[{exp}] Learned betas (mean across {folds} folds):")
    print(beta_mean.to_string())
    return metrics


def crossfit_predict(df: pd.DataFrame, features: list[str], target: str, folds: int, seed: int, emb_pca: int | None) -> tuple[np.ndarray, dict]:
    X = df[features]
    y = df[target].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    alphas = []
    for tr, te in kf.split(X):
        pipe = make_pipe(features, seed=seed, emb_pca=emb_pca)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict(X.iloc[te])
        alphas.append(float(pipe.named_steps["ridge"].alpha_))
    return oof, {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "cv_r2": float(r2_score(y, oof)),
        "mean_alpha_cv_folds": float(np.mean(alphas)),
        "std_alpha_cv_folds": float(np.std(alphas)),
    }


def run_judge(df: pd.DataFrame, features: list[str], folds: int, seed: int, emb_pca: int | None, out_dir: Path, exp: str) -> pd.DataFrame:
    y_pred_oof, m = crossfit_predict(df, features, TARGET, folds, seed, emb_pca)
    y = df[TARGET].astype(float).values
    cv_spearman = float(pd.Series(y).corr(pd.Series(y_pred_oof), method="spearman"))
    metrics = pd.DataFrame(
        [
            {
                "experiment": exp,
                "n_samples": int(len(df)),
                "n_features_raw": int(len(features)),
                "cv_rmse": m["cv_rmse"],
                "cv_r2": m["cv_r2"],
                "cv_spearman": cv_spearman,
                "mean_alpha_cv_folds": m["mean_alpha_cv_folds"],
                "std_alpha_cv_folds": m["std_alpha_cv_folds"],
            }
        ]
    )
    preds = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "y_pred_oof": y_pred_oof,
            "residual_oof": y - y_pred_oof,
            "residual_sq_oof": (y - y_pred_oof) ** 2,
        }
    )
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    return metrics


def run_raw_avg_baseline(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    out_dir: Path,
    exp: str,
) -> pd.DataFrame:
    """Baseline: average all judges' raw scores per essay, compare directly to human labels.
    No calibration or model fitting."""
    pivot = llm_df.pivot_table(index="id", columns="model", values="predicted_score", aggfunc="first")
    raw_avg = pivot.mean(axis=1, skipna=True).reset_index().rename(columns={0: "raw_avg"})
    merged = human_df.merge(raw_avg, on="id", how="inner")
    mask = np.isfinite(merged["raw_avg"].values) & np.isfinite(merged[TARGET].astype(float).values)
    merged = merged[mask].reset_index(drop=True)
    y = merged[TARGET].astype(float).values
    y_pred = merged["raw_avg"].values

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = float(r2_score(y, y_pred))
    sp   = float(pd.Series(y).corr(pd.Series(y_pred), method="spearman"))

    metrics = pd.DataFrame([{
        "experiment": exp,
        "n_samples": int(len(merged)),
        "n_features_raw": int(pivot.shape[1]),
        "cv_rmse": rmse, "cv_r2": r2, "cv_spearman": sp,
        "mean_alpha_cv_folds": np.nan, "std_alpha_cv_folds": np.nan,
    }])
    preds = pd.DataFrame({
        "id": merged["id"].values, "y_true": y, "y_pred_oof": y_pred,
        "residual_oof": y - y_pred, "residual_sq_oof": (y - y_pred) ** 2,
    })
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    return metrics


def run_raw_single_judge_avg(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    out_dir: Path,
    exp: str,
) -> pd.DataFrame:
    """Baseline: for each judge compute raw-score metrics vs human individually, then average.
    No calibration or model fitting."""
    model_names = sorted(llm_df["model"].unique().tolist())
    rows = []
    for model_name in model_names:
        sub = llm_df[llm_df["model"] == model_name][["id", "predicted_score"]].drop_duplicates("id")
        merged = human_df.merge(sub, on="id", how="inner")
        if len(merged) < 10:
            continue
        y_all      = merged[TARGET].astype(float).values
        y_pred_all = merged["predicted_score"].astype(float).values
        mask = np.isfinite(y_pred_all) & np.isfinite(y_all)
        y, y_pred = y_all[mask], y_pred_all[mask]
        if len(y) < 10:
            continue
        rows.append({
            "model": model_name,
            "n": len(merged),
            "r2":   float(r2_score(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "sp":   float(pd.Series(y).corr(pd.Series(y_pred), method="spearman")),
        })

    per_model_df = pd.DataFrame(rows)
    per_model_df.to_csv(out_dir / f"mrr_persuade_raw_single_per_model.csv", index=False)

    avg_r2   = float(per_model_df["r2"].mean())
    avg_rmse = float(per_model_df["rmse"].mean())
    avg_sp   = float(per_model_df["sp"].mean())

    metrics = pd.DataFrame([{
        "experiment": exp,
        "n_samples": int(per_model_df["n"].mean()),
        "n_features_raw": 1,
        "cv_rmse": avg_rmse, "cv_r2": avg_r2, "cv_spearman": avg_sp,
        "mean_alpha_cv_folds": np.nan, "std_alpha_cv_folds": np.nan,
    }])
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    return metrics


def run_judge_intercept_only(df: pd.DataFrame, folds: int, seed: int, out_dir: Path, exp: str) -> pd.DataFrame:
    y = df[TARGET].astype(float).values
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred_oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(df):
        mu = float(np.mean(y[tr]))
        y_pred_oof[te] = mu

    cv_rmse = float(np.sqrt(mean_squared_error(y, y_pred_oof)))
    cv_r2 = float(r2_score(y, y_pred_oof))
    cv_spearman = float(pd.Series(y).corr(pd.Series(y_pred_oof), method="spearman"))

    metrics = pd.DataFrame(
        [
            {
                "experiment": exp,
                "n_samples": int(len(df)),
                "n_features_raw": 0,
                "cv_rmse": cv_rmse,
                "cv_r2": cv_r2,
                "cv_spearman": cv_spearman,
                "mean_alpha_cv_folds": np.nan,
                "std_alpha_cv_folds": np.nan,
            }
        ]
    )
    preds = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "y_pred_oof": y_pred_oof,
            "residual_oof": y - y_pred_oof,
            "residual_sq_oof": (y - y_pred_oof) ** 2,
        }
    )
    metrics.to_csv(out_dir / f"mrr_persuade_judge_metrics_{exp}.csv", index=False)
    preds.to_csv(out_dir / f"mrr_persuade_judge_oof_{exp}.csv", index=False)
    return metrics


def run_two_stage_hardness(
    df: pd.DataFrame,
    judge_features: list[str],
    hardness_features: list[str],
    folds: int,
    seed: int,
    judge_emb_pca: int | None,
    hardness_emb_pca: int | None,
    out_dir: Path,
    eps: float = 1e-6,
) -> pd.DataFrame:
    y = df[TARGET].astype(float).values
    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_judge = np.zeros_like(y, dtype=float)
    oof_r2 = np.zeros_like(y, dtype=float)
    oof_h = np.zeros_like(y, dtype=float)

    for tr_idx, te_idx in outer.split(df):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        te = df.iloc[te_idx].reset_index(drop=True)

        tr_judge_oof, _ = crossfit_predict(tr, judge_features, TARGET, folds, seed, judge_emb_pca)
        tr_r2 = (tr[TARGET].astype(float).values - tr_judge_oof) ** 2
        tr_h = np.log(tr_r2 + eps)

        h_pipe = make_pipe(hardness_features, seed=seed, emb_pca=hardness_emb_pca)
        h_pipe.fit(tr[hardness_features], tr_h)

        j_pipe = make_pipe(judge_features, seed=seed, emb_pca=judge_emb_pca)
        j_pipe.fit(tr[judge_features], tr[TARGET].astype(float).values)
        te_j = j_pipe.predict(te[judge_features])
        te_r2 = (te[TARGET].astype(float).values - te_j) ** 2
        te_h = h_pipe.predict(te[hardness_features])

        oof_judge[te_idx] = te_j
        oof_r2[te_idx] = te_r2
        oof_h[te_idx] = te_h

    y_h = np.log(oof_r2 + eps)
    hard_r2 = float(r2_score(y_h, oof_h))
    hard_rmse = float(np.sqrt(mean_squared_error(y_h, oof_h)))
    hard_spear = float(pd.Series(oof_h).corr(pd.Series(oof_r2), method="spearman"))
    q90 = float(np.quantile(oof_h, 0.9))
    lift = float(np.mean(oof_r2[oof_h >= q90]) / np.mean(oof_r2))

    out = pd.DataFrame(
        [
            {
                "n_samples": int(len(df)),
                "judge_cv_rmse_outer": float(np.sqrt(mean_squared_error(y, oof_judge))),
                "judge_cv_r2_outer": float(r2_score(y, oof_judge)),
                "hardness_cv_rmse_log_resid_sq": hard_rmse,
                "hardness_cv_r2_log_resid_sq": hard_r2,
                "hardness_spearman_pred_vs_resid_sq": hard_spear,
                "hardness_top_decile_resid_sq_lift": lift,
            }
        ]
    )
    rows = pd.DataFrame(
        {
            "id": df["id"].values,
            "y_true": y,
            "judge_pred_oof_outer": oof_judge,
            "residual_sq_oof_outer": oof_r2,
            "hardness_log_resid_sq_pred_oof_outer": oof_h,
            "hardness_log_resid_sq_true_oof_outer": y_h,
        }
    )
    out.to_csv(out_dir / "mrr_persuade_two_stage_hardness_metrics.csv", index=False)
    rows.to_csv(out_dir / "mrr_persuade_two_stage_hardness_oof_rows.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PERSUADE MRR + hardness.")
    parser.add_argument("--csv-path", type=Path, default=Path("datasets/persuade/data/persuade_corpus_2.0_train.csv"))
    parser.add_argument("--llm-path", type=Path, default=Path("datasets/persuade/results/llm_scores.jsonl"))
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=Path("datasets/persuade/results/openai_text_embedding_3_large_essay_embeddings.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/persuade/results"))
    parser.add_argument("--grade", type=int, default=10, help="Use -1 for all grades")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-emb-pca", type=int, default=64)
    parser.add_argument("--hardness-emb-pca", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset from llm_scores.jsonl",
    )
    parser.add_argument("--sigmoid-l2", type=float, default=1e-3, help="L2 penalty for SigmoidWeightedJudge")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grade = None if args.grade == -1 else args.grade

    human = load_human(args.csv_path.resolve(), grade=grade)
    llm = load_llm_scores(args.llm_path.resolve(), models=args.models)
    emb = load_embeddings(args.emb_path.resolve())

    # Full judge: scores + uncertainty + embedding intercept
    judge_df, judge_features = build_feature_table(
        human_df=human,
        llm_df=llm,
        emb_df=emb,
        use_uncertainty=True,
    )
    print(f"Full judge rows={len(judge_df)} | features(raw)={len(judge_features)}")

    judge_metrics_full = run_judge(
        df=judge_df,
        features=judge_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=args.judge_emb_pca,
        out_dir=out_dir,
        exp="scores_uncertainty_emb_intercept",
    )

    # Ablation 1: LLM score/uncertainty only, no embedding intercept.
    llm_only_df, llm_only_features = build_feature_table(
        human_df=human,
        llm_df=llm,
        emb_df=None,
        use_uncertainty=True,
    )
    judge_metrics_llm_only = run_judge(
        df=llm_only_df,
        features=llm_only_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=None,
        out_dir=out_dir,
        exp="scores_uncertainty_no_emb_intercept",
    )

    # Ablation 2: embedding-only ridge.
    emb_only_df, emb_only_features = build_feature_table(
        human_df=human,
        llm_df=llm,
        emb_df=emb,
        use_uncertainty=False,
    )
    emb_only_features = [c for c in emb_only_features if c.startswith("emb__")]
    judge_metrics_emb_only = run_judge(
        df=emb_only_df,
        features=emb_only_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=args.judge_emb_pca,
        out_dir=out_dir,
        exp="emb_only",
    )

    # Baseline A: raw judge average (uncalibrated)
    judge_metrics_raw_avg = run_raw_avg_baseline(
        human_df=human,
        llm_df=llm,
        out_dir=out_dir,
        exp="raw_judge_avg",
    )

    # Baseline B: raw single-judge metrics averaged across all judges
    judge_metrics_raw_single_avg = run_raw_single_judge_avg(
        human_df=human,
        llm_df=llm,
        out_dir=out_dir,
        exp="raw_single_judge_avg",
    )

    # Ablation 3: intercept-only baseline.
    judge_metrics_intercept_only = run_judge_intercept_only(
        df=judge_df,
        folds=args.cv_folds,
        seed=args.seed,
        out_dir=out_dir,
        exp="intercept_only",
    )

    # Ordinal regression (proportional-odds): same features as ridge ablations
    judge_metrics_ord_full = run_judge_ordinal(
        df=judge_df,
        features=judge_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=args.judge_emb_pca,
        out_dir=out_dir,
        exp="ordinal_scores_uncertainty_emb",
    )
    judge_metrics_ord_llm_only = run_judge_ordinal(
        df=llm_only_df,
        features=llm_only_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=None,
        out_dir=out_dir,
        exp="ordinal_scores_uncertainty_no_emb",
    )

    # Multinomial logistic regression: same features as ridge ablations
    judge_metrics_mn_full = run_judge_multinomial(
        df=judge_df,
        features=judge_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=args.judge_emb_pca,
        out_dir=out_dir,
        exp="multinomial_scores_uncertainty_emb",
    )
    judge_metrics_mn_llm_only = run_judge_multinomial(
        df=llm_only_df,
        features=llm_only_features,
        folds=args.cv_folds,
        seed=args.seed,
        emb_pca=None,
        out_dir=out_dir,
        exp="multinomial_scores_uncertainty_no_emb",
    )

    # Sigmoid-weighted model: PCA emb + sum_m [ s_m * beta0_m * sigmoid(beta1_m*ent + beta2_m*logppl) ]
    model_names_for_sigmoid = [c[len("s__"):] for c in judge_df.columns if c.startswith("s__")]
    judge_metrics_sigmoid = run_judge_sigmoid(
        df=judge_df,
        model_names=model_names_for_sigmoid,
        folds=args.cv_folds,
        seed=args.seed,
        n_pca=args.judge_emb_pca,
        l2=args.sigmoid_l2,
        out_dir=out_dir,
        exp="sigmoid_weighted_emb",
    )

    # Ablation: sigmoid model without embeddings (scores + uncertainty only)
    llm_sigmoid_df, _ = build_feature_table(human_df=human, llm_df=llm, emb_df=None, use_uncertainty=True)
    judge_metrics_sigmoid_no_emb = run_judge_sigmoid(
        df=llm_sigmoid_df,
        model_names=model_names_for_sigmoid,
        folds=args.cv_folds,
        seed=args.seed,
        n_pca=0,
        l2=args.sigmoid_l2,
        out_dir=out_dir,
        exp="sigmoid_weighted_no_emb",
    )

    # Single-judge ordinal ablation: run each LLM judge alone, then average
    model_names_all = sorted(llm["model"].unique().tolist())
    single_metrics_list = []
    single_emb_metrics_list = []

    for model_name in model_names_all:
        llm_m = llm[llm["model"] == model_name]

        df_m, feats_m = build_feature_table(human, llm_m, None, use_uncertainty=True)
        met_m = run_judge_ordinal(
            df=df_m, features=feats_m, folds=args.cv_folds, seed=args.seed,
            emb_pca=None, out_dir=out_dir, exp=f"ordinal_single_{model_name}",
        )
        met_m = met_m.copy(); met_m["model_name"] = model_name
        single_metrics_list.append(met_m)

        df_m_emb, feats_m_emb = build_feature_table(human, llm_m, emb, use_uncertainty=True)
        met_m_emb = run_judge_ordinal(
            df=df_m_emb, features=feats_m_emb, folds=args.cv_folds, seed=args.seed,
            emb_pca=args.judge_emb_pca, out_dir=out_dir, exp=f"ordinal_single_{model_name}_emb",
        )
        met_m_emb = met_m_emb.copy(); met_m_emb["model_name"] = model_name
        single_emb_metrics_list.append(met_m_emb)

    single_df     = pd.concat(single_metrics_list,     ignore_index=True)
    single_emb_df = pd.concat(single_emb_metrics_list, ignore_index=True)
    single_df.to_csv(out_dir / "mrr_persuade_single_judge_ordinal_per_model.csv", index=False)
    single_emb_df.to_csv(out_dir / "mrr_persuade_single_judge_emb_ordinal_per_model.csv", index=False)

    single_avg_r2    = float(single_df["cv_r2"].mean())
    single_avg_rmse  = float(single_df["cv_rmse"].mean())
    single_avg_spear = float(single_df["cv_spearman"].mean())
    single_avg_acc   = float(single_df["cv_accuracy"].mean())
    single_emb_avg_r2    = float(single_emb_df["cv_r2"].mean())
    single_emb_avg_rmse  = float(single_emb_df["cv_rmse"].mean())
    single_emb_avg_spear = float(single_emb_df["cv_spearman"].mean())
    single_emb_avg_acc   = float(single_emb_df["cv_accuracy"].mean())

    print(f"\n=== Single-judge ordinal avg ({len(model_names_all)} models) ===")
    print(f"  R²={single_avg_r2:.4f}  RMSE={single_avg_rmse:.4f}  Spearman={single_avg_spear:.4f}")
    print(f"\n=== Single-judge + emb ordinal avg ({len(model_names_all)} models) ===")
    print(f"  R²={single_emb_avg_r2:.4f}  RMSE={single_emb_avg_rmse:.4f}  Spearman={single_emb_avg_spear:.4f}")

    judge_metrics_single_avg = pd.DataFrame([{
        "experiment": "ordinal_single_judge_avg",
        "n_samples": int(single_df["n_samples"].mean()),
        "n_features_raw": int(single_df["n_features_raw"].mean()),
        "cv_rmse": single_avg_rmse, "cv_r2": single_avg_r2,
        "cv_spearman": single_avg_spear, "cv_accuracy": single_avg_acc,
        "mean_C_cv_folds": np.nan,
        "mean_alpha_cv_folds": float(single_df["mean_alpha_cv_folds"].mean()),
        "std_alpha_cv_folds": np.nan,
    }])
    judge_metrics_single_emb_avg = pd.DataFrame([{
        "experiment": "ordinal_single_judge_emb_avg",
        "n_samples": int(single_emb_df["n_samples"].mean()),
        "n_features_raw": int(single_emb_df["n_features_raw"].mean()),
        "cv_rmse": single_emb_avg_rmse, "cv_r2": single_emb_avg_r2,
        "cv_spearman": single_emb_avg_spear, "cv_accuracy": single_emb_avg_acc,
        "mean_C_cv_folds": np.nan,
        "mean_alpha_cv_folds": float(single_emb_df["mean_alpha_cv_folds"].mean()),
        "std_alpha_cv_folds": np.nan,
    }])

    # Hardness: embedding-only as requested.
    hardness_features = [c for c in judge_df.columns if c.startswith("emb__")]
    hard_metrics = run_two_stage_hardness(
        df=judge_df,
        judge_features=judge_features,
        hardness_features=hardness_features,
        folds=args.cv_folds,
        seed=args.seed,
        judge_emb_pca=args.judge_emb_pca,
        hardness_emb_pca=args.hardness_emb_pca,
        out_dir=out_dir,
    )

    summary = {
        "target": TARGET,
        "grade_filter": grade,
        "n_rows_used": int(len(judge_df)),
        "raw_avg_cv_r2":      float(judge_metrics_raw_avg["cv_r2"].iloc[0]),
        "raw_avg_cv_rmse":    float(judge_metrics_raw_avg["cv_rmse"].iloc[0]),
        "raw_avg_cv_spearman":float(judge_metrics_raw_avg["cv_spearman"].iloc[0]),
        "raw_single_avg_cv_r2":      float(judge_metrics_raw_single_avg["cv_r2"].iloc[0]),
        "raw_single_avg_cv_rmse":    float(judge_metrics_raw_single_avg["cv_rmse"].iloc[0]),
        "raw_single_avg_cv_spearman":float(judge_metrics_raw_single_avg["cv_spearman"].iloc[0]),
        "judge_cv_r2": float(judge_metrics_full["cv_r2"].iloc[0]),
        "judge_cv_rmse": float(judge_metrics_full["cv_rmse"].iloc[0]),
        "judge_cv_spearman": float(judge_metrics_full["cv_spearman"].iloc[0]),
        "judge_llm_only_cv_r2": float(judge_metrics_llm_only["cv_r2"].iloc[0]),
        "judge_llm_only_cv_rmse": float(judge_metrics_llm_only["cv_rmse"].iloc[0]),
        "judge_llm_only_cv_spearman": float(judge_metrics_llm_only["cv_spearman"].iloc[0]),
        "judge_emb_only_cv_r2": float(judge_metrics_emb_only["cv_r2"].iloc[0]),
        "judge_emb_only_cv_rmse": float(judge_metrics_emb_only["cv_rmse"].iloc[0]),
        "judge_emb_only_cv_spearman": float(judge_metrics_emb_only["cv_spearman"].iloc[0]),
        "judge_intercept_only_cv_r2": float(judge_metrics_intercept_only["cv_r2"].iloc[0]),
        "judge_intercept_only_cv_rmse": float(judge_metrics_intercept_only["cv_rmse"].iloc[0]),
        "judge_intercept_only_cv_spearman": float(judge_metrics_intercept_only["cv_spearman"].iloc[0]),
        "judge_sigmoid_emb_cv_r2": float(judge_metrics_sigmoid["cv_r2"].iloc[0]),
        "judge_sigmoid_emb_cv_rmse": float(judge_metrics_sigmoid["cv_rmse"].iloc[0]),
        "judge_sigmoid_emb_cv_spearman": float(judge_metrics_sigmoid["cv_spearman"].iloc[0]),
        "judge_sigmoid_no_emb_cv_r2": float(judge_metrics_sigmoid_no_emb["cv_r2"].iloc[0]),
        "judge_sigmoid_no_emb_cv_rmse": float(judge_metrics_sigmoid_no_emb["cv_rmse"].iloc[0]),
        "judge_sigmoid_no_emb_cv_spearman": float(judge_metrics_sigmoid_no_emb["cv_spearman"].iloc[0]),
        "judge_ord_emb_cv_r2": float(judge_metrics_ord_full["cv_r2"].iloc[0]),
        "judge_ord_emb_cv_rmse": float(judge_metrics_ord_full["cv_rmse"].iloc[0]),
        "judge_ord_emb_cv_spearman": float(judge_metrics_ord_full["cv_spearman"].iloc[0]),
        "judge_ord_emb_cv_accuracy": float(judge_metrics_ord_full["cv_accuracy"].iloc[0]),
        "judge_ord_no_emb_cv_r2": float(judge_metrics_ord_llm_only["cv_r2"].iloc[0]),
        "judge_ord_no_emb_cv_rmse": float(judge_metrics_ord_llm_only["cv_rmse"].iloc[0]),
        "judge_ord_no_emb_cv_spearman": float(judge_metrics_ord_llm_only["cv_spearman"].iloc[0]),
        "judge_ord_no_emb_cv_accuracy": float(judge_metrics_ord_llm_only["cv_accuracy"].iloc[0]),
        "judge_mn_emb_cv_r2": float(judge_metrics_mn_full["cv_r2"].iloc[0]),
        "judge_mn_emb_cv_rmse": float(judge_metrics_mn_full["cv_rmse"].iloc[0]),
        "judge_mn_emb_cv_spearman": float(judge_metrics_mn_full["cv_spearman"].iloc[0]),
        "judge_mn_emb_cv_accuracy": float(judge_metrics_mn_full["cv_accuracy"].iloc[0]),
        "judge_mn_no_emb_cv_r2": float(judge_metrics_mn_llm_only["cv_r2"].iloc[0]),
        "judge_mn_no_emb_cv_rmse": float(judge_metrics_mn_llm_only["cv_rmse"].iloc[0]),
        "judge_mn_no_emb_cv_spearman": float(judge_metrics_mn_llm_only["cv_spearman"].iloc[0]),
        "judge_mn_no_emb_cv_accuracy": float(judge_metrics_mn_llm_only["cv_accuracy"].iloc[0]),
        "hardness_cv_r2_log_resid_sq": float(hard_metrics["hardness_cv_r2_log_resid_sq"].iloc[0]),
        "hardness_spearman_pred_vs_resid_sq": float(hard_metrics["hardness_spearman_pred_vs_resid_sq"].iloc[0]),
        "hardness_top_decile_resid_sq_lift": float(hard_metrics["hardness_top_decile_resid_sq_lift"].iloc[0]),
        "judge_single_ord_avg_cv_r2": single_avg_r2,
        "judge_single_ord_avg_cv_rmse": single_avg_rmse,
        "judge_single_ord_avg_cv_spearman": single_avg_spear,
        "judge_single_ord_emb_avg_cv_r2": single_emb_avg_r2,
        "judge_single_ord_emb_avg_cv_rmse": single_emb_avg_rmse,
        "judge_single_ord_emb_avg_cv_spearman": single_emb_avg_spear,
    }
    (out_dir / "mrr_persuade_summary.json").write_text(json.dumps(summary, indent=2))

    judge_metrics_all = pd.concat(
        [
            judge_metrics_full,
            judge_metrics_llm_only,
            judge_metrics_emb_only,
            judge_metrics_intercept_only,
            judge_metrics_raw_avg,
            judge_metrics_raw_single_avg,
            judge_metrics_ord_full,
            judge_metrics_ord_llm_only,
            judge_metrics_single_avg,
            judge_metrics_single_emb_avg,
            judge_metrics_sigmoid,
            judge_metrics_sigmoid_no_emb,
            judge_metrics_mn_full,
            judge_metrics_mn_llm_only,
        ],
        axis=0,
        ignore_index=True,
    )
    judge_metrics_all.to_csv(out_dir / "mrr_persuade_judge_metrics_all.csv", index=False)

    print(judge_metrics_all.to_string(index=False))
    print(hard_metrics.to_string(index=False))
    print(f"Saved summary: {out_dir / 'mrr_persuade_summary.json'}")


if __name__ == "__main__":
    main()
