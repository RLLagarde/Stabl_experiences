#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PED-4 late-fusion pipeline (clean version)

Step 1  – Per-stim out-of-fold (OOF) predictions
           • 5 × 200 RepeatedStratifiedKFold  → 1,000 folds
           • Pipeline: Imputer ▸ StandardScaler ▸ Lasso(α = 0.01)
Step 2  – Late fusion meta-model
           • Logistic regression (saga) + grid search on {L1, L2, ElasticNet}
           • 5-fold external CV to keep evaluation unbiased
Step 3  – Metrics + export
           • ROC-AUC, Mann-Whitney p-value, Pearson r
           • Mean / median fusion weights, intercepts, predictions (CSV)
"""

# ───────────────────────────── Imports ──────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from scipy.stats import mannwhitneyu, pearsonr

# ──────────────────────────── I/O & const ───────────────────────────
ROOT   = Path("./files")
STIMS  = ["16", "17", "18"]
CSV_X  = "data4.csv"
CSV_M  = "selectedFeats.csv"
Y_FILE = Path("./outcome4.csv")
SEED   = 42

# ──────────────────────── Data loading ──────────────────────────────
y_full = pd.read_csv(Y_FILE)["group"].astype(int)
X_full = pd.read_csv(ROOT / CSV_X, index_col=0)

# ──────────────────── 1) OOF per stim (Lasso) ───────────────────────
prep = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("std", StandardScaler())])

cache = {s: [] for s in STIMS}
rskf  = RepeatedStratifiedKFold(n_splits=5, n_repeats=200, random_state=SEED)

for fold, (tr, te) in enumerate(rskf.split(X_full, y_full), 1):
    y_tr = y_full.iloc[tr]
    for stim in STIMS:
        mask_path = ROOT / stim / CSV_M
        if not mask_path.exists():
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue
        feats = pd.read_csv(mask_path, index_col=0).astype(bool)
        cols  = feats.columns[feats.any()].tolist()
        if not cols:
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue

        X_tr, X_te = X_full.loc[tr, cols], X_full.loc[te, cols]
        model = Pipeline([("prep", prep),
                          ("reg",  Lasso(alpha=0.01, max_iter=100_000))])
        model.fit(X_tr, y_tr)
        cache[stim].append((te, model.predict(X_te)))

# median aggregation → one OOF score / stim / sample
oof_proba = {}
for stim in STIMS:
    stack = np.full((len(cache[stim]), len(y_full)), np.nan)
    for k, (idx, pred) in enumerate(cache[stim]):
        stack[k, idx] = pred
    oof_proba[stim] = np.nanmedian(stack, axis=0)

P = np.column_stack([oof_proba[s] for s in STIMS])
P[np.isnan(P)] = 0.5
y = y_full.values

# ─────────────── 2) Late-fusion meta-model ──────────────────────────
def mw_scorer(est, X, y):
    p = est.predict_proba(X)[:, 1]
    return mannwhitneyu(p[y == 0], p[y == 1], alternative="two-sided")[1]

meta_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(solver="saga",
                                  class_weight="balanced",
                                  max_iter=20_000,
                                  random_state=SEED))
])
param_grid = {
    "clf__penalty":  ["l1", "l2", "elasticnet"],
    "clf__C":        np.logspace(-4, 3, 40),
    "clf__l1_ratio": [0.2, 0.5, 0.8]
}

kf_ext     = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv   = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
meta_oof   = np.zeros_like(y, dtype=float)
weights, icpts = [], []

for tr, te in kf_ext.split(P):
    gs = GridSearchCV(meta_pipe, param_grid, scoring=make_scorer(mw_scorer,
                    greater_is_better=False), cv=inner_cv, n_jobs=-1)
    gs.fit(P[tr], y[tr])
    best = gs.best_estimator_
    meta_oof[te] = best.predict_proba(P[te])[:, 1]
    weights.append(best.named_steps["clf"].coef_[0])
    icpts.append(best.named_steps["clf"].intercept_[0])

# ──────────────── 3) Metrics & exports ──────────────────────────────
auc  = roc_auc_score(y, meta_oof)
mw_p = mannwhitneyu(meta_oof[y == 0], meta_oof[y == 1], alternative="two-sided")[1]
r, rp = pearsonr(y, meta_oof)

print(f"AUC (OOF)      : {auc:.3f}")
print(f"Mann–Whitney p : {mw_p:.3e}")
print(f"Pearson r      : {r:.3f}  p={rp:.3e}")

w_mean, w_med = np.mean(weights, 0), np.median(weights, 0)
b_mean, b_med = np.mean(icpts), np.median(icpts)
pd.Series(w_mean,  STIMS).to_csv("lf_weights_mean.csv")
pd.Series(w_med,  STIMS).to_csv("lf_weights_median.csv")
pd.Series([b_mean], index=["intercept_mean"]).to_csv("lf_intercept_mean.csv")
pd.Series([b_med],  index=["intercept_median"]).to_csv("lf_intercept_median.csv")
pd.DataFrame({"y_true": y, "y_pred": meta_oof}).to_csv("lf_predictions.csv", index=False)

print("Exports: lf_weights_*.csv  |  lf_intercept_*.csv  |  lf_predictions.csv")
