# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
02_comparative_study.py
========================
Rigorous comparison of GEMEX vs SHAP vs LIME vs ELI5 on:
  - Cleveland Heart Disease (13 features, GBM + MLP + DeepMLP)
  - Pima Indians Diabetes   (8 features,  GBM + MLP + DeepMLP)
  - Wisconsin Breast Cancer (10 features, GBM + MLP + DeepMLP)

Metrics (all from peer-reviewed literature)
--------------------------------------------
  Faithfulness   — Alvarez Melis, D., & Jaakkola, T. (2018). Towards robust interpretability with self-explaining 
                   neural networks. Advances in neural information processing systems, 31.
  Monotonicity   — Luss, R. et al. (2021). Leveraging latent features for local explanations. 
                   In the 27th ACM SIGKDD (pp. 1139-1149).
  Completeness   — Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. 
                   In International conference on machine learning (pp. 3319-3328). PMLR.
  Stability      — Alvarez Melis, D., & Jaakkola, T. (2018). Towards robust interpretability with self-explaining 
                   neural networks. Advances in neural information processing systems, 31.

Outputs
-------
  Per-metric comparison tables  (console + CSV)
  Per-dataset grouped bar charts (one per metric)
  Attribution comparison plot   (side-by-side bars)
  Radar chart                   (all metrics per method)
  Separate metric heatmap       (method × dataset)
  Wilcoxon signed-rank tests    (GEMEX vs each baseline)
  Bootstrap 95% confidence intervals

Requirements
------------
  pip install gemex shap lime eli5 scikit-learn pandas matplotlib scipy

Usage
-----
  python 02_comparative_study.py
  python 02_comparative_study.py --model all --n-instances 30
  python 02_comparative_study.py --model both --theme light
  python 02_comparative_study.py --heart-path cleveland_heart.csv
                                  --pima-path pima_diabetes.csv
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')
from gemex import Explainer, GemexConfig

METHODS = ['GEMEX', 'SHAP', 'LIME', 'ELI5']

DARK = dict(
    bg='#0D0D1A', panel='#131326', grid='#1E1E38', border='#2E2E55',
    text='#E8E8F0', text2='#9999BB', text3='#444466',
    c_gemex='#00C896', c_shap='#378ADD', c_lime='#BA7517', c_eli5='#7F77DD',
)
LIGHT = dict(
    bg='#F4F4F9', panel='#FFFFFF', grid='#EBEBF5', border='#CCCCDD',
    text='#1A1A2E', text2='#555577', text3='#AAAACC',
    c_gemex='#0A7A5A', c_shap='#185FA5', c_lime='#6B4500', c_eli5='#3C3489',
)

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_heart(path=None):
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']
    if path and os.path.exists(path):
        df = pd.read_csv(path).dropna()
        if 'target' in df.columns:
            df['target'] = (df['target'] > 0).astype(int)
    else:
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
               'heart-disease/processed.cleveland.data')
        try:
            df = pd.read_csv(url, names=cols, na_values='?').dropna()
            df['target'] = (df['target'] > 0).astype(int)
        except Exception:
            return None, None, None, None
    feat = [c for c in df.columns if c != 'target']
    return df[feat].values.astype(float), df['target'].values.astype(int), feat, 'Heart Disease'


def load_pima(path=None):
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if df.iloc[0,0] == 'Pregnancies':
            df = df.iloc[1:].reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
    else:
        try:
            url = ('https://raw.githubusercontent.com/jbrownlee/Datasets/'
                   'master/pima-indians-diabetes.data.csv')
            df = pd.read_csv(url, names=cols)
        except Exception:
            return None, None, None, None
    feat = [c for c in df.columns if c != 'Outcome']
    return df[feat].values.astype(float), df['Outcome'].values.astype(int), feat, 'Pima Diabetes'


def load_breast():
    bc   = load_breast_cancer()
    X    = bc.data[:, :10].astype(float)
    feat = list(bc.feature_names[:10])
    return X, bc.target.astype(int), feat, 'Breast Cancer'


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def metric_faithfulness(model, X, atfs, n=30):
    atfs  = np.array(atfs, dtype=float)
    if atfs.ndim == 3:
        atfs = (atfs[1] if (atfs.shape[0] == 2 and
                atfs.shape[0] != atfs.shape[1])
                else atfs[:, :, 1])
    idx   = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    corrs = []
    for i in idx:
        x     = X[i].copy()
        p0    = model.predict_proba(x.reshape(1,-1))[0, 1]
        order = np.argsort(np.abs(atfs[i]))[::-1]
        drops, scores = [], []
        xm = x.copy()
        for fi in order[:min(8, len(order))]:
            xm[fi] = 0.0
            drops.append(abs(p0 - model.predict_proba(xm.reshape(1,-1))[0, 1]))
            scores.append(abs(float(np.asarray(atfs[i][fi]).flat[0])))
        scores_arr = np.array(scores, dtype=float)
        drops_arr  = np.array(drops,  dtype=float)
        if len(set(scores_arr.tolist())) > 1:
            c, _ = spearmanr(scores_arr, drops_arr)
            if not np.isnan(c): corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def metric_monotonicity(model, X, atfs, n=30):
    atfs   = np.array(atfs, dtype=float)
    if atfs.ndim == 3:
        atfs = (atfs[1] if (atfs.shape[0] == 2 and
                atfs.shape[0] != atfs.shape[1])
                else atfs[:, :, 1])
    idx    = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    checks = []
    for i in idx:
        x  = X[i].copy()
        p0 = model.predict_proba(x.reshape(1,-1))[0, 1]
        for fi in range(X.shape[1]):
            eps = 0.2 * (X[:, fi].std() + 1e-8)
            xp  = x.copy(); xp[fi] += eps
            pp  = model.predict_proba(xp.reshape(1,-1))[0, 1]
            if np.sign(atfs[i][fi]) != 0:
                checks.append(int(np.sign(atfs[i][fi]) == np.sign(pp - p0)))
    return float(np.mean(checks)) if checks else 0.0


def metric_completeness(model, X, atfs, baseline, n=30):
    atfs = np.array(atfs, dtype=float)
    if atfs.ndim == 3:
        atfs = (atfs[1] if (atfs.shape[0] == 2 and
                atfs.shape[0] != atfs.shape[1])
                else atfs[:, :, 1])
    idx  = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    pb   = model.predict_proba(baseline.reshape(1,-1))[0, 1]
    errs = []
    for i in idx:
        px = model.predict_proba(X[i].reshape(1,-1))[0, 1]
        errs.append(abs(atfs[i].sum() - (px - pb)))
    return float(np.mean(errs))


def metric_stability(X, atfs, n_pairs=40):
    atfs = np.array(atfs, dtype=float)
    if atfs.ndim == 3:
        atfs = (atfs[1] if (atfs.shape[0] == 2 and
                atfs.shape[0] != atfs.shape[1])
                else atfs[:, :, 1])
    n   = min(len(X), 80)
    idx = np.random.RandomState(0).choice(len(X), n, replace=False)
    rs  = np.random.RandomState(1)
    ratios = []
    for _ in range(n_pairs):
        i, j = rs.choice(n, 2, replace=False)
        dx = np.linalg.norm(X[idx[i]] - X[idx[j]]) + 1e-10
        da = np.linalg.norm(atfs[idx[i]] - atfs[idx[j]])
        ratios.append(da / dx)
    return float(np.mean(ratios))


def bootstrap_ci(vals, n_boot=500, ci=0.95):
    bs = [np.mean(np.random.choice(vals, len(vals), replace=True))
          for _ in range(n_boot)]
    lo = np.percentile(bs, (1-ci)/2*100)
    hi = np.percentile(bs, (1-(1-ci)/2)*100)
    return float(np.mean(vals)), float(lo), float(hi)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    ps = np.sqrt(((na-1)*np.std(a)**2 + (nb-1)*np.std(b)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (ps + 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# EXPLAINERS
# ═══════════════════════════════════════════════════════════════════════

def get_gemex(model, X_tr, X_te, feat, n, seed):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    cfg = GemexConfig(n_geodesic_steps=15, n_reference_samples=30,
                      interaction_order=1, verbose=False)
    exp = Explainer(model, data_type='tabular', feature_names=feat,
                    class_names=['No','Yes'], config=cfg)
    atfs, riccivals = [], []
    for i in idx:
        r = exp.explain(X_te[i], X_reference=X_tr)
        atfs.append(r.gsf_scores)
        riccivals.append(abs(r.manifold_curvature))
    return np.array(atfs), X_te[idx], float(np.mean(riccivals))


def get_shap(model, X_tr, X_te, n, seed, model_type='gbm'):
    """
    Model-aware SHAP wrapper — one correct explainer per model family.

    GBM      → TreeExplainer, interventional  (exact, fast on boosted trees)
    MLP /
    DeepMLP  → KernelExplainer in regression mode: wraps predict_proba to
               return only class-1 scalar, so SHAP always produces a single
               (n, d) array — no list, no 3-D ambiguity, no version issues.

    Output is always (n_instances, n_features) float64.
    """
    rng   = np.random.RandomState(seed)
    idx   = rng.choice(len(X_te), min(n, len(X_te)), replace=False)
    X_sub = X_te[idx]

    if model_type == 'gbm':
        exp  = shap.TreeExplainer(model,
                                  data=X_tr[:50],
                                  feature_perturbation='interventional')
        vals = exp.shap_values(X_sub, check_additivity=False)

    else:
        # Neural networks: wrap predict_proba to return only the class-1
        # probability scalar.  This forces KernelExplainer into regression
        # mode, giving a clean (n, d) array regardless of SHAP version.
        def predict_class1(X):
            return model.predict_proba(X)[:, 1]

        bg_idx = rng.choice(len(X_tr), min(30, len(X_tr)), replace=False)
        exp    = shap.KernelExplainer(predict_class1, X_tr[bg_idx])
        vals   = exp.shap_values(X_sub, nsamples=100, silent=True)

    # ── Normalise to (n_instances, n_features) ────────────────────────
    vals = np.array(vals, dtype=float)

    if vals.ndim == 3:
        # Two possible layouts depending on SHAP version:
        #   (n_classes, n, d) — first axis is classes → take [1]
        #   (n, d, n_classes) — last axis is classes  → take [:,:,1]
        if vals.shape[0] == 2 and vals.shape[0] != vals.shape[1]:
            atfs = vals[1]          # (n_classes, n, d) layout
        else:
            atfs = vals[:, :, 1]   # (n, d, n_classes) layout
    elif vals.ndim == 2:
        atfs = vals                 # already (n, d) — regression mode
    elif vals.ndim == 1:
        atfs = vals.reshape(1, -1) # single instance edge case
    else:
        atfs = vals

    return atfs.astype(float), X_sub


def get_lime(model, X_tr, X_te, feat, n, seed):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    exp = LimeTabularExplainer(X_tr, feature_names=feat,
                               class_names=['No','Yes'],
                               discretize_continuous=True, random_state=seed)
    atfs = []
    for i in idx:
        ex  = exp.explain_instance(X_te[i], model.predict_proba,
                                   num_features=len(feat), num_samples=300)
        atf = np.zeros(len(feat))
        for fi_str, val in ex.as_list():
            for j, fn in enumerate(feat):
                if fn in str(fi_str):
                    atf[j] = val; break
        atfs.append(atf)
    return np.array(atfs), X_te[idx]


def get_eli5(model, X_tr, X_te, y_te, n, seed):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    from eli5.sklearn import PermutationImportance
    pi   = PermutationImportance(model, random_state=seed, n_iter=5)
    pi.fit(X_te, y_te)
    base = X_tr.mean(axis=0)
    atfs = []
    for i in idx:
        signs = np.sign(X_te[i] - base); signs[signs==0] = 1
        atfs.append(pi.feature_importances_ * signs)
    return np.array(atfs), X_te[idx]


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def _apply_theme(fig, axes_list, t):
    fig.patch.set_facecolor(t['bg'])
    for ax in (axes_list if hasattr(axes_list, '__iter__') else [axes_list]):
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.6)
        ax.tick_params(colors=t['text2'], labelsize=9)
        ax.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)


def plot_metric_bars(records, ds_name, save_path, t):
    """Grouped bar chart: one subplot per metric, bars per method."""
    metrics = [('faithfulness', 'Faithfulness ↑\n(higher = better)'),
               ('monotonicity', 'Monotonicity ↑\n(higher = better)'),
               ('completeness', 'Completeness err ↓\n(lower = better)'),
               ('stability',    'Stability ↓\n(lower = better)')]
    cols = [t['c_gemex'], t['c_shap'], t['c_lime'], t['c_eli5']]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    _apply_theme(fig, axes, t)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.86,
                        bottom=0.14, wspace=0.30)

    for mi, (key, label) in enumerate(metrics):
        ax   = axes[mi]
        vals = [np.mean(records[m][key]) for m in METHODS]
        errs = [np.std(records[m][key])  for m in METHODS]
        x    = np.arange(len(METHODS))

        bars = ax.bar(x, vals, color=cols, alpha=0.82,
                      edgecolor='none', width=0.6)
        ax.errorbar(x, vals, yerr=errs, fmt='none',
                    ecolor=t['text3'], elinewidth=1.5, capsize=4)
        for xi, v in enumerate(vals):
            ax.text(xi, v + (max(map(abs, vals)) or 1)*0.04,
                    f'{v:.3f}', ha='center', fontsize=9,
                    color=t['text'], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=20, ha='right',
                           fontsize=9, color=t['text'])
        ax.set_title(label, fontsize=10, color=t['text'],
                     fontweight='bold', pad=6)

    fig.suptitle(f'{ds_name}  ·  GEMEX vs SHAP vs LIME vs ELI5',
                 fontsize=12, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_radar(records, ds_name, save_path, t):
    """Radar/spider chart — 5 axes including interaction richness."""
    cats   = ['Faithfulness', 'Monotonicity', 'Stability', 'Completeness', 'Speed']
    n_cats = len(cats)
    angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    method_colors = [t['c_gemex'], t['c_shap'], t['c_lime'], t['c_eli5']]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['bg'])
    ax.spines['polar'].set_color(t['border'])
    ax.tick_params(colors=t['text2'], labelsize=8)

    for mi, (method, col) in enumerate(zip(METHODS, method_colors)):
        m    = records[method]
        # Normalise each metric to [0,1] (higher always better after transform)
        vals = [
            np.clip(np.mean(m['faithfulness']) + 1, 0, 1),  # shift [-1,0]→[0,1]
            np.clip(np.mean(m['monotonicity']), 0, 1),
            np.clip(1/(1 + np.mean(m['stability'])), 0, 1),
            np.clip(1/(1 + np.mean(m['completeness'])), 0, 1),
            np.clip(1/(1 + np.mean(m['time'])), 0, 1),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=col, lw=2.0, label=method)
        ax.fill(angles, vals, color=col, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=t['text'], fontsize=10)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.0'],
                       color=t['text2'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=10, framealpha=0.4,
              facecolor=t['panel'], edgecolor=t['border'],
              labelcolor=t['text'])
    ax.set_title(f'{ds_name}  ·  Method Comparison',
                 fontsize=11, color=t['text'],
                 fontweight='bold', pad=18)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_attribution_comparison(atfs_dict, feat, instance_idx,
                                ds_name, save_path, t):
    """Side-by-side attribution bar chart — all methods for one instance."""
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5*len(METHODS), 6))
    _apply_theme(fig, axes, t)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.88,
                        bottom=0.05, wspace=0.35)

    method_colors = {'GEMEX': t['c_gemex'], 'SHAP': t['c_shap'],
                     'LIME':  t['c_lime'],  'ELI5': t['c_eli5']}

    for mi, method in enumerate(METHODS):
        ax   = axes[mi]
        atf  = atfs_dict[method][0] if len(atfs_dict[method]) > 0 else np.zeros(len(feat))
        order = np.argsort(np.abs(atf))[::-1]
        vals  = atf[order]
        names = [feat[i] for i in order]
        y_pos = np.arange(len(feat))

        cols = [method_colors[method] if v >= 0 else '#E24B4A' for v in vals]
        ax.barh(y_pos, vals, color=cols, alpha=0.85, edgecolor='none')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8.5, color=t['text'])
        ax.axvline(0, color=t['border'], lw=0.8)
        ax.set_title(method, fontsize=11, color=method_colors[method],
                     fontweight='bold', pad=6)
        ax.tick_params(axis='x', colors=t['text2'])

        for yi, v in enumerate(vals):
            ax.text(v + (0.005 if v >= 0 else -0.005), yi,
                    f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right',
                    fontsize=7.5, color=t['text'])

    fig.suptitle(f'{ds_name}  ·  Attribution Comparison  '
                 f'(instance #{instance_idx})',
                 fontsize=12, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_separate_metric_heatmap(all_results, save_path, t, theme='dark'):
    """8-panel heatmap — one per metric, method × dataset."""
    ds_names = list(all_results.keys())
    metrics  = [
        ('faithfulness',  'Faithfulness ↑',    lambda v: np.clip(v+1, 0, 1)),
        ('monotonicity',  'Monotonicity ↑',    lambda v: np.clip(v, 0, 1)),
        ('completeness',  'Completeness err ↓', lambda v: np.clip(1/(1+v), 0, 1)),
        ('stability',     'Stability ↓',        lambda v: np.clip(1/(1+v), 0, 1)),
        ('time',          'Speed ↑',            lambda v: np.clip(1/(1+v), 0, 1)),
        ('ricci',         'Ricci scalar ↑',     lambda v: np.clip(v/(v+1), 0, 1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _apply_theme(fig, axes.flatten(), t)
    plt.subplots_adjust(hspace=0.45, wspace=0.28,
                        left=0.08, right=0.97, top=0.91, bottom=0.06)

    for mi, (key, label, norm_fn) in enumerate(metrics):
        ax  = axes[mi//3][mi%3]
        raw = np.zeros((len(METHODS), len(ds_names)))
        nrm = np.zeros_like(raw)
        for ri, method in enumerate(METHODS):
            for ci, ds in enumerate(ds_names):
                v = np.mean(all_results[ds][method].get(key, [0]))
                raw[ri, ci] = v
                nrm[ri, ci] = float(norm_fn(v))

        cmap = 'YlGn' if theme == 'light' else 'viridis'
        im   = ax.imshow(nrm, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(ds_names)))
        ax.set_xticklabels([d[:13] for d in ds_names],
                           rotation=18, ha='right',
                           fontsize=8, color=t['text'])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=9, color=t['text'])
        for ri in range(len(METHODS)):
            for ci in range(len(ds_names)):
                ax.text(ci, ri, f'{raw[ri,ci]:.3f}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if nrm[ri,ci] < 0.55 else 'black')
        ax.set_title(label, fontsize=9.5, color=t['text'],
                     fontweight='bold', pad=6)

    fig.suptitle('GEMEX Comparative Study  ·  Separate Metric Tables',
                 fontsize=13, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# RUN ONE DATASET
# ═══════════════════════════════════════════════════════════════════════

def run_dataset(X, y, feat, ds_name, n_instances, n_seeds,
                model_type, save_dir, theme):
    t = DARK if theme == 'dark' else LIGHT
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.20, random_state=42, stratify=y)

    if model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh',
                              max_iter=300, random_state=42,
                              early_stopping=True)
    elif model_type == 'deep_mlp':
        # Deeper neural network — 3 hidden layers
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                              activation='relu', max_iter=400,
                              learning_rate='adaptive',
                              random_state=42, early_stopping=True)
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)

    cv_auc = cross_val_score(
        model, X_s, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc').mean()
    te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"\n  [{ds_name}] CV-AUC={cv_auc:.4f}  Test-AUC={te_auc:.4f}")

    base     = X_tr.mean(axis=0)
    records  = {m: {'faithfulness':[], 'monotonicity':[],
                    'completeness':[], 'stability':[], 'time':[],
                    'ricci':[]} for m in METHODS}
    atfs_for_plot = {}

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds} ...", end=' ', flush=True)

        g_atf, g_idx, g_ricci = get_gemex(
            model, X_tr, X_te, feat, n_instances, seed)
        s_atf, s_idx = get_shap(model, X_tr, X_te, n_instances, seed, model_type)
        l_atf, l_idx = get_lime(model, X_tr, X_te, feat, n_instances, seed)
        e_atf, e_idx = get_eli5(model, X_tr, X_te, y_te, n_instances, seed)

        for method, atf, idx in [('GEMEX', g_atf, g_idx),
                                   ('SHAP',  s_atf, s_idx),
                                   ('LIME',  l_atf, l_idx),
                                   ('ELI5',  e_atf, e_idx)]:
            t0 = time.perf_counter()
            records[method]['faithfulness'].append(
                metric_faithfulness(model, idx, atf))
            records[method]['monotonicity'].append(
                metric_monotonicity(model, idx, atf))
            records[method]['completeness'].append(
                metric_completeness(model, idx, atf, base))
            records[method]['stability'].append(
                metric_stability(idx, atf))
            records[method]['time'].append(time.perf_counter() - t0)
            if method == 'GEMEX':
                records['GEMEX']['ricci'].append(g_ricci)
            else:
                records[method]['ricci'].append(0.0)

        if seed == 0:
            atfs_for_plot = {'GEMEX': g_atf, 'SHAP': s_atf,
                             'LIME':  l_atf, 'ELI5': e_atf}
        print("done")

    # ── Console tables ────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  {ds_name}  [{model_type.upper()}]  "
          f"n={n_instances} × {n_seeds} seeds")
    print(f"  {'='*65}")
    metrics_list = ['faithfulness', 'monotonicity', 'completeness', 'stability']
    metric_notes = ['↑ higher', '↑ higher', '↓ lower', '↓ lower']

    for metric, note in zip(metrics_list, metric_notes):
        print(f"\n  {metric.capitalize()} [{note} is better]")
        print(f"  {'Method':<8}  {'Mean':>8}  {'Std':>7}  "
              f"{'CI 95% Lo':>10}  {'CI 95% Hi':>10}")
        print(f"  {'-'*50}")
        for method in METHODS:
            vals        = np.array(records[method][metric])
            mn, lo, hi  = bootstrap_ci(vals)
            print(f"  {method:<8}  {mn:>+8.4f}  "
                  f"{np.std(vals):>7.4f}  {lo:>10.4f}  {hi:>10.4f}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests: GEMEX vs baselines")
    print(f"  {'Baseline':<8}  {'Metric':<14}  "
          f"{'p':>8}  {'d (Cohen)':>10}  {'sig':>5}")
    print(f"  {'-'*50}")
    for method in ['SHAP', 'LIME', 'ELI5']:
        for metric in ['faithfulness', 'monotonicity']:
            a = np.array(records['GEMEX'][metric])
            b = np.array(records[method][metric])
            n = min(len(a), len(b))
            try:
                _, p = wilcoxon(a[:n], b[:n])
                d    = cohens_d(a[:n], b[:n])
                sig  = '**' if p < 0.05 else ('*' if p < 0.10 else 'ns')
                print(f"  {method:<8}  {metric:<14}  "
                      f"{p:>8.4f}  {d:>10.4f}  {sig:>5}")
            except Exception:
                pass

    # ── Plots ─────────────────────────────────────────────────────────
    tag = ds_name.lower().replace(' ', '_')
    plot_metric_bars(records, ds_name,
                     f"{save_dir}/{tag}_metric_bars_{theme}.png", t)
    plot_radar(records, ds_name,
               f"{save_dir}/{tag}_radar_{theme}.png", t)
    plot_attribution_comparison(
        atfs_for_plot, feat, 0, ds_name,
        f"{save_dir}/{tag}_attributions_{theme}.png", t)

    # ── CSV ───────────────────────────────────────────────────────────
    rows = []
    for method in METHODS:
        for si in range(n_seeds):
            row = {'dataset': ds_name, 'model': model_type,
                   'method': method, 'seed': si}
            for metric in metrics_list + ['ricci']:
                vals = records[method][metric]
                row[metric] = vals[si] if si < len(vals) else 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = f"{save_dir}/{tag}_{model_type}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    return records, atfs_for_plot



# ═══════════════════════════════════════════════════════════════════════
# COMBINED GBM vs MLP COMPARISON (produced when --model both)
# ═══════════════════════════════════════════════════════════════════════

def _plot_model_comparison(results_by_model, datasets, save_dir, theme, t):
    """
    Side-by-side bar chart: GBM vs MLP for GEMEX across all datasets.
    Shows where GEMEX gains most from smooth model geometry.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ds_names  = [d[3] for d in datasets]
    metrics   = [('monotonicity', 'Monotonicity ↑'),
                 ('stability',    'Stability ↓'),
                 ('faithfulness', 'Faithfulness ↑')]
    higher    = [True, False, True]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5.5))
    fig.patch.set_facecolor(t['bg'])
    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.18, wspace=0.32)

    model_colors = {'gbm': '#00C896', 'mlp': '#C97EFA',
                    'deep_mlp': '#F5C842'}
    x = np.arange(len(ds_names))
    w = 0.35

    for mi, ((metric, label), hi) in enumerate(zip(metrics, higher)):
        ax = axes[mi]
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.6)
        ax.tick_params(colors=t['text2'], labelsize=9)
        ax.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)

        for mi2, (model_type, col) in enumerate(model_colors.items()):
            vals = [np.mean(results_by_model[model_type][ds]['GEMEX'][metric])
                    for ds in ds_names]
            bars = ax.bar(x + mi2*w - w/2, vals, w*0.85,
                          color=col, alpha=0.85, edgecolor='none',
                          label=f'GEMEX ({model_type.upper()})')
            for xi, v in enumerate(vals):
                ax.text(xi + mi2*w - w/2, v + abs(v)*0.05,
                        f'{v:.3f}', ha='center', fontsize=8,
                        color=t['text'])

        ax.set_xticks(x)
        ax.set_xticklabels([d[:13] for d in ds_names],
                           rotation=18, ha='right',
                           fontsize=9, color=t['text'])
        ax.set_title(label, fontsize=10, color=t['text'], fontweight='bold')
        if mi == 0:
            ax.legend(fontsize=9, framealpha=0.4,
                      facecolor=t['panel'], edgecolor=t['border'],
                      labelcolor=t['text'])

    fig.suptitle('GEMEX — GBM vs MLP Model Comparison\n'
                 'Shows how model smoothness affects GEMEX geometric quality',
                 fontsize=12, color=t['text'], fontweight='bold')
    path = f'{save_dir}/gemex_gbm_vs_mlp_{theme}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f'  -> {path}')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX Comparative Study')
    parser.add_argument('--heart-path',  default=None)
    parser.add_argument('--pima-path',   default=None)
    parser.add_argument('--save-dir',    default='./gemex_comparative')
    parser.add_argument('--n-instances', type=int, default=30)
    parser.add_argument('--n-seeds',     type=int, default=5)
    parser.add_argument('--model',       default='all',
                        choices=['gbm', 'mlp', 'deep_mlp', 'both', 'all'],
                        help='gbm=GradientBoosting, mlp=MLP(64,32), '
                             'deep_mlp=MLP(128,64,32), '
                             'both=gbm+mlp, all=all four models')
    parser.add_argument('--theme',       default='dark',
                        choices=['dark', 'light'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    t = DARK if args.theme == 'dark' else LIGHT

    print("\n" + "="*65)
    print("  GEMEX Comparative Study — v1.2.0")
    print(f"  Methods     : {', '.join(METHODS)}")
    print(f"  n_instances : {args.n_instances} per seed")
    print(f"  n_seeds     : {args.n_seeds}")
    print(f"  model       : {args.model.upper()}")
    print("="*65)

    # Load datasets once
    datasets = []
    Xh, yh, fh, nh = load_heart(args.heart_path)
    if Xh is not None: datasets.append((Xh, yh, fh, nh))
    Xp, yp, fp, np_ = load_pima(args.pima_path)
    if Xp is not None: datasets.append((Xp, yp, fp, np_))
    Xb, yb, fb, nb = load_breast()
    datasets.append((Xb, yb, fb, nb))

    # Determine which model types to run
    if args.model == 'both':
        model_types = ['gbm', 'mlp']
    elif args.model == 'all':
        model_types = ['gbm', 'mlp', 'deep_mlp']
    else:
        model_types = [args.model]

    all_results_by_model = {}  # {model_type: {ds_name: records}}

    for model_type in model_types:
        print(f"\n{'='*65}")
        print(f"  Running model: {model_type.upper()}")
        print(f"{'='*65}")
        model_dir    = os.path.join(args.save_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        all_results  = {}
        all_atfs     = {}

        for X, y, feat, ds_name in datasets:
            records, atfs = run_dataset(
                X, y, feat, ds_name,
                args.n_instances, args.n_seeds,
                model_type, model_dir, args.theme)
            all_results[ds_name] = records
            all_atfs[ds_name]    = atfs

        # Per-model summary heatmap
        plot_separate_metric_heatmap(
            all_results,
            f"{model_dir}/summary_heatmap_{args.theme}.png",
            t, theme=args.theme)
        print(f"  -> {model_dir}/summary_heatmap_{args.theme}.png")
        all_results_by_model[model_type] = all_results

    # If both models run, produce a combined GBM vs MLP comparison
    if len(model_types) >= 2:
        _plot_model_comparison(
            all_results_by_model, datasets,
            args.save_dir, args.theme, t)

    print(f"\n  All outputs: {os.path.abspath(args.save_dir)}/")


if __name__ == '__main__':
    main()