# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
05_statistical_comparison.py
=============================
Statistically rigorous comparison of GEMEX vs SHAP vs LIME vs ELI5.

Runs the comparison across multiple random seeds and reports:
  - Mean ± standard deviation per metric per method
  - Wilcoxon signed-rank test (GEMEX vs each baseline)
  - Effect size (Cohen's d)
  - Bootstrap 95% confidence intervals

This script produces publication-ready statistical tables
suitable for inclusion in a conference or journal paper.

Usage
-----
  python 05_statistical_comparison.py
  python 05_statistical_comparison.py --n-seeds 10 --n-instances 50
  python 05_statistical_comparison.py --model mlp --dataset pima
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from eli5.sklearn import PermutationImportance
    from gemex import Explainer, GemexConfig
except ImportError as e:
    print(f"Missing package: {e}")
    print("pip install gemex shap lime eli5")
    raise

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_heart(path):
    df = pd.read_csv(path).dropna()
    df['target'] = (df['target'] > 0).astype(int)
    feat = [c for c in df.columns if c != 'target']
    return df[feat].values.astype(float), df['target'].values.astype(int), feat

def load_pima(path):
    df = pd.read_csv(path)
    if df.iloc[0, 0] == 'Pregnancies':
        df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    feat = [c for c in df.columns if c != 'Outcome']
    return df[feat].values.astype(float), df['Outcome'].values.astype(int), feat

# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def faithfulness(model, X, atfs, n=30):
    idx   = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    corrs = []
    for i in idx:
        x  = X[i].copy()
        p0 = model.predict_proba(x.reshape(1,-1))[0,1]
        order = np.argsort(np.abs(atfs[i]))[::-1]
        drops, scores = [], []
        xm = x.copy()
        for fi in order[:min(8, len(order))]:
            xm[fi] = 0.0
            drops.append(abs(p0 - model.predict_proba(xm.reshape(1,-1))[0,1]))
            scores.append(abs(float(atfs[i][fi])))
        if len(set(scores)) > 1:
            c, _ = spearmanr(scores, drops)
            if not np.isnan(c): corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0

def monotonicity(model, X, atfs, n=30):
    idx    = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    checks = []
    for i in idx:
        x  = X[i].copy()
        p0 = model.predict_proba(x.reshape(1,-1))[0,1]
        for fi in range(X.shape[1]):
            eps = 0.2 * (X[:,fi].std() + 1e-8)
            xp  = x.copy(); xp[fi] += eps
            pp  = model.predict_proba(xp.reshape(1,-1))[0,1]
            if np.sign(atfs[i][fi]) != 0:
                checks.append(int(np.sign(atfs[i][fi]) == np.sign(pp - p0)))
    return float(np.mean(checks)) if checks else 0.0

def completeness(model, X, atfs, baseline, n=30):
    idx  = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    pb   = model.predict_proba(baseline.reshape(1,-1))[0,1]
    errs = []
    for i in idx:
        px = model.predict_proba(X[i].reshape(1,-1))[0,1]
        errs.append(abs(atfs[i].sum() - (px - pb)))
    return float(np.mean(errs))

def stability(X, atfs, n_pairs=40):
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

def bootstrap_ci(vals, n_boot=1000, ci=0.95):
    """Return (mean, lower, upper) bootstrap confidence interval."""
    bs = [np.mean(np.random.choice(vals, len(vals), replace=True))
          for _ in range(n_boot)]
    lo = np.percentile(bs, (1-ci)/2*100)
    hi = np.percentile(bs, (1-(1-ci)/2)*100)
    return float(np.mean(vals)), float(lo), float(hi)

def cohens_d(a, b):
    """Effect size: Cohen's d = (mean_a - mean_b) / pooled_std."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.std(a)**2 + (nb-1)*np.std(b)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-10)

# ═══════════════════════════════════════════════════════════════════════
# EXPLAINERS
# ═══════════════════════════════════════════════════════════════════════

def get_gemex_atfs(model, X_tr, X_te, feat, n, seed, is_tree):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    cfg = GemexConfig(n_geodesic_steps=15, n_reference_samples=30,
                      interaction_order=1, verbose=False)
    exp = Explainer(model, data_type='tabular', feature_names=feat,
                    class_names=['No','Yes'], config=cfg)
    atfs = []
    for i in idx:
        r = exp.explain(X_te[i], X_reference=X_tr)
        atfs.append(r.gsf_scores)
    return np.nan_to_num(np.array(atfs)), X_te[idx]

def get_shap_atfs(model, X_tr, X_te, n, seed, is_tree):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    if is_tree:
        exp  = shap.TreeExplainer(model, data=X_tr[:50])
        vals = exp.shap_values(X_te[idx], check_additivity=False)
    else:
        bg   = X_tr[np.random.RandomState(seed).choice(len(X_tr), 30, replace=False)]
        exp  = shap.KernelExplainer(model.predict_proba, bg)
        vals = exp.shap_values(X_te[idx], nsamples=80, silent=True)
    atfs = vals[1] if isinstance(vals, list) else vals
    return np.array(atfs, dtype=float), X_te[idx]

def get_lime_atfs(model, X_tr, X_te, feat, n, seed):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    exp = LimeTabularExplainer(X_tr, feature_names=feat,
                               class_names=['No','Yes'],
                               discretize_continuous=True, random_state=seed)
    atfs = []
    for i in idx:
        expl = exp.explain_instance(X_te[i], model.predict_proba,
                                    num_features=len(feat), num_samples=300)
        atf = np.zeros(len(feat))
        for fi_str, val in expl.as_list():
            for j, fname in enumerate(feat):
                if fname in str(fi_str):
                    atf[j] = val; break
        atfs.append(atf)
    return np.array(atfs), X_te[idx]

def get_eli5_atfs(model, X_tr, X_te, feat, y_te, n, seed):
    idx = np.random.RandomState(seed).choice(len(X_te), min(n, len(X_te)), replace=False)
    pi  = PermutationImportance(model, random_state=seed, n_iter=5)
    pi.fit(X_te, y_te)
    base = X_tr.mean(axis=0)
    atfs = []
    for i in idx:
        signs = np.sign(X_te[i] - base); signs[signs==0] = 1
        atfs.append(pi.feature_importances_ * signs)
    return np.array(atfs), X_te[idx]

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run(heart_path, pima_path, save_dir, n_instances, n_seeds,
        model_type, dataset_filter, theme):

    os.makedirs(save_dir, exist_ok=True)

    datasets = []
    if dataset_filter in ('both', 'heart'):
        datasets.append(('heart', heart_path, load_heart, 'Cleveland Heart Disease'))
    if dataset_filter in ('both', 'pima'):
        datasets.append(('pima',  pima_path,  load_pima,  'Pima Indians Diabetes'))

    METHODS = ['GEMEX', 'SHAP', 'LIME', 'ELI5']
    METRICS = ['faithfulness', 'monotonicity', 'completeness', 'stability']

    for tag, path, loader, ds_name in datasets:
        print(f"\n{'='*62}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*62}")

        X, y, feat = loader(path)
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.20, random_state=42, stratify=y)

        is_tree = (model_type == 'gbm')
        if is_tree:
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, random_state=42)
        else:
            model = MLPClassifier(hidden_layer_sizes=(64,32),
                                  activation='tanh', max_iter=500,
                                  random_state=42, early_stopping=True)
        model.fit(X_tr, y_tr)
        base = X_tr.mean(axis=0)

        # Collect metric values per method per seed
        records = {m: {mt: [] for mt in METRICS} for m in METHODS}

        for seed in range(n_seeds):
            print(f"\n  Seed {seed+1}/{n_seeds}:")

            g_atf, g_idx = get_gemex_atfs(model, X_tr, X_te, feat,
                                           n_instances, seed, is_tree)
            s_atf, s_idx = get_shap_atfs(model, X_tr, X_te,
                                          n_instances, seed, is_tree)
            l_atf, l_idx = get_lime_atfs(model, X_tr, X_te, feat,
                                          n_instances, seed)
            e_atf, e_idx = get_eli5_atfs(model, X_tr, X_te, feat,
                                          y_te, n_instances, seed)

            for method, atf, idx in [
                ('GEMEX', g_atf, g_idx), ('SHAP', s_atf, s_idx),
                ('LIME',  l_atf, l_idx), ('ELI5', e_atf, e_idx),
            ]:
                records[method]['faithfulness'].append(
                    faithfulness(model, idx, atf))
                records[method]['monotonicity'].append(
                    monotonicity(model, idx, atf))
                records[method]['completeness'].append(
                    completeness(model, idx, atf, base))
                records[method]['stability'].append(
                    stability(idx, atf))
                print(f"    {method:<8}  "
                      f"faith={records[method]['faithfulness'][-1]:+.4f}  "
                      f"mono={records[method]['monotonicity'][-1]:.4f}  "
                      f"compl={records[method]['completeness'][-1]:.4f}  "
                      f"stab={records[method]['stability'][-1]:.4f}")

        # ── Summary table ─────────────────────────────────────────────
        print(f"\n  RESULTS — {ds_name} [{model_type.upper()}]")
        print(f"  n={n_instances} instances × {n_seeds} seeds  "
              f"→  mean ± std  [95% CI]")

        for metric in METRICS:
            print(f"\n  {metric.capitalize()}"
                  f"  ({'↑ higher' if metric in ('faithfulness','monotonicity') else '↓ lower'} is better)")
            print(f"  {'Method':<8}  {'Mean':>8}  {'Std':>7}  "
                  f"{'95% CI Lower':>13}  {'95% CI Upper':>13}")
            print(f"  {'-'*58}")
            for method in METHODS:
                vals = np.array(records[method][metric])
                mn, lo, hi = bootstrap_ci(vals)
                print(f"  {method:<8}  {mn:>+8.4f}  "
                      f"{np.std(vals):>7.4f}  "
                      f"{lo:>13.4f}  {hi:>13.4f}")

        # ── Wilcoxon tests ────────────────────────────────────────────
        print("\n  Wilcoxon signed-rank: GEMEX vs each baseline")
        print(f"  {'Baseline':<8}  {'Metric':<14}  "
              f"{'p-value':>9}  {'d (effect)':>12}  {'sig':>5}")
        print(f"  {'-'*55}")
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
                          f"{p:>9.4f}  {d:>12.4f}  {sig:>5}")
                except Exception:
                    pass

        # ── Save CSV ──────────────────────────────────────────────────
        rows = []
        for method in METHODS:
            for seed_i, seed in enumerate(range(n_seeds)):
                row = {'dataset': ds_name, 'model': model_type,
                       'method': method, 'seed': seed}
                for metric in METRICS:
                    row[metric] = records[method][metric][seed_i]
                rows.append(row)
        csv_path = f"{save_dir}/{tag}_{model_type}_statistical.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n  -> {csv_path}")

    print(f"\n  All outputs: {os.path.abspath(save_dir)}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GEMEX Statistical Comparison Study')
    parser.add_argument('--heart',       default='cleveland_heart.csv')
    parser.add_argument('--pima',        default='pima_diabetes.csv')
    parser.add_argument('--save-dir',    default='./gemex_stats')
    parser.add_argument('--n-instances', type=int, default=40)
    parser.add_argument('--n-seeds',     type=int, default=5)
    parser.add_argument('--model',       default='gbm',
                        choices=['gbm', 'mlp'])
    parser.add_argument('--dataset',     default='both',
                        choices=['both', 'heart', 'pima'])
    parser.add_argument('--theme',       default='dark',
                        choices=['dark', 'light'])
    args = parser.parse_args()

    print("\n" + "="*62)
    print("  GEMEX Statistical Comparison — v1.2.1")
    print("="*62)
    run(heart_path      = args.heart,
        pima_path       = args.pima,
        save_dir        = args.save_dir,
        n_instances     = args.n_instances,
        n_seeds         = args.n_seeds,
        model_type      = args.model,
        dataset_filter  = args.dataset,
        theme           = args.theme)
