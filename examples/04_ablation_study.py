# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
04_ablation_study.py
====================
Full ablation study for GEMEX v1.2.0.

Tests the contribution of each algorithmic component by disabling
them one at a time and measuring the impact on standard XAI metrics.

Ablation variants
-----------------
  GEMEX-Full      : all components active (baseline)
  No-RK4          : straight-line path instead of RK4 geodesic
  No-KernelFIM    : single-point FIM, no neighbourhood averaging
  No-AdaptiveEps  : fixed epsilon=1e-3, no adaptive search
  No-Balance      : raw cosine GSF without FIM-diagonal balancing
  No-ConfWeight   : uniform path weights, no confidence adaptation
  No-StableSign   : midpoint sign instead of full-path integral
  No-Interactions : PTI disabled (interaction_order=1)

Metrics evaluated (n_seeds seeds, n_instances per dataset)
-----------------------------------------------------------
  Faithfulness   (Alvarez-Melis & Jaakkola, 2018)
  Monotonicity   (Luss et al. (2019))
  Completeness   (Sundararajan et al., 2017)
  Stability      (Lipschitz)

Datasets: Cleveland Heart Disease + Pima Indians Diabetes

Usage
-----
  python 04_ablation_study.py
  python 04_ablation_study.py --n-instances 30 --n-seeds 5 --theme light
  python 04_ablation_study.py --dataset heart   # single dataset
"""

import argparse
import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from gemex import Explainer, GemexConfig

# ═══════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════

DARK = dict(
    bg='#0D0D1A', panel='#131326', grid='#1E1E38', border='#2E2E55',
    text='#E8E8F0', text2='#9999BB', text3='#444466',
    full='#00C896', ablate='#F5A442',
)
LIGHT = dict(
    bg='#F4F4F9', panel='#FFFFFF', grid='#EBEBF5', border='#CCCCDD',
    text='#1A1A2E', text2='#555577', text3='#AAAACC',
    full='#0A7A5A', ablate='#C05A10',
)

# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_heart(path='cleveland_heart.csv'):
    df = pd.read_csv(path).dropna()
    df['target'] = (df['target'] > 0).astype(int)
    feat = [c for c in df.columns if c != 'target']
    return df[feat].values.astype(float), df['target'].values.astype(int), feat

def load_pima(path='pima_diabetes.csv'):
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
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    corrs = []
    for i in idx:
        x  = X[i].copy()
        p0 = model.predict_proba(x.reshape(1, -1))[0, 1]
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
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
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
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    pb  = model.predict_proba(baseline.reshape(1,-1))[0,1]
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

# ═══════════════════════════════════════════════════════════════════════
# ABLATION CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

def make_configs():
    """
    Return dict of variant_name -> GemexConfig for each ablation variant.
    Each variant disables exactly one component relative to the full config.
    """
    base = dict(
        n_geodesic_steps    = 20,
        n_reference_samples = 40,
        fim_epsilon         = 1e-3,
        fim_epsilon_auto    = True,
        fim_local_avg       = True,
        fim_local_sigma     = 0.10,
        fim_local_n         = 16,
        interaction_order   = 2,
        gsf_normalise       = False,
        verbose             = False,
    )

    variants = {}

    # Full GEMEX — all components active
    variants['GEMEX-Full'] = GemexConfig(**base)

    # No-RK4: straight-line path (Euler integration, steps=1)
    cfg = GemexConfig(**base)
    cfg.n_geodesic_steps = 2   # effectively straight line
    variants['No-RK4'] = cfg

    # No-KernelFIM: single-point FIM, no neighbourhood averaging
    cfg = GemexConfig(**base)
    cfg.fim_local_avg = False
    cfg.n_reference_samples = 1
    variants['No-KernelFIM'] = cfg

    # No-AdaptiveEps: fixed base epsilon, no auto-search
    cfg = GemexConfig(**base)
    cfg.fim_epsilon_auto = False
    variants['No-AdaptiveEps'] = cfg

    # No-Interactions: PTI disabled
    cfg = GemexConfig(**base)
    cfg.interaction_order = 1
    variants['No-Interactions'] = cfg

    # No-NormGSF: no completeness normalisation (already default,
    # so this variant enables gsf_normalise to show its effect)
    cfg = GemexConfig(**base)
    cfg.gsf_normalise = True
    variants['With-NormGSF'] = cfg

    return variants

# ═══════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_variant(variant_name, cfg, model, X_tr, X_te, feat, n_inst, seed):
    """Run one ablation variant and return metric dict."""
    n   = min(n_inst, len(X_te))
    idx = np.random.RandomState(seed).choice(len(X_te), n, replace=False)
    exp = Explainer(model, data_type='tabular',
                    feature_names=feat,
                    class_names=['No', 'Yes'],
                    config=cfg)
    atfs = []
    t0   = time.perf_counter()
    for i in idx:
        r = exp.explain(X_te[i], X_reference=X_tr)
        atfs.append(r.gsf_scores)
    elapsed = (time.perf_counter() - t0) / len(idx)
    atfs    = np.nan_to_num(np.array(atfs), nan=0.0)

    base = X_tr.mean(axis=0)
    return {
        'variant'      : variant_name,
        'faithfulness' : faithfulness(model, X_te[idx], atfs),
        'monotonicity' : monotonicity(model, X_te[idx], atfs),
        'completeness' : completeness(model, X_te[idx], atfs, base),
        'stability'    : stability(X_te[idx], atfs),
        'time_s'       : elapsed,
    }

# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_ablation_bars(df_mean, df_std, ds_name, save_path, t):
    """
    Grouped bar chart: one group per metric, one bar per variant.
    Full GEMEX bar is highlighted; ablated variants are muted.
    """
    metrics = ['faithfulness', 'monotonicity', 'completeness', 'stability']
    labels  = ['Faithfulness ↑', 'Monotonicity ↑',
               'Completeness err ↓', 'Stability ↓']
    variants = df_mean['variant'].tolist()
    n_v      = len(variants)
    n_m      = len(metrics)
    x        = np.arange(n_v)
    width    = 0.18

    fig, axes = plt.subplots(1, n_m, figsize=(18, 5.5))
    fig.patch.set_facecolor(t['bg'])
    for ax in axes:
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.7)
        ax.tick_params(colors=t['text2'], labelsize=8.5)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88,
                        bottom=0.22, wspace=0.32)

    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        ax   = axes[mi]
        vals = df_mean[metric].values
        errs = df_std[metric].values

        colors = [t['full'] if v == 'GEMEX-Full' else t['ablate']
                  for v in variants]
        bars = ax.bar(x, vals, color=colors, alpha=0.82,
                      edgecolor=t['bg'], lw=0.5, width=0.7)
        ax.errorbar(x, vals, yerr=errs, fmt='none',
                    ecolor=t['text3'], elinewidth=1.2,
                    capsize=3, capthick=1.0)

        # Annotate values
        for xi, (v, e) in enumerate(zip(vals, errs)):
            ax.text(xi, v + max(vals)*0.03, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7.5,
                    color=t['text'], fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=35, ha='right',
                           fontsize=8, color=t['text'])
        ax.set_title(label, fontsize=10, color=t['text'],
                     fontweight='bold', pad=7)
        ax.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)
        ax.set_xlim(-0.6, n_v - 0.4)

    # Legend
    handles = [
        plt.Rectangle((0,0),1,1, color=t['full'],   alpha=0.82,
                       label='GEMEX-Full (all components)'),
        plt.Rectangle((0,0),1,1, color=t['ablate'], alpha=0.82,
                       label='Ablated variant'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               fontsize=9, framealpha=0.4,
               facecolor=t['panel'], edgecolor=t['border'],
               labelcolor=t['text'], bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{ds_name}  ·  GEMEX Ablation Study\n'
                 f'(mean ± std across seeds, error bars shown)',
                 fontsize=12, color=t['text'], fontweight='bold', y=1.04)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_delta_heatmap(df_mean, ds_name, save_path, t):
    """
    Heatmap of percentage change relative to GEMEX-Full.
    Red = worse than full, green = better or equal.
    """
    metrics  = ['faithfulness', 'monotonicity', 'completeness', 'stability']
    variants = [v for v in df_mean['variant'] if v != 'GEMEX-Full']
    full_row = df_mean[df_mean['variant'] == 'GEMEX-Full'].iloc[0]

    data = np.zeros((len(variants), len(metrics)))
    for ri, var in enumerate(variants):
        row = df_mean[df_mean['variant'] == var].iloc[0]
        for ci, metric in enumerate(metrics):
            ref = full_row[metric]
            val = row[metric]
            if abs(ref) > 1e-8:
                pct = (val - ref) / abs(ref) * 100
            else:
                pct = 0.0
            # For lower-is-better metrics flip sign so red always = worse
            if metric in ('completeness', 'stability'):
                pct = -pct
            data[ri, ci] = pct

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(variants)*0.8)))
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['panel'])
    for sp in ax.spines.values():
        sp.set_color(t['border']); sp.set_linewidth(0.7)

    plt.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.16)
    lim = max(abs(data).max(), 5.0)
    im  = ax.imshow(data, cmap='RdYlGn', vmin=-lim, vmax=lim, aspect='auto')

    col_labels = ['Faithfulness ↑', 'Monotonicity ↑',
                  'Compl. err ↓', 'Stability ↓']
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, fontsize=9, color=t['text'], rotation=18, ha='right')
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=9.5, color=t['text'])

    for ri in range(len(variants)):
        for ci in range(len(metrics)):
            ax.text(ci, ri, f'{data[ri,ci]:+.1f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if abs(data[ri,ci]) > lim*0.5 else t['text'])

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('% change vs GEMEX-Full\n(green = better, red = worse)',
                 color=t['text2'], fontsize=8.5)
    cb.ax.tick_params(colors=t['text2'])
    cb.outline.set_edgecolor(t['border'])

    ax.set_title(f'{ds_name}  ·  Ablation Δ vs GEMEX-Full',
                 fontsize=11, color=t['text'], fontweight='bold', pad=9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run(heart_path, pima_path, save_dir, theme, n_instances,
        n_seeds, model_type, dataset_filter):

    os.makedirs(save_dir, exist_ok=True)
    t        = DARK if theme == 'dark' else LIGHT
    variants = make_configs()

    datasets = []
    if dataset_filter in ('both', 'heart'):
        datasets.append(('heart', heart_path, load_heart,
                         'Cleveland Heart Disease'))
    if dataset_filter in ('both', 'pima'):
        datasets.append(('pima', pima_path, load_pima,
                         'Pima Indians Diabetes'))

    all_tables = {}   # ds_name -> DataFrame of results

    for tag, path, loader, ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        X, y, feat = loader(path)
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        # Train model (same type across seeds for fairness)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y, test_size=0.20, random_state=42, stratify=y)

        if model_type == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(64, 32),
                                  activation='tanh', max_iter=500,
                                  random_state=42, early_stopping=True)
            model_label = 'MLP'
        else:
            model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               random_state=42)
            model_label = 'GBM'

        model.fit(X_tr, y_tr)
        print(f"  Model: {model_label}")

        records = []
        for seed in range(n_seeds):
            print(f"\n  Seed {seed+1}/{n_seeds}")
            for var_name, cfg in variants.items():
                print(f"    {var_name:<20} ...", end='', flush=True)
                try:
                    rec = run_variant(var_name, cfg, model,
                                      X_tr, X_te, feat, n_instances, seed)
                    records.append({'seed': seed, **rec})
                    print(f" faith={rec['faithfulness']:+.3f}  "
                          f"mono={rec['monotonicity']:.3f}")
                except Exception as e:
                    print(f" ERROR: {e}")

        df = pd.DataFrame(records)
        df_mean = df.groupby('variant').mean(numeric_only=True).reset_index()
        df_std  = df.groupby('variant').std(numeric_only=True).reset_index()

        # Preserve order
        order   = list(variants.keys())
        df_mean['variant'] = pd.Categorical(df_mean['variant'], order)
        df_std ['variant'] = pd.Categorical(df_std ['variant'], order)
        df_mean = df_mean.sort_values('variant').reset_index(drop=True)
        df_std  = df_std .sort_values('variant').reset_index(drop=True)

        all_tables[ds_name] = (df_mean, df_std)

        # ── Console table ─────────────────────────────────────────────
        print(f"\n  Results — {ds_name} [{model_label}]  "
              f"(n={n_instances} × {n_seeds} seeds)")
        hdr = f"  {'Variant':<22}  {'Faith':>8}  {'Mono':>7}  "
        hdr += f"{'Compl':>8}  {'Stab':>8}  {'Time(s)':>8}"
        print(hdr)
        print('  ' + '-'*72)
        for _, row in df_mean.iterrows():
            std_row = df_std[df_std['variant'] == row['variant']].iloc[0]
            print(f"  {row['variant']:<22}  "
                  f"{row['faithfulness']:+8.4f}  "
                  f"{row['monotonicity']:7.4f}  "
                  f"{row['completeness']:8.4f}  "
                  f"{row['stability']:8.4f}  "
                  f"{row['time_s']:8.4f}")

        # ── Statistical test: GEMEX-Full vs each variant ──────────────
        print("\n  Wilcoxon signed-rank test vs GEMEX-Full:")
        full_vals = df[df['variant'] == 'GEMEX-Full']
        for var_name in variants:
            if var_name == 'GEMEX-Full': continue
            var_vals = df[df['variant'] == var_name]
            if len(full_vals) < 4 or len(var_vals) < 4: continue
            for metric in ('faithfulness', 'monotonicity'):
                a = full_vals[metric].values
                b = var_vals[metric].values
                n = min(len(a), len(b))
                try:
                    _, p = wilcoxon(a[:n], b[:n])
                    sig = '**' if p < 0.05 else ('*' if p < 0.10 else 'ns')
                    print(f"    {var_name:<20} vs Full  [{metric}]  p={p:.3f} {sig}")
                except Exception:
                    pass

        # ── Plots ─────────────────────────────────────────────────────
        b = f"{save_dir}/{tag}"
        plot_ablation_bars(df_mean, df_std, f"{ds_name} [{model_label}]",
                           f"{b}_ablation_bars_{theme}.png", t)
        plot_delta_heatmap(df_mean, f"{ds_name} [{model_label}]",
                           f"{b}_ablation_delta_{theme}.png", t)

        # Save CSV
        df.to_csv(f"{b}_ablation_raw_{theme}.csv", index=False)
        df_mean.to_csv(f"{b}_ablation_mean_{theme}.csv", index=False)
        print(f"  -> {b}_ablation_raw_{theme}.csv")
        print(f"  -> {b}_ablation_mean_{theme}.csv")

    print(f"\n{'='*60}")
    print(f"  Done. Outputs in: {os.path.abspath(save_dir)}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GEMEX Ablation Study')
    parser.add_argument('--heart',       default='cleveland_heart.csv')
    parser.add_argument('--pima',        default='pima_diabetes.csv')
    parser.add_argument('--save-dir',    default='./gemex_ablation')
    parser.add_argument('--theme',       default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--n-instances', type=int, default=30,
                        help='Instances to explain per seed (default 30)')
    parser.add_argument('--n-seeds',     type=int, default=5,
                        help='Random seeds for stability (default 5)')
    parser.add_argument('--model',       default='gbm',
                        choices=['gbm', 'mlp'],
                        help='Model type to explain (default gbm)')
    parser.add_argument('--dataset',     default='both',
                        choices=['both', 'heart', 'pima'])
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GEMEX Ablation Study — v1.2.0")
    print("="*60)
    print(f"  n_instances : {args.n_instances} per seed")
    print(f"  n_seeds     : {args.n_seeds}")
    print(f"  model       : {args.model.upper()}")
    print(f"  dataset     : {args.dataset}")
    print(f"  theme       : {args.theme}\n")

    run(heart_path      = args.heart,
        pima_path       = args.pima,
        save_dir        = args.save_dir,
        theme           = args.theme,
        n_instances     = args.n_instances,
        n_seeds         = args.n_seeds,
        model_type      = args.model,
        dataset_filter  = args.dataset)
