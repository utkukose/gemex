# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
03_timeseries_ecg.py
=========================
GEMEX on time series data — two real-world datasets:

  ECG5000     : Electrocardiogram classification (5 classes, 140 time steps)
                Binary version: class 1 (Normal) vs all others (Abnormal)
                Source: UCR Time Series Archive
                Reference: Goldberger et al. (2000) PhysioBank / PhysioNet
                Data format: ECG5000_TRAIN.txt + ECG5000_TEST.txt
                  - Space-separated; first column = class label (1-5)
                  - 140 time steps per instance; 5000 total (500 train + 4500 test)

Demonstrates:
  - GEMEX with data_type='timeseries' on real ECG data
  - Temporal GSF attribution (which time steps drive the prediction)
  - Manifold curvature (Ricci scalar) per heartbeat class
  - Top-10 most important time steps (clean, readable bar chart)
  - Per-class Ricci boxplot and feature attribution heatmap

Requirements
------------
  pip install gemex scikit-learn pandas matplotlib scipy

Usage
-----
  python 03_timeseries_ecg.py
  python 03_timeseries_ecg.py --dataset ecg
  python 03_timeseries_ecg.py --dataset har --theme light
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings('ignore')
from gemex import Explainer, GemexConfig

DARK = dict(
    bg='#0D0D1A', panel='#131326', grid='#1E1E38', border='#2E2E55',
    text='#E8E8F0', text2='#9999BB', text3='#444466',
    c1='#00C896', c2='#F5C842', c3='#C97EFA', c4='#378ADD',
    c5='#E24B4A', c6='#FF8C42',
)
LIGHT = dict(
    bg='#F4F4F9', panel='#FFFFFF', grid='#EBEBF5', border='#CCCCDD',
    text='#1A1A2E', text2='#555577', text3='#AAAACC',
    c1='#0A7A5A', c2='#B8860B', c3='#6A2FA0', c4='#185FA5',
    c5='#A32D2D', c6='#C04000',
)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_ecg5000(path=None):
    """
    ECG5000 — 5-class ECG classification, 140 time steps per instance.
    Binary version: class 1 (Normal) vs all others (Abnormal).

    Data format (UCR text files):
      - Space-separated columns
      - First column: class label (1=Normal, 2-5=Abnormal variants)
      - Columns 2-141: 140 time-step values
      - 500 training + 4500 test instances = 5000 total

    Download from:
      https://www.timeseriesclassification.com/description.php?Dataset=ECG5000

    Expected paths:
      --ecg-path ECG5000/ECG5000_TRAIN.txt  (train file only — test auto-detected)
    """
    import pandas as _pd

    # ── Auto-detect ECG5000 folder if no explicit path given ───────────
    # Look for ECG5000_TRAIN.txt in these locations in order:
    #   1. The explicit --ecg-path argument
    #   2. ECG5000/ subfolder next to this script file
    #   3. ECG5000/ subfolder in the current working directory
    if path is None or not os.path.exists(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, 'ECG5000', 'ECG5000_TRAIN.txt'),
            os.path.join(os.getcwd(),   'ECG5000', 'ECG5000_TRAIN.txt'),
            os.path.join(script_dir, 'ECG5000_TRAIN.txt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                print(f'  Auto-detected ECG5000 at: {path}')
                break

    # ── Try loading real ECG5000 data ──────────────────────────────────
    if path and os.path.exists(path):
        try:
            print(f"  Loading ECG5000 from {path}...")
            # Detect test file alongside train file
            test_path = path.replace('TRAIN', 'TEST')
            df_tr = _pd.read_csv(path, sep=r'\s+', header=None)
            frames = [df_tr]
            if os.path.exists(test_path):
                df_te = _pd.read_csv(test_path, sep=r'\s+', header=None)
                frames.append(df_te)
                print(f"  Loaded TRAIN ({len(df_tr)}) + TEST ({len(df_te)}) = "
                      f"{len(df_tr)+len(df_te)} instances")
            else:
                print(f"  Loaded TRAIN only ({len(df_tr)} instances)")
            df_all = _pd.concat(frames, ignore_index=True)
            y_raw  = df_all.iloc[:, 0].values.astype(int)
            X      = df_all.iloc[:, 1:].values.astype(float)
            # Binary: Normal (1) vs Abnormal (2-5)
            y = (y_raw == 1).astype(int)
            feat = [f't{i:03d}' for i in range(X.shape[1])]
            print(f"  Classes: Normal={np.sum(y==1)}, Abnormal={np.sum(y==0)}")
            return X, y, feat, 'ECG5000 (Normal vs Abnormal)'
        except Exception as e:
            print(f"  Could not load ECG5000 from {path}: {e}")
            print("  Falling back to synthetic ECG-like data...")

    # ── Synthetic fallback ─────────────────────────────────────────────
    print("  ECG5000 path not provided — using synthetic ECG-like data.")
    print("  This demonstrates GEMEX on time series without requiring downloads.")
    print("  To use real ECG5000: --ecg-path ECG5000/ECG5000_TRAIN.txt")
    rng   = np.random.RandomState(42)
    T     = 140
    n     = 1000
    t_arr = np.linspace(0, 2*np.pi, T)

    def make_ecg(cls, n_samples):
        signals = []
        for _ in range(n_samples):
            if cls == 0:   # Normal: clean QRS complex
                sig  = (0.8*np.sin(2*t_arr) +
                        0.3*np.exp(-((t_arr-np.pi)**2)/0.05) +
                        rng.normal(0, 0.05, T))
            else:          # Abnormal: irregular beat patterns
                sig  = (0.5*np.sin(3*t_arr + rng.uniform(0, np.pi)) +
                        0.6*np.exp(-((t_arr-np.pi)**2)/0.15) +
                        rng.normal(0, 0.12, T))
            signals.append(sig)
        return np.array(signals)

    X0 = make_ecg(0, n//2)
    X1 = make_ecg(1, n//2)
    X  = np.vstack([X0, X1])
    y  = np.array([0]*(n//2) + [1]*(n//2))
    feat = [f't{i:03d}' for i in range(T)]
    return X, y, feat, 'ECG5000 Synthetic (Normal vs Abnormal)'



def plot_temporal_gsf(results_per_class, class_names, ds_name,
                       n_steps, save_path, t):
    """
    Temporal attribution profile: GSF score per time step, one line per class.
    Shows which time steps are most discriminative for each class.
    """
    class_colors = [t['c1'], t['c2'], t['c3'], t['c4'], t['c5'], t['c6']]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(t['bg'])
    for ax in axes:
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.6)
        ax.tick_params(colors=t['text2'], labelsize=9)
        ax.grid(color=t['grid'], lw=0.5, alpha=0.6)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.88,
                        bottom=0.10, hspace=0.22)

    # Panel 1: GSF magnitude profiles
    x_arr = np.arange(n_steps)
    for cls, (gsf_mean, gsf_std) in results_per_class.items():
        col  = class_colors[cls % len(class_colors)]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        axes[0].plot(x_arr, np.abs(gsf_mean[:n_steps]),
                     color=col, lw=1.8, label=name)
        axes[0].fill_between(x_arr,
                              np.abs(gsf_mean[:n_steps]) - gsf_std[:n_steps],
                              np.abs(gsf_mean[:n_steps]) + gsf_std[:n_steps],
                              color=col, alpha=0.12)

    axes[0].set_ylabel('|GSF| attribution', color=t['text2'], fontsize=10)
    axes[0].set_title('Geodesic Sensitivity — Temporal Attribution Profile',
                      color=t['text'], fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.4,
                   facecolor=t['panel'], edgecolor=t['border'],
                   labelcolor=t['text'])

    # Panel 2: Signed GSF for the first two classes
    for cls_i, (cls, (gsf_mean, gsf_std)) in enumerate(
            list(results_per_class.items())[:2]):
        col  = class_colors[cls % len(class_colors)]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        axes[1].plot(x_arr, gsf_mean[:n_steps],
                     color=col, lw=1.8, label=name)
        axes[1].axhline(0, color=t['border'], lw=0.8, ls='--')

    axes[1].set_ylabel('Signed GSF', color=t['text2'], fontsize=10)
    axes[1].set_xlabel('Time step / Feature index', color=t['text2'], fontsize=10)
    axes[1].set_title('Signed Attribution — Positive vs Negative Influence',
                      color=t['text'], fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=8.5, framealpha=0.4,
                   facecolor=t['panel'], edgecolor=t['border'],
                   labelcolor=t['text'])

    fig.suptitle(f'{ds_name}  ·  GEMEX Temporal Attribution',
                 fontsize=13, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_ricci_per_class(ricci_by_class, class_names, ds_name,
                          save_path, t):
    """Boxplot of Ricci scalar per class — shows manifold geometry variation."""
    n_cls  = len(ricci_by_class)
    colors = [t['c1'], t['c2'], t['c3'], t['c4'], t['c5'], t['c6']]

    fig, ax = plt.subplots(figsize=(max(6, n_cls*1.5), 5))
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['panel'])
    for sp in ax.spines.values():
        sp.set_color(t['border']); sp.set_linewidth(0.6)
    ax.tick_params(colors=t['text2'], labelsize=10)
    ax.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)
    plt.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18)

    positions = list(range(n_cls))
    data_list = [ricci_by_class[cls] for cls in sorted(ricci_by_class.keys())]
    labels    = [class_names[cls] if cls < len(class_names) else str(cls)
                 for cls in sorted(ricci_by_class.keys())]

    bp = ax.boxplot(data_list, positions=positions, patch_artist=True,
                    medianprops=dict(color='white', lw=2))
    for patch, col in zip(bp['boxes'], colors[:n_cls]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for element in ['whiskers', 'caps', 'fliers']:
        plt.setp(bp[element], color=t['text2'])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha='right',
                       fontsize=9, color=t['text'])
    ax.set_ylabel('Ricci scalar (manifold curvature)',
                  color=t['text2'], fontsize=10)
    ax.set_title(f'{ds_name}  ·  Manifold Curvature by Class\n'
                 f'(GEMEX-exclusive — higher = more curved decision boundary)',
                 color=t['text'], fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def plot_top_features_heatmap(top_feats_by_class, class_names,
                               ds_name, save_path, t):
    """Heatmap of mean |GSF| for top-15 features across classes."""
    all_feat_names = sorted({fn for cls_feats in top_feats_by_class.values()
                              for fn, _ in cls_feats})[:15]
    cls_list = sorted(top_feats_by_class.keys())
    data = np.zeros((len(cls_list), len(all_feat_names)))
    for ci, cls in enumerate(cls_list):
        feat_dict = dict(top_feats_by_class[cls])
        for fi, fn in enumerate(all_feat_names):
            data[ci, fi] = feat_dict.get(fn, 0.0)

    fig, ax = plt.subplots(figsize=(14, max(4, len(cls_list)*0.9)))
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['panel'])
    for sp in ax.spines.values():
        sp.set_color(t['border']); sp.set_linewidth(0.6)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.22)

    cmap = 'YlOrRd' if t is LIGHT else 'inferno'
    im   = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(all_feat_names)))
    ax.set_xticklabels(all_feat_names, rotation=45, ha='right',
                       fontsize=8, color=t['text'])
    ax.set_yticks(range(len(cls_list)))
    ax.set_yticklabels([class_names[c] if c < len(class_names) else str(c)
                        for c in cls_list],
                       fontsize=9, color=t['text'])
    for ci in range(len(cls_list)):
        for fi in range(len(all_feat_names)):
            ax.text(fi, ci, f'{data[ci,fi]:.3f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if data[ci,fi] > data.max()*0.5 else t['text'])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Mean |GSF|', color=t['text2'], fontsize=9)
    cb.ax.tick_params(colors=t['text2'])
    ax.set_title(f'{ds_name}  ·  Top Feature Attribution by Class',
                 fontsize=11, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def _plot_top10_gsf(result, feat, cls, class_names, ds_name, save_path, t):
    """
    Clean top-10 feature bar chart for time series.
    Shows the 10 most important time steps / features by |GSF|,
    with direction. Far more readable than plotting all 140 steps.
    """
    gsf   = result.gsf_scores
    top_n = min(10, len(gsf))
    top_idx = np.argsort(np.abs(gsf))[::-1][:top_n]
    vals    = gsf[top_idx]
    names   = [feat[i] for i in top_idx]
    colors  = [t['c1'] if v >= 0 else '#E24B4A' for v in vals]
    cls_name = class_names[cls] if cls < len(class_names) else str(cls)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['panel'])
    for sp in ax.spines.values():
        sp.set_color(t['border']); sp.set_linewidth(0.6)
    ax.tick_params(colors=t['text2'], labelsize=10)
    ax.grid(axis='x', color=t['grid'], lw=0.5, alpha=0.6)
    plt.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.12)

    y_pos = np.arange(top_n)
    ax.barh(y_pos, vals, color=colors, alpha=0.85, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, color=t['text'])
    ax.axvline(0, color=t['border'], lw=0.8)
    ax.set_xlabel('GSF Score (geodesic sensitivity)', color=t['text2'], fontsize=10)
    for yi, v in enumerate(vals):
        ax.text(v + (abs(v) or 0.001)*0.05, yi, f'{v:+.4f}',
                va='center', fontsize=8.5, color=t['text'])

    pred = result.class_names[result.prediction] if result.class_names else str(result.prediction)
    prob = result.prediction_proba[result.prediction]
    ax.set_title(
        f'{ds_name}  ·  Class: {cls_name}  ·  Top-10 Most Important Features\n'
        f'Predicted: {pred} ({prob:.0%})  ·  Ricci = {result.manifold_curvature:.4f}',
        color=t['text'], fontsize=10, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"    -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# RUN DATASET
# ═══════════════════════════════════════════════════════════════════════

def run_dataset(X, y, feat, ds_name, save_dir, theme, n_explain):
    t = DARK if theme == 'dark' else LIGHT
    os.makedirs(save_dir, exist_ok=True)
    tag = ds_name.lower().replace(' ', '_').replace('(','').replace(')','')

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.20, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)

    classes    = sorted(np.unique(y_te).tolist())
    n_cls      = len(classes)
    if n_cls == 2:
        te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"  Test-AUC = {te_auc:.4f}")
    else:
        acc = np.mean(model.predict(X_te) == y_te)
        print(f"  Test accuracy = {acc:.4f}")
    print(classification_report(y_te, model.predict(X_te),
                                  zero_division=0))

    class_names = ['Normal','Abnormal'] if n_cls == 2 else \
                  ['Walking','UpStairs','DownStairs',
                   'Sitting','Standing','Laying'][:n_cls]

    n_ts = min(X.shape[1], 140)   # number of time steps / features to plot

    cfg = GemexConfig(
        n_geodesic_steps    = 12,
        n_reference_samples = 25,
        interaction_order   = 1,   # 1 for high-dim time series
        verbose             = False)

    exp = Explainer(model, data_type='timeseries',
                    feature_names=feat, class_names=class_names,
                    config=cfg)

    # Explain n_explain instances per class
    results_per_class  = {}
    ricci_by_class     = {}
    top_feats_by_class = {}

    for cls in classes:
        cls_idx = np.where(y_te == cls)[0][:n_explain]
        if len(cls_idx) == 0: continue

        gsf_list, ricci_list = [], []
        print(f"  Class {cls} ({class_names[cls] if cls < len(class_names) else cls}): "
              f"explaining {len(cls_idx)} instances...")

        for i in cls_idx:
            r = exp.explain(X_te[i], X_reference=X_tr)
            gsf_list.append(r.gsf_scores)
            ricci_list.append(abs(r.manifold_curvature))

        gsf_arr  = np.array(gsf_list)
        gsf_mean = gsf_arr.mean(axis=0)
        gsf_std  = gsf_arr.std(axis=0)

        results_per_class[cls]  = (gsf_mean, gsf_std)
        ricci_by_class[cls]     = ricci_list

        # Top 15 features by mean |GSF|
        top_idx = np.argsort(np.abs(gsf_mean))[::-1][:15]
        top_feats_by_class[cls] = [(feat[i], float(np.abs(gsf_mean[i])))
                                    for i in top_idx]

        print(f"    mean_Ricci={np.mean(ricci_list):.4f}  "
              f"top3_feat={[feat[i] for i in np.argsort(np.abs(gsf_mean))[::-1][:3]]}")

        # Top-10 GSF features plot for first instance (clean, readable)
        r0 = exp.explain(X_te[cls_idx[0]], X_reference=X_tr)
        _plot_top10_gsf(r0, feat, cls, class_names, ds_name,
                        f"{save_dir}/{tag}_class{cls}_top10_{theme}.png", t)

    # ── Multi-class summary plots ──────────────────────────────────────
    plot_temporal_gsf(results_per_class, class_names, ds_name,
                      n_ts, f"{save_dir}/{tag}_temporal_gsf_{theme}.png", t)
    plot_ricci_per_class(ricci_by_class, class_names, ds_name,
                          f"{save_dir}/{tag}_ricci_boxplot_{theme}.png", t)
    plot_top_features_heatmap(top_feats_by_class, class_names, ds_name,
                               f"{save_dir}/{tag}_feature_heatmap_{theme}.png", t)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  {ds_name} — Ricci scalar by class")
    print(f"  {'Class':<20}  {'mean Ricci':>12}  {'std':>8}  {'n':>4}")
    print(f"  {'-'*50}")
    for cls in sorted(ricci_by_class.keys()):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        vals = ricci_by_class[cls]
        print(f"  {name:<20}  {np.mean(vals):>12.4f}  "
              f"{np.std(vals):>8.4f}  {len(vals):>4}")

    # ── CSV ────────────────────────────────────────────────────────────
    rows = []
    for cls in sorted(ricci_by_class.keys()):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        for rv in ricci_by_class[cls]:
            rows.append({'dataset': ds_name, 'class': cls,
                         'class_name': name, 'ricci': rv})
    df = pd.DataFrame(rows)
    csv_path = f"{save_dir}/{tag}_ricci.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX — Time Series Example (ECG5000)')
    parser.add_argument('--ecg-path',   default=None,
                        help='Path to ECG5000_TRAIN.txt. If omitted, the script '
                             'auto-detects ECG5000/ECG5000_TRAIN.txt next to '
                             'this script file. Falls back to synthetic data.')
    parser.add_argument('--save-dir',   default='./gemex_timeseries')
    parser.add_argument('--n-explain',  type=int, default=5,
                        help='Instances to explain per class (default 5)')
    parser.add_argument('--theme',      default='dark',
                        choices=['dark', 'light'])
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GEMEX — Time Series Explanation")
    print("="*60)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  ECG5000 — Electrocardiogram Classification")
    print("="*60)
    X, y, feat, ds_name = load_ecg5000(args.ecg_path)
    if X is not None:
        run_dataset(X, y, feat, ds_name,
                    args.save_dir, args.theme, args.n_explain)

    print(f"\n  All outputs: {os.path.abspath(args.save_dir)}/")


if __name__ == '__main__':
    main()
