# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
11_gemex_tabular_plots.py
=======================
Demonstrates three plot types added in v1.2.2 on tabular medical data:

  waterfall     — cumulative GSF attribution from baseline to prediction
  heatmap       — feature × instance GSF grid over a batch
  curvature     — geodesic arc-length profile (manifold curvature along path)

Dataset: Cleveland Heart Disease (UCI, 13 features)

Requirements
------------
  pip install gemex scikit-learn pandas matplotlib

Usage
-----
  python 11_gemex_tabular_plots.py
  python 11_gemex_tabular_plots.py --heart-path ./cleveland_heart.csv
  python 11_gemex_tabular_plots.py --theme light --save-dir ./gemex_tabular_plots
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from gemex import Explainer, GemexConfig


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_heart(path=None):
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']
    if path and os.path.exists(path):
        df = pd.read_csv(path, names=cols, na_values='?').dropna()
    else:
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
               'heart-disease/processed.cleveland.data')
        try:
            df = pd.read_csv(url, names=cols, na_values='?').dropna()
        except Exception:
            print("  Could not fetch Heart Disease data. "
                  "Pass --heart-path to a local CSV.")
            return None, None, None
    df['target'] = (df['target'] > 0).astype(int)
    feat = [c for c in df.columns if c != 'target']
    return df[feat].values.astype(float), df['target'].values.astype(int), feat


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX v1.2.2 — Alternative plots: waterfall, heatmap, curvature')
    parser.add_argument('--heart-path', default=None)
    parser.add_argument('--theme',      default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--save-dir',   default='./gemex_tabular_plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    th = args.theme

    print("\n" + "="*60)
    print("  GEMEX v1.2.2 — Alternative plots on tabular data")
    print("="*60)

    X, y, feat = load_heart(args.heart_path)
    if X is None:
        return

    # ── Train GBM ────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_s      = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.20, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"\n  GBM  Test-AUC = {auc:.4f}")

    # ── GEMEX config ─────────────────────────────────────────────────
    # interaction_order=2 gives attention sequence (needed for curvature overlay)
    cfg = GemexConfig(
        n_geodesic_steps    = 20,
        n_reference_samples = 60,
        interaction_order   = 2,
        verbose             = False,
    )
    exp = Explainer(model, data_type='tabular',
                    feature_names=feat,
                    class_names=['No Disease', 'Disease'],
                    config=cfg)

    # ── Pick one positive and one negative instance ──────────────────
    pos_idx = np.where(y_te == 1)[0][0]
    neg_idx = np.where(y_te == 0)[0][0]

    print("\n  Explaining instances ...")
    r_pos = exp.explain(X_te[pos_idx], X_reference=X_tr)
    r_neg = exp.explain(X_te[neg_idx], X_reference=X_tr)

    for r, label in [(r_pos, 'positive'), (r_neg, 'negative')]:
        pred_name = ['No Disease', 'Disease'][r.prediction]
        print(f"\n  [{label}]  pred={pred_name}"
              f"  p={r.prediction_proba[r.prediction]:.3f}"
              f"  Ricci={r.manifold_curvature:.4f}"
              f"  FIM={r.fim_quality}")

        # ── 1. WATERFALL ─────────────────────────────────────────────
        fig = r.plot('waterfall', theme=th)
        path = f"{args.save_dir}/heart_{label}_waterfall_{th}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> waterfall:  {path}")

        # ── 2. CURVATURE ─────────────────────────────────────────────
        fig = r.plot('curvature', theme=th)
        path = f"{args.save_dir}/heart_{label}_curvature_{th}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> curvature:  {path}")

    # ── 3. HEATMAP (batch — 20 test instances) ───────────────────────
    print("\n  Running batch explanation for heatmap (20 instances) ...")
    batch = exp.explain_batch(X_te[:20], X_reference=X_tr)

    # Heatmap with the positive instance highlighted
    fig = r_pos.plot('heatmap', theme=th, batch_results=batch)
    path = f"{args.save_dir}/heart_heatmap_{th}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> heatmap:    {path}")

    print(f"\n  All outputs saved to: {os.path.abspath(args.save_dir)}/")


if __name__ == '__main__':
    main()
