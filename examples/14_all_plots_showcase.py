# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
14_all_plots_showcase.py
============================
One-stop showcase of every plot type available in GEMEX v1.2.2.

Dataset: Pima Indians Diabetes (UCI, 8 features)

Produces one output PNG per plot kind for a positive (Diabetes) instance:
  gsf_bar              — feature attribution bar with uncertainty
  force                — force/push-pull diagram
  waterfall            — cumulative attribution from baseline to prediction  
  heatmap              — feature × instance GSF grid (20-instance batch)    
  curvature            — geodesic arc-length profile                         
  triplet_hypergraph   — three-way RCT interactions as hypergraph            
  network              — holonomy interaction network (PTI)
  attention_heatmap    — FAS geodesic attention heatmap
  attention_dwell      — FAS dwell-time bar chart
  attention_vs_effect  — attention vs attribution scatter
  bias                 — bias trap detection (BTD)
  beeswarm             — batch GSF distribution
  image_trio           — not applicable to tabular data

interaction_order=3 is used so that all plot types including
triplet_hypergraph are available.

Requirements
------------
  pip install gemex scikit-learn pandas matplotlib

Usage
-----
  python 14_all_plots_showcase.py
  python 14_all_plots_showcase.py --pima-path ./pima_diabetes.csv
  python 14_all_plots_showcase.py --theme light --save-dir ./showcase_results
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

def load_pima(path=None):
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if 'Outcome' not in df.columns:
            df = pd.read_csv(path, names=cols)
    else:
        try:
            url = ('https://raw.githubusercontent.com/jbrownlee/Datasets/'
                   'master/pima-indians-diabetes.data.csv')
            df = pd.read_csv(url, names=cols)
        except Exception:
            print("  Could not fetch Pima data. "
                  "Pass --pima-path to a local CSV.")
            return None, None, None
    feat = [c for c in df.columns if c != 'Outcome']
    return df[feat].values.astype(float), df['Outcome'].values.astype(int), feat


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX v1.2.2 — All plots showcase on Pima Diabetes')
    parser.add_argument('--pima-path', default=None)
    parser.add_argument('--theme',     default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--save-dir',  default='./all_plots_showcase')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    th = args.theme

    print("\n" + "="*60)
    print("  GEMEX v1.2.2 — All plots showcase  (Pima Diabetes)")
    print("="*60)

    X, y, feat = load_pima(args.pima_path)
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

    # ── GEMEX config — order=3 to unlock triplet_hypergraph ──────────
    print(f"\n  GEMEX config: interaction_order=3")
    print(f"  (C({len(feat)},3) = "
          f"{len(feat)*(len(feat)-1)*(len(feat)-2)//6} triplets per instance)\n")

    cfg = GemexConfig(
        n_geodesic_steps    = 20,
        n_reference_samples = 60,
        interaction_order   = 3,   # enables all plot types
        rst_n_components    = 5,
        verbose             = False,
    )
    exp = Explainer(model, data_type='tabular',
                    feature_names=feat,
                    class_names=['No Diabetes', 'Diabetes'],
                    config=cfg)

    # ── Explain one positive instance ────────────────────────────────
    pos_idx = np.where(y_te == 1)[0][0]
    print("  Explaining positive instance (Diabetes) ...")
    r = exp.explain(X_te[pos_idx], X_reference=X_tr)

    print(f"\n  pred={['No Diabetes','Diabetes'][r.prediction]}"
          f"  p={r.prediction_proba[r.prediction]:.3f}"
          f"  Ricci={r.manifold_curvature:.4f}"
          f"  FIM={r.fim_quality}")
    print(r.summary())

    # ── Batch for heatmap and beeswarm ───────────────────────────────
    print("\n  Running batch explanation (20 instances) for heatmap/beeswarm ...")
    batch = exp.explain_batch(X_te[:20], X_reference=X_tr)

    # ── Plot every kind ──────────────────────────────────────────────
    plot_specs = [
        # (kind,           kwargs,                        description)
        ('gsf_bar',          {},                           'GSF attribution bar'),
        ('force',            {},                           'Force / push-pull diagram'),
        ('waterfall',        {},                           'Cumulative waterfall'),
        ('heatmap',          {'batch_results': batch},     'Feature x instance heatmap'),
        ('curvature',        {},                           'Geodesic arc-length profile'),
        ('triplet_hypergraph', {'top_n': 10},              'RCT triplet hypergraph'),
        ('network',          {},                           'Holonomy interaction network'),
        ('attention_heatmap',{},                           'Geodesic attention heatmap'),
        ('attention_dwell',  {},                           'Geodesic dwell-time bar chart'),
        ('attention_vs_effect',{},                         'Attention vs effect scatter'),
        ('bias',             {},                           'Bias trap detection (BTD)'),
        ('beeswarm',         {'batch_results': batch},     'Batch beeswarm distribution'),
    ]

    print(f"\n  Saving {len(plot_specs)} plots to {args.save_dir}/\n")
    saved, skipped = 0, 0

    for kind, kwargs, desc in plot_specs:
        try:
            fig  = r.plot(kind, theme=th, **kwargs)
            path = f"{args.save_dir}/pima_{kind}_{th}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓  {desc:<42}  {os.path.basename(path)}")
            saved += 1
        except Exception as e:
            print(f"  ✗  {desc:<42}  ({e})")
            skipped += 1

    print(f"\n  Saved: {saved}  |  Skipped: {skipped}")
    print(f"\n  All outputs saved to: {os.path.abspath(args.save_dir)}/")


if __name__ == '__main__':
    main()
