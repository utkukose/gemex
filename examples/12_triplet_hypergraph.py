# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
12_triplet_hypergraph.py
========================
Demonstrates the Triplet Hypergraph plot added to result.plot() in v1.2.2.

The Riemannian Curvature Triplet (RCT) measures three-way feature
interactions via the Riemann curvature tensor.  A non-zero RCT(i,j,k)
means features i and j interact differently in the presence of feature k —
a modulation that cannot be captured by any combination of pairwise PTI values.

The hypergraph visualises:
  - Feature nodes sized by |GSF| attribution
  - Triangles = triplets (i × j → k):  gold = synergistic, purple = antagonistic
  - Edge thickness encodes |RCT| magnitude
  - Dashed lines = probe feature being modulated

NOTE: interaction_order=3 is required.  This is slower than order=2 because
GEMEX computes C(n_features, 3) curvature tensor entries.  For 13 features
that is 286 triplets.  Expect ~10-30 seconds per instance on CPU.

Dataset: Cleveland Heart Disease (UCI, 13 features)

Requirements
------------
  pip install gemex scikit-learn pandas matplotlib

Usage
-----
  python 12_triplet_hypergraph.py
  python 12_triplet_hypergraph.py --heart-path ./cleveland_heart.csv
  python 12_triplet_hypergraph.py --top-n 10 --theme light
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
from sklearn.neural_network import MLPClassifier
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
        description='GEMEX v1.2.2 — Triplet Hypergraph (RCT three-way interactions)')
    parser.add_argument('--heart-path', default=None)
    parser.add_argument('--model',      default='gbm',
                        choices=['gbm', 'mlp'],
                        help='gbm = GradientBoosting, mlp = MLPClassifier')
    parser.add_argument('--top-n',      type=int, default=12,
                        help='Number of top triplets to display (default 12)')
    parser.add_argument('--theme',      default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--save-dir',   default='./gemex_triplet_plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    th = args.theme

    print("\n" + "="*60)
    print("  GEMEX v1.2.2 — Triplet Hypergraph (RCT)")
    print("="*60)

    X, y, feat = load_heart(args.heart_path)
    if X is None:
        return

    # ── Train model ──────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_s      = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.20, random_state=42, stratify=y)

    if args.model == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              activation='tanh', max_iter=400,
                              random_state=42, early_stopping=True)
        model_label = 'MLP'
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42)
        model_label = 'GBM'

    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"\n  {model_label}  Test-AUC = {auc:.4f}")

    # ── GEMEX config — interaction_order=3 required for RCT ──────────
    print(f"\n  Configuring GEMEX with interaction_order=3 ...")
    print(f"  (Computing C({len(feat)},3) = "
          f"{len(feat)*(len(feat)-1)*(len(feat)-2)//6} triplets per instance)")
    print("  This may take 10-30 seconds per instance on CPU.\n")

    cfg = GemexConfig(
        n_geodesic_steps    = 20,
        n_reference_samples = 60,
        interaction_order   = 3,   # REQUIRED for triplet_hypergraph
        verbose             = False,
    )
    exp = Explainer(model, data_type='tabular',
                    feature_names=feat,
                    class_names=['No Disease', 'Disease'],
                    config=cfg)

    # ── Explain one positive and one negative instance ───────────────
    pos_idx = np.where(y_te == 1)[0][0]
    neg_idx = np.where(y_te == 0)[0][0]

    for x_idx, label in [(pos_idx, 'positive'), (neg_idx, 'negative')]:
        print(f"  Explaining [{label}] instance ...")
        r = exp.explain(X_te[x_idx], X_reference=X_tr)

        pred_name = ['No Disease', 'Disease'][r.prediction]
        print(f"    pred={pred_name}"
              f"  p={r.prediction_proba[r.prediction]:.3f}"
              f"  Ricci={r.manifold_curvature:.4f}"
              f"  FIM={r.fim_quality}")

        if r.rct is None:
            print("    WARNING: RCT is None — interaction_order=3 may not "
                  "have been applied. Check config.")
            continue

        n_trips = len(r.rct.get('top_triplets', []))
        print(f"    RCT triplets computed: {n_trips}")
        print(r.summary())

        # ── Triplet Hypergraph ────────────────────────────────────────
        fig = r.plot('triplet_hypergraph', theme=th, top_n=args.top_n)
        path = (f"{args.save_dir}/heart_{label}_{args.model}"
                f"_triplet_hypergraph_{th}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> triplet_hypergraph: {path}")

        # ── Also save gsf_bar and network for context ─────────────────
        for kind in ['gsf_bar', 'network']:
            fig = r.plot(kind, theme=th)
            path2 = (f"{args.save_dir}/heart_{label}_{args.model}"
                     f"_{kind}_{th}.png")
            fig.savefig(path2, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> {kind}: {path2}")

        print()

    print(f"  All outputs saved to: {os.path.abspath(args.save_dir)}/")
    print("\n  Reading the hypergraph:")
    print("  - Each triangle connects three features (i × j → k)")
    print("  - Gold triangles: synergistic three-way interaction (RCT > 0)")
    print("  - Purple triangles: antagonistic (RCT < 0)")
    print("  - Node size: |GSF| attribution magnitude")
    print("  - Edge thickness: |RCT| strength")
    print("  - Dashed line: probe feature k being modulated by i and j")


if __name__ == '__main__':
    main()
