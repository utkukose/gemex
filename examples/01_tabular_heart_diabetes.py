# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
01_tabular_heart_diabetes.py
=============================
GEMEX on real-world medical tabular datasets:
  - Cleveland Heart Disease (UCI, 13 clinical features)
  - Pima Indians Diabetes   (UCI, 8 physiological features)

Demonstrates:
  - Instance-level GSF attribution with uncertainty bands
  - Holonomy-based feature interactions (PTI)
  - Manifold curvature (Ricci scalar) per instance
  - Feature Attention Sequence (FAS) — geodesic dwell time
  - Bias Trap Detection (BTD)
  - All GEMEX visualisation plots saved to disk

Requirements
------------
  pip install gemex scikit-learn pandas matplotlib

Usage
-----
  python 01_tabular_heart_diabetes.py
  python 01_tabular_heart_diabetes.py --dataset heart
  python 01_tabular_heart_diabetes.py --dataset pima --theme light
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from gemex import Explainer, GemexConfig

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_heart(path=None):
    """
    Cleveland Heart Disease — UCI ML Repository.
    303 instances, 13 features, binary target (0=no disease, 1=disease).
    Download: https://archive.ics.uci.edu/ml/datasets/heart+Disease
    """
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']
    if path and os.path.exists(path):
        df = pd.read_csv(path, names=cols, na_values='?').dropna()
    else:
        # Fetch from UCI if no local file provided
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
               'heart-disease/processed.cleveland.data')
        try:
            df = pd.read_csv(url, names=cols, na_values='?').dropna()
        except Exception:
            print("  Could not fetch Heart Disease data. "
                  "Pass --heart-path to a local CSV.")
            return None, None, None, None
    df['target'] = (df['target'] > 0).astype(int)
    feat = [c for c in df.columns if c != 'target']
    return (df[feat].values.astype(float), df['target'].values.astype(int),
            feat, 'Cleveland Heart Disease')


def load_pima(path=None):
    """
    Pima Indians Diabetes — UCI ML Repository.
    768 instances, 8 features, binary target (0=no diabetes, 1=diabetes).
    Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if list(df.columns) == cols[:-1]:   # no header
            df = pd.read_csv(path, names=cols)
    else:
        try:
            url = ('https://raw.githubusercontent.com/jbrownlee/Datasets/'
                   'master/pima-indians-diabetes.data.csv')
            df = pd.read_csv(url, names=cols)
        except Exception:
            print("  Could not fetch Pima data. "
                  "Pass --pima-path to a local CSV.")
            return None, None, None, None
    feat = [c for c in df.columns if c != 'Outcome']
    return (df[feat].values.astype(float), df['Outcome'].values.astype(int),
            feat, 'Pima Indians Diabetes')


# ═══════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════

def train(X, y, ds_name, model_type='gbm', random_state=42):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.20, random_state=random_state, stratify=y)

    if model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                              activation='tanh', max_iter=300,
                              random_state=random_state,
                              early_stopping=True)
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=random_state)

    model.fit(X_tr, y_tr)

    cv = cross_val_score(model, X_s, y,
                         cv=StratifiedKFold(5, shuffle=True,
                                            random_state=random_state),
                         scoring='roc_auc').mean()
    te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"  [{ds_name}]  CV-AUC={cv:.4f}   Test-AUC={te_auc:.4f}")
    print(classification_report(y_te, model.predict(X_te),
                                 target_names=['No', 'Yes'],
                                 zero_division=0))
    return model, scaler, X_tr, X_te, y_tr, y_te


# ═══════════════════════════════════════════════════════════════════════
# GEMEX EXPLAIN + PLOTS
# ═══════════════════════════════════════════════════════════════════════

def run_gemex(model, X_tr, X_te, y_te, feat, class_names,
              ds_name, save_dir, theme):
    os.makedirs(save_dir, exist_ok=True)
    t_str = theme

    cfg = GemexConfig(
        n_geodesic_steps    = 20,
        n_reference_samples = 60,
        interaction_order   = 2,
        rst_n_components    = 5,
        verbose             = False)

    exp = Explainer(model, data_type='tabular',
                    feature_names=feat, class_names=class_names,
                    config=cfg)

    # Select one positive and one negative instance
    pos_idx = np.where(y_te == 1)[0]
    neg_idx = np.where(y_te == 0)[0]
    instances = [(pos_idx[0], 'positive'), (neg_idx[0], 'negative')]

    results = []
    for idx, label in instances:
        x   = X_te[idx]
        r   = exp.explain(x, X_reference=X_tr)
        results.append(r)

        print(f"\n  Instance [{label}]  pred={class_names[r.prediction]}"
              f"  p={r.prediction_proba[r.prediction]:.3f}"
              f"  Ricci={r.manifold_curvature:.4f}"
              f"  FIM={r.fim_quality}")
        print(r.summary())

        # All available plots
        plots = ['gsf_bar', 'force', 'heatmap', 'waterfall',
                 'uncertainty', 'curvature', 'network',
                 'attention_heatmap', 'attention_dwell',
                 'attention_vs_effect', 'bias']
        for kind in plots:
            try:
                fig = r.plot(kind, theme=theme)
                path = f"{save_dir}/{label}_{kind}_{t_str}.png"
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved: {path}")
            except Exception as e:
                pass   # skip unavailable plots silently

    # Batch beeswarm
    batch = exp.explain_batch(X_te[:10], X_reference=X_tr)
    try:
        fig = results[0].plot('beeswarm', theme=theme,
                               batch_results=batch)
        path = f"{save_dir}/batch_beeswarm_{t_str}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path}")
    except Exception:
        pass

    # Summary table
    print(f"\n  Top features across {len(batch)} instances:")
    print(f"  {'Instance':>8}  {'Pred':>10}  {'p':>6}  "
          f"{'Ricci':>7}  {'Top Feature'}")
    print(f"  {'-'*55}")
    for i, r in enumerate(batch):
        tf = r.top_features(1)[0][0] if r.top_features(1) else '—'
        print(f"  {i:>8}  "
              f"{class_names[r.prediction]:>10}  "
              f"{r.prediction_proba[r.prediction]:>6.3f}  "
              f"{r.manifold_curvature:>7.4f}  {tf}")

    return results, batch


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX — Tabular Medical Data Example')
    parser.add_argument('--dataset',    default='both',
                        choices=['heart', 'pima', 'both'])
    parser.add_argument('--heart-path', default=None,
                        help='Path to cleveland_heart.csv')
    parser.add_argument('--pima-path',  default=None,
                        help='Path to pima_diabetes.csv')
    parser.add_argument('--model',      default='gbm',
                        choices=['gbm', 'mlp'])
    parser.add_argument('--theme',      default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--save-dir',   default='./gemex_tabular')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GEMEX — Tabular Medical Data")
    print("="*60)

    datasets = []
    if args.dataset in ('heart', 'both'):
        datasets.append(('heart', load_heart(args.heart_path),
                         ['No Disease', 'Disease']))
    if args.dataset in ('pima', 'both'):
        datasets.append(('pima', load_pima(args.pima_path),
                         ['No Diabetes', 'Diabetes']))

    for tag, (X, y, feat, ds_name), class_names in datasets:
        if X is None:
            continue
        print(f"\n{'='*60}")
        print(f"  {ds_name}  —  {X.shape[0]} instances, {X.shape[1]} features")
        print(f"{'='*60}")

        model, scaler, X_tr, X_te, y_tr, y_te = train(
            X, y, ds_name, model_type=args.model)

        run_gemex(model, X_tr, X_te, y_te, feat, class_names,
                  ds_name, f"{args.save_dir}/{tag}", args.theme)

    print(f"\n  All outputs: {os.path.abspath(args.save_dir)}/")


if __name__ == '__main__':
    main()
