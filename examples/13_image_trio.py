# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
13_image_trio.py
================
Demonstrates the GEMEX Image Explanation Suite (image_trio) as a
standalone four-panel figure, separate from the GradCAM comparison plots
in examples 06-10.

The four panels are:
  1. Original       — greyscale input image
  2. GeodesicCAM    — GSF attribution heatmap overlaid on image
                      (upsampled from patch space to pixel space)
  3. ManifoldSeg    — iso-information contour regions derived from FIM
  4. PerturbFlow    — geodesic gradient vector field showing how pixel
                      changes propagate on the statistical manifold

Datasets supported (all from the MedMNIST repository):
  --dataset pneumonia   PneumoniaMNIST   greyscale  2 classes
  --dataset organ       OrganAMNIST      greyscale  11 classes
  --dataset blood       BloodMNIST       RGB → grey  8 classes

MedMNIST data is downloaded automatically via the medmnist package.

Requirements
------------
  pip install gemex medmnist scikit-learn torch torchvision matplotlib

Usage
-----
  python 13_image_trio.py
  python 13_image_trio.py --dataset blood --n-images 4
  python 13_image_trio.py --dataset organ --theme light --save-dir ./trio_results
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from gemex import Explainer, GemexConfig


# ═══════════════════════════════════════════════════════════════════════
# DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════════════

DATASETS = {
    'pneumonia': {
        'medmnist':  'pneumoniamnist',
        'class_name': 'PneumoniaMNIST',
        'classes':   ['Normal', 'Pneumonia'],
        'channels':  1,
        'task':      'chest X-ray pneumonia classification',
    },
    'organ': {
        'medmnist':  'organamnist',
        'class_name': 'OrganAMNIST',
        'classes':   ['Bladder','Femur-L','Femur-R','Heart',
                      'Kidney-L','Kidney-R','Liver','Lung-L',
                      'Lung-R','Pancreas','Spleen'],
        'channels':  1,
        'task':      'abdominal CT organ classification (11 types)',
    },
    'blood': {
        'medmnist':  'bloodmnist',
        'class_name': 'BloodMNIST',
        'classes':   ['Basophil','Eosinophil','Erythroblast',
                      'IG','Lymphocyte','Monocyte','Neutrophil','Platelet'],
        'channels':  3,
        'task':      'blood cell microscopy classification (8 types, RGB)',
    },
}

_NAME_MAP = {
    'pneumoniamnist': 'PneumoniaMNIST',
    'organamnist':    'OrganAMNIST',
    'bloodmnist':     'BloodMNIST',
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════

def load_dataset(dataset_key, n_train=5000, n_test=200):
    info    = DATASETS[dataset_key]
    n_ch    = info['channels']
    classes = info['classes']

    try:
        import medmnist
        DataClass = getattr(medmnist, _NAME_MAP[info['medmnist']])
        print(f"  Loading {info['class_name']} via medmnist ...")
        tr_ds = DataClass(split='train', download=True, size=28)
        te_ds = DataClass(split='test',  download=True, size=28)

        def to_arrays(ds, limit):
            imgs, labels, raws = [], [], []
            for i, (img, lbl) in enumerate(ds):
                if i >= limit:
                    break
                arr = np.array(img, dtype=np.float32) / 255.0
                raws.append(arr)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    grey = (0.299*arr[:,:,0] + 0.587*arr[:,:,1]
                            + 0.114*arr[:,:,2])
                elif arr.ndim == 3:
                    grey = arr[:,:,0]
                else:
                    grey = arr
                imgs.append(grey.flatten())
                labels.append(int(np.array(lbl).flatten()[0]))
            return (np.array(imgs),
                    np.array(labels, dtype=int),
                    raws)

        X_tr, y_tr, raws_tr = to_arrays(tr_ds, n_train)
        X_te, y_te, raws_te = to_arrays(te_ds, n_test)

    except Exception as e:
        print(f"  medmnist load failed: {e}")
        print("  Falling back to synthetic data (28×28 greyscale).")
        n_feat  = 784
        X_tr    = np.random.RandomState(0).rand(200, n_feat).astype(np.float32)
        y_tr    = (X_tr.mean(axis=1) > 0.5).astype(int)
        X_te    = np.random.RandomState(1).rand(50, n_feat).astype(np.float32)
        y_te    = (X_te.mean(axis=1) > 0.5).astype(int)
        raws_te = [x.reshape(28, 28) for x in X_te]

    print(f"  Train: {X_tr.shape}   Test: {X_te.shape}")
    return X_tr, y_tr, X_te, y_te, raws_te, n_ch, classes, info


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='GEMEX v1.2.2 — image_trio standalone for MedMNIST')
    parser.add_argument('--dataset',   default='pneumonia',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--n-images',  type=int, default=3,
                        help='Number of images to explain (one per class '
                             'where possible, default 3)')
    parser.add_argument('--epochs',    type=int, default=15,
                        help='CNN training epochs (default 15)')
    parser.add_argument('--theme',     default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--save-dir',  default='./image_trio_results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    th = args.theme

    print("\n" + "="*60)
    print(f"  GEMEX v1.2.2 — image_trio  [{args.dataset.upper()}]")
    print("="*60)

    (X_tr, y_tr, X_te, y_te,
     raws_te, n_ch, classes, info) = load_dataset(args.dataset)

    n_cls = len(classes)

    # ── Train a simple GBM on greyscale features ──────────────────────
    # (GBM is used for GEMEX; same pipeline as examples 06-10)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"  Training GBM ({n_cls} classes) ...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    if n_cls == 2:
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_tr_s, y_tr)
    else:
        # Multi-class: use sklearn's native multi-class GBM
        model = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42)
        model.fit(X_tr_s, y_tr)

    preds = model.predict(X_te_s)
    acc   = np.mean(preds == y_te)
    print(f"  Test accuracy: {acc:.4f}")

    # ── GEMEX config ──────────────────────────────────────────────────
    # Use patch_size=4 → 49 patches (7×7 grid) for speed
    patch_size = 4
    n_patches  = (28 // patch_size) ** 2   # = 49
    feat_names = [f"patch_{i}" for i in range(n_patches)]

    cfg = GemexConfig(
        n_geodesic_steps    = 12,
        n_reference_samples = 30,
        interaction_order   = 1,       # attribution only — faster for images
        image_patch_size    = patch_size,
        verbose             = False,
    )
    exp = Explainer(model, data_type='image',
                    feature_names=feat_names,
                    class_names=classes,
                    config=cfg)

    # ── Select one image per class (up to n_images) ───────────────────
    selected = {}
    for i, (lbl, raw) in enumerate(zip(y_te, raws_te)):
        if lbl not in selected and len(selected) < args.n_images:
            selected[lbl] = i
        if len(selected) >= min(args.n_images, n_cls):
            break

    print(f"\n  Explaining {len(selected)} images ...")

    for cls, idx in sorted(selected.items()):
        x_flat  = X_te_s[idx]          # flat scaled greyscale (784,)
        x_raw   = raws_te[idx]          # spatial numpy array (H,W) or (H,W,C)

        # Build greyscale 2D for image_trio display
        if np.array(x_raw).ndim == 3 and np.array(x_raw).shape[-1] == 3:
            x_grey = (0.299*x_raw[:,:,0] + 0.587*x_raw[:,:,1]
                      + 0.114*x_raw[:,:,2])
        elif np.array(x_raw).ndim == 3:
            x_grey = x_raw[:,:,0]
        else:
            x_grey = np.array(x_raw).reshape(28, 28)

        # Build patch-level reference from training set
        x_ref = X_tr_s[:30]

        class_name = classes[cls] if cls < len(classes) else str(cls)
        print(f"\n  Class {cls} — {class_name}")

        try:
            r = exp.explain(x_flat, X_reference=x_ref)
            print(f"    FIM={r.fim_quality}  "
                  f"Ricci={r.manifold_curvature:.4f}  "
                  f"pred={classes[r.prediction]}")
        except Exception as e:
            print(f"    GEMEX failed: {e}")
            continue

        # ── image_trio — four panels ──────────────────────────────────
        # Pass x_grey so the Original panel shows the real image.
        # The GeodesicCAM panel uses r.gsf_scores (patch-level) upsampled
        # to pixel space automatically by the plot function.
        fig = r.plot('image_trio', theme=th, image=x_grey)

        ds_tag  = args.dataset
        path    = (f"{args.save_dir}/"
                   f"{ds_tag}_class{cls}_{class_name.replace(' ','_')}"
                   f"_trio_{th}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> image_trio: {path}")

        # Also save gsf_bar for attribution context
        try:
            fig2 = r.plot('gsf_bar', theme=th)
            path2 = (f"{args.save_dir}/"
                     f"{ds_tag}_class{cls}_{class_name.replace(' ','_')}"
                     f"_gsf_bar_{th}.png")
            fig2.savefig(path2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"    -> gsf_bar:    {path2}")
        except Exception:
            pass

    print(f"\n  All outputs saved to: {os.path.abspath(args.save_dir)}/")
    print("\n  Panel guide:")
    print("    Col 1 — Original:      greyscale input image")
    print("    Col 2 — GeodesicCAM:   GSF heatmap (bright = model most sensitive)")
    print("    Col 3 — ManifoldSeg:   iso-information contour regions")
    print("    Col 4 — PerturbFlow:   geodesic gradient vector field")


if __name__ == '__main__':
    main()
