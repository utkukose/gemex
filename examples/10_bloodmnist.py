# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
10_bloodmnist.py
================
GEMEX vs GradCAM vs GradCAM++ vs Saliency — BloodMNIST

Dataset
-------
  Name    : BloodMNIST
  Task    : blood cell microscopy classification (8 cell types)
  Classes : 8
  Note    : RGB 28×28 — converted to greyscale for GEMEX. 8-class.

Reference
---------
  Yang, J. et al. (2023). MedMNIST v2 — A large-scale lightweight benchmark
  for 2D and 3D biomedical image classification. Scientific Data, 10(1), 41.
  https://doi.org/10.1038/s41597-022-01721-8

Explanation methods
-------------------
  GEMEX     : GeodesicCAM — patch-based (7×7=49 patches), model-agnostic
  GradCAM   : Gradient-weighted Class Activation Map (last conv layer)
  GradCAM++ : Improved GradCAM with second-order gradients
  Saliency  : Vanilla input gradient (sparse foreground-only)

Quantitative metrics
--------------------
  Spearman rank correlation (GEMEX vs GradCAM++)
  Top-10% pixel precision (fraction in foreground)
  Deletion AUC (lower = more faithful)
  Ricci scalar (manifold curvature — GEMEX exclusive)

Requirements
------------
  pip install medmnist torch torchvision gemex scipy matplotlib scikit-learn

Usage
-----
  python 10_bloodmnist.py
  python 10_bloodmnist.py --theme light
  python 10_bloodmnist.py --save-dir ./my_results
  python 10_bloodmnist.py --n-images 8 --epochs 20 --backend cnn
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from gemex import Explainer, GemexConfig

# ═══════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════

DARK = dict(
    bg='#0D0D1A', panel='#131326', grid='#1E1E38', border='#2E2E55',
    text='#E8E8F0', text2='#9999BB', text3='#444466',
    gemex='#00C896', cam='#F5C842', cam2='#FF8C42', sal='#C97EFA',
)
LIGHT = dict(
    bg='#F4F4F9', panel='#FFFFFF', grid='#EBEBF5', border='#CCCCDD',
    text='#1A1A2E', text2='#555577', text3='#AAAACC',
    gemex='#0A7A5A', cam='#B8860B', cam2='#C04000', sal='#6A2FA0',
)

# ═══════════════════════════════════════════════════════════════════════
# DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════════════

MEDMNIST_DATASETS = {
    'blood': {
        'name'    : 'BloodMNIST',
        'medmnist': 'bloodmnist',
        'classes' : ['Basophil','Eosinophil','Erythroblast',
                     'Immature Granulocyte','Lymphocyte',
                     'Monocyte','Neutrophil','Platelet'],
        'channels': 3,
        'task'    : 'blood cell microscopy classification',
    },
}
# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_medmnist(dataset_key, n_train=10000, n_test=500):
    """
    Load a MedMNIST dataset.
    Returns X_tr, y_tr, X_te, y_te as float32 arrays (N, H*W*C) in [0,1].
    Also returns n_channels and class_names.
    """
    info = MEDMNIST_DATASETS[dataset_key]
    try:
        import medmnist
        from medmnist import INFO
        _name_map = {
            'pneumoniamnist': 'PneumoniaMNIST',
            'pathmnist':      'PathMNIST',
            'dermamnist':     'DermaMNIST',
            'organamnist':    'OrganAMNIST',
            'bloodmnist':     'BloodMNIST',
        }
        DataClass = getattr(medmnist,
                            _name_map[info['medmnist']])
        print(f"  Loading {info['name']} via medmnist...")
        tr_ds = DataClass(split='train', download=True, size=28)
        te_ds = DataClass(split='test',  download=True, size=28)

        def ds_to_arrays(ds, limit):
            imgs, labels = [], []
            for i, (img, lbl) in enumerate(ds):
                if i >= limit: break
                imgs.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(int(np.array(lbl).flatten()[0]))
            X = np.array(imgs)
            if X.ndim == 4 and X.shape[-1] in (1, 3):
                # Convert RGB to greyscale for GEMEX (pixel features)
                if X.shape[-1] == 3:
                    X_grey = 0.299*X[:,:,:,0] + 0.587*X[:,:,:,1] + \
                             0.114*X[:,:,:,2]
                else:
                    X_grey = X[:,:,:,0]
                X_flat = X_grey.reshape(len(X_grey), -1)
            else:
                X_flat = X.reshape(len(X), -1)
            return X_flat, np.array(labels, dtype=int), X

        X_tr, y_tr, X_tr_raw = ds_to_arrays(tr_ds, n_train)
        X_te, y_te, X_te_raw = ds_to_arrays(te_ds, n_test)
        print(f"  Train: {X_tr.shape}   Test: {X_te.shape}")
        return (X_tr, y_tr, X_te, y_te,
                X_tr_raw, X_te_raw,
                info['channels'], info['classes'], info['name'], info['task'])

    except (ImportError, Exception) as e:
        print(f"  MedMNIST not available ({e})")
        print("  Install with: pip install medmnist")
        print("  Falling back to standard MNIST...")
        return load_standard_mnist()


def load_standard_mnist(n_train=10000, n_test=1000):
    """Standard MNIST fallback via sklearn."""
    print("  Loading standard MNIST (fallback)...")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[60000:60000+n_test], y[60000:60000+n_test]
    X_tr_raw = X_tr.reshape(-1, 28, 28, 1)
    X_te_raw = X_te.reshape(-1, 28, 28, 1)
    classes   = [str(i) for i in range(10)]
    return (X_tr, y_tr, X_te, y_te,
            X_tr_raw, X_te_raw,
            1, classes, 'Standard MNIST (fallback)', 'digit classification')

# ═══════════════════════════════════════════════════════════════════════
# STRONG CNN (MedCNN)
# ═══════════════════════════════════════════════════════════════════════

def train_medcnn(X_tr_raw, y_tr, n_classes, n_channels_in, epochs=20):
    """
    MedCNN: ResNet-inspired medical image classifier.

    Architecture (28×28 input):
      Block 1: Conv(c_in→32) BN ReLU Conv(32→32) BN ReLU MaxPool Dropout(0.25)
      Block 2: Conv(32→64)   BN ReLU Conv(64→64) BN ReLU MaxPool Dropout(0.25)
      Block 3: Conv(64→128)  BN ReLU AdaptiveAvgPool(1×1)
      Head   : FC(128→256) BN ReLU Dropout(0.5) FC(256→n_classes)

    Training: Adam + CosineAnnealingLR, 20 epochs, batch 128.
    Achieves ~94% on PneumoniaMNIST, ~85% on PathMNIST.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class MedCNN(nn.Module):
        def __init__(self, c_in, n_cls):
            super().__init__()

            def conv_bn_relu(c_i, c_o, k=3, p=1):
                return nn.Sequential(
                    nn.Conv2d(c_i, c_o, k, padding=p, bias=False),
                    nn.BatchNorm2d(c_o),
                    nn.ReLU(inplace=True))

            self.block1 = nn.Sequential(
                conv_bn_relu(c_in, 32),
                conv_bn_relu(32, 32),
                nn.MaxPool2d(2),          # 14×14
                nn.Dropout2d(0.25))

            self.block2 = nn.Sequential(
                conv_bn_relu(32, 64),
                conv_bn_relu(64, 64),
                nn.MaxPool2d(2),          # 7×7
                nn.Dropout2d(0.25))

            self.block3 = nn.Sequential(
                conv_bn_relu(64, 128),
                nn.AdaptiveAvgPool2d(1))  # 1×1

            # Store last conv output for GradCAM hook
            self.last_conv = self.block3[0]

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, n_cls))

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return self.head(x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    net = MedCNN(n_channels_in, n_classes).to(device)

    # Prepare tensors — input shape (N, C, H, W)
    if X_tr_raw.ndim == 4:
        if X_tr_raw.shape[-1] in (1, 3):          # (N, H, W, C)
            X_t = torch.tensor(
                X_tr_raw.transpose(0, 3, 1, 2).astype(np.float32))
        else:                                      # already (N, C, H, W)
            X_t = torch.tensor(X_tr_raw.astype(np.float32))
    else:                                          # (N, H*W) flat
        X_t = torch.tensor(
            X_tr_raw.reshape(-1, 1, 28, 28).astype(np.float32))

    y_t  = torch.tensor(y_tr, dtype=torch.long)
    ds   = TensorDataset(X_t, y_t)
    dl   = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    opt  = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    print(f"  Training MedCNN ({epochs} epochs)...")
    net.train()
    for epoch in range(epochs):
        total, correct = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = net(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total   += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
        sched.step()
        acc = correct / len(y_tr)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs}  "
                  f"loss={total/len(dl):.4f}  train_acc={acc:.4f}")

    net.eval()

    # ── Wrapper with predict_proba() ─────────────────────────────────
    class MedCNNWrapper:
        def __init__(self, net, device, c_in):
            self.net    = net
            self.device = device
            self.c_in   = c_in

        def _to_tensor(self, X):
            import torch
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim == 3 and X.shape[-1] in (1, 3):   # (H,W,C) single image
                X = X.transpose(2, 0, 1)[np.newaxis]     # → (1,C,H,W)
            elif X.ndim == 4 and X.shape[-1] in (1, 3): # (N,H,W,C) batch
                X = X.transpose(0, 3, 1, 2)              # → (N,C,H,W)
            elif X.ndim == 2:                            # (N, H*W*C) flat
                X = X.reshape(-1, self.c_in, 28, 28)
            return torch.tensor(X.astype(np.float32)).to(self.device)

        def predict_proba(self, X):
            import torch.nn.functional as F
            with torch.no_grad():
                logits = self.net(self._to_tensor(X))
            return F.softmax(logits, dim=1).cpu().numpy()

        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)

    wrapper = MedCNNWrapper(net, device, n_channels_in)
    return wrapper, net, device


def train_mlp_sklearn(X_tr, y_tr):
    """Sklearn MLP fallback — no PyTorch needed."""
    from sklearn.neural_network import MLPClassifier
    print("  Training MLP (sklearn)...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='tanh',
        solver='adam',
        alpha=1e-4,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False)
    model.fit(X_tr, y_tr)
    return model

# ═══════════════════════════════════════════════════════════════════════
# EXPLANATION METHODS
# ═══════════════════════════════════════════════════════════════════════

def compute_gradcam(net, x_raw, target_class, device, c_in):
    """
    GradCAM: gradient-weighted class activation map.
    Hooks on the last conv layer (block3[0]).
    """
    import torch, torch.nn.functional as F

    acts, grads = {}, {}
    h1 = net.last_conv.register_forward_hook(
        lambda m, i, o: acts.update({'x': o}))
    h2 = net.last_conv.register_full_backward_hook(
        lambda m, gi, go: grads.update({'x': go[0]}))

    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(1, c_in, 28, 28)
    elif x_raw.ndim == 3:
        x_raw = x_raw[None]
    elif x_raw.ndim == 4 and x_raw.shape[-1] == c_in:
        x_raw = x_raw.transpose(0, 3, 1, 2)

    Xt = torch.tensor(x_raw.astype(np.float32)).to(device)
    Xt.requires_grad_(True)
    net.zero_grad()
    logits = net(Xt)
    logits[0, target_class].backward()
    h1.remove(); h2.remove()

    a = acts['x'][0].detach()       # (C, H, W)
    g = grads['x'][0].detach()      # (C, H, W)
    w = g.mean(dim=(1, 2))           # (C,)
    cam = F.relu((w[:, None, None] * a).sum(dim=0)).detach().cpu().numpy()

    from scipy.ndimage import zoom
    if cam.shape[0] != 28:
        cam = zoom(cam, 28 / cam.shape[0])
    return _norm01(cam)


def compute_gradcam_pp(net, x_raw, target_class, device, c_in):
    """
    GradCAM++: second-order gradient weighting.
    More accurate than GradCAM for multi-object scenes.
    Reference: Chattopadhay et al. (2018) WACV.
    """
    import torch, torch.nn.functional as F

    acts, grads = {}, {}
    h1 = net.last_conv.register_forward_hook(
        lambda m, i, o: acts.update({'x': o}))
    h2 = net.last_conv.register_full_backward_hook(
        lambda m, gi, go: grads.update({'x': go[0]}))

    if x_raw.ndim == 1:
        x_raw = x_raw.reshape(1, c_in, 28, 28)
    elif x_raw.ndim == 3:
        x_raw = x_raw[None]
    elif x_raw.ndim == 4 and x_raw.shape[-1] == c_in:
        x_raw = x_raw.transpose(0, 3, 1, 2)

    Xt = torch.tensor(x_raw.astype(np.float32)).to(device)
    Xt.requires_grad_(True)
    net.zero_grad()
    logits = net(Xt)
    score  = logits[0, target_class]
    score.backward(retain_graph=True)
    h1.remove(); h2.remove()

    a = acts['x'][0].detach().cpu().numpy()   # (C, H, W)
    g = grads['x'][0].detach().cpu().numpy()  # (C, H, W)

    # GradCAM++ weights: alpha * ReLU(grad)
    g2 = g ** 2
    g3 = g ** 3
    denom = 2.0 * g2 + a.sum(axis=(1,2), keepdims=True) * g3 + 1e-7
    alpha = g2 / denom
    w     = (alpha * np.maximum(g, 0)).sum(axis=(1,2))   # (C,)
    cam   = (w[:, None, None] * a).sum(axis=0)
    cam   = np.maximum(cam, 0)

    from scipy.ndimage import zoom
    if cam.shape[0] != 28:
        cam = zoom(cam, 28 / cam.shape[0])
    return _norm01(cam)


def compute_saliency(model, x_input, target_class, eps=0.01):
    """
    Fast sparse saliency via finite differences.
    Only computes gradient for non-background pixels (value > 0.05)
    to avoid 784 model calls — typically 100-300 calls on medical images.
    """
    d    = len(x_input)
    grad = np.zeros(d)
    # Only perturb pixels that carry actual signal
    active = np.where(np.abs(x_input) > 0.05)[0]
    if len(active) == 0:
        active = np.arange(d)  # fallback: all pixels
    for i in active:
        xp = x_input.copy(); xp[i] += eps
        xm = x_input.copy(); xm[i] -= eps
        grad[i] = (model.predict_proba(xp.reshape(1,-1))[0, target_class] -
                   model.predict_proba(xm.reshape(1,-1))[0, target_class]) / (2*eps)
    # Reshape grad back to spatial — works for both grey (784) and RGB (2352)
    n_el = len(grad)
    if n_el == 784:
        return _norm01(np.abs(grad).reshape(28, 28))
    else:
        # RGB: average gradient magnitude across channels → (28,28)
        n_ch_sal = n_el // 784
        return _norm01(np.abs(grad).reshape(n_ch_sal, 28, 28).mean(axis=0))


def compute_gemex(exp, x_input, x_ref):
    """Run GEMEX and return (gsf_scores, result_object)."""
    r     = exp.explain(x_input, X_reference=x_ref)
    g_map = r.gsf_scores   # (784,)
    return g_map, r


def _norm01(v):
    a = np.abs(v) if v.ndim > 1 else np.abs(v.flatten())
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)


def _to_28x28(heatmap_flat, target_h=28, target_w=28):
    """
    Convert a flat heatmap to a 28×28 display map.
    Handles both:
      - 784 pixels  → reshape directly to (28,28)
      - 49 patches  → bilinear upsample from (7,7) to (28,28)
      - any n_patch → upsample from (sqrt(n),sqrt(n)) to (28,28)
    """
    n = len(heatmap_flat)
    if n == target_h * target_w:
        # Already pixel-level
        return heatmap_flat.reshape(target_h, target_w)
    # Patch-level: infer grid size and upsample
    side = int(np.round(np.sqrt(n)))
    patch_grid = heatmap_flat.reshape(side, side)
    scale_h = target_h / side
    scale_w = target_w / side
    return zoom(patch_grid, (scale_h, scale_w), order=1)


def _smooth(heatmap, sigma=1.5):
    """Gaussian smooth a 2D heatmap for visual quality."""
    return gaussian_filter(heatmap.reshape(28, 28), sigma=sigma)

# ═══════════════════════════════════════════════════════════════════════
# DELETION AUC (quantitative faithfulness for image XAI)
# ═══════════════════════════════════════════════════════════════════════

def deletion_auc(model, x_input, heatmap_flat, target_class,
                 n_steps=10):
    """
    Deletion AUC (Samek et al., 2017).
    Progressively delete (zero out) pixels in order of attribution magnitude.
    AUC of prediction score vs fraction deleted.
    Lower = more faithful (important pixels identified correctly).
    """
    order  = np.argsort(heatmap_flat)[::-1]
    n_pix  = len(order)
    scores = []
    x_mod  = x_input.copy()
    for step in range(n_steps + 1):
        k = int(step / n_steps * n_pix)
        if k > 0:
            x_mod[order[:k]] = 0.0
        p = model.predict_proba(x_mod.reshape(1, -1))[0, target_class]
        scores.append(float(p))
    return float(np.trapz(scores, dx=1.0/n_steps))

# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(x_img_grey, x_img_raw,
                    g_map, c_map, cpp_map, s_map,
                    gemex_result, class_name, true_cls, pred_cls,
                    img_idx, dataset_name, save_path, t):
    """
    Five-panel figure with improved overlay blending.

    Col 1 : Original image (greyscale for consistency)
    Col 2 : GEMEX GeodesicCAM (Gaussian-smoothed, corrected blend)
    Col 3 : GradCAM
    Col 4 : GradCAM++
    Col 5 : Vanilla Saliency
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.patch.set_facecolor(t['bg'])
    for ax in axes:
        ax.set_facecolor(t['bg'])
        ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.84,
                        bottom=0.02, wspace=0.06)

    # ── Panel 1: Original ────────────────────────────────────────────
    axes[0].imshow(x_img_grey, cmap='gray', vmin=0, vmax=1,
                   interpolation='nearest')
    axes[0].set_title(f'Original\n{class_name}',
                      fontsize=10, color=t['text'], fontweight='bold', pad=6)

    # ── Improved overlay helper ───────────────────────────────────────
    def overlay(ax, heatmap_2d, cmap, title, col):
        """
        Fixed overlay: image alpha=0.50, heatmap alpha=0.65.
        Gaussian-smoothed heatmap for visual quality.
        """
        sm = _smooth(heatmap_2d, sigma=1.5)
        ax.imshow(x_img_grey, cmap='gray', vmin=0, vmax=1,
                  alpha=0.50, interpolation='bilinear')
        im = ax.imshow(sm, cmap=cmap, vmin=0, vmax=1,
                       alpha=0.65, interpolation='bilinear')
        ax.set_title(title, fontsize=10, color=col,
                     fontweight='bold', pad=6)
        # Thin colourbar
        pos  = ax.get_position()
        cax  = fig.add_axes([pos.x1 - 0.015, pos.y0 + 0.05,
                              0.008, pos.height * 0.6])
        cb   = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, colors=t['text2'])
        cb.outline.set_edgecolor(t['border'])

    overlay(axes[1], g_map,   'inferno', 'GEMEX\n(GeodesicCAM)', t['gemex'])
    overlay(axes[2], c_map,   'jet',     'GradCAM',              t['cam'])
    overlay(axes[3], cpp_map, 'plasma',  'GradCAM++',            t['cam2'])
    overlay(axes[4], s_map,   'hot',     'Saliency',             t['sal'])

    # ── Footer info ───────────────────────────────────────────────────
    match_str  = '✓ correct' if true_cls == pred_cls else f'✗ pred={pred_cls}'
    ricci_str  = (f'Ricci={gemex_result.manifold_curvature:.3f}  '
                  f'FIM={gemex_result.fim_quality}'
                  if gemex_result else 'GEMEX unavailable')
    fig.text(0.01, 0.01, f'{dataset_name}  ·  Image #{img_idx}  '
             f'[{match_str}]  ·  {ricci_str}',
             fontsize=8.5, color=t['text2'])

    fig.suptitle(f'{dataset_name}  ·  GEMEX vs GradCAM vs GradCAM++ vs Saliency  '
                 f'·  Class: {class_name}',
                 fontsize=11, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()


def plot_quantitative(records, dataset_name, save_path, t):
    """
    Four-panel quantitative summary:
      P1: Rank correlation GEMEX vs GradCAM / GradCAM++
      P2: Top-10% pixel precision
      P3: Deletion AUC (lower = better)
      P4: Ricci scalar distribution (GEMEX-only)
    """
    methods = ['GEMEX', 'GradCAM', 'GradCAM++', 'Saliency']
    cols    = [t['gemex'], t['cam'], t['cam2'], t['sal']]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(t['bg'])
    for ax in axes:
        ax.set_facecolor(t['panel'])
        for sp in ax.spines.values():
            sp.set_color(t['border']); sp.set_linewidth(0.7)
        ax.tick_params(colors=t['text2'], labelsize=9)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.86,
                        bottom=0.12, wspace=0.32)

    # P1: Rank correlation with GradCAM++ as reference
    # GEMEX stores patch-level scores (49) — upsample to 784 for comparison
    rk = {'GEMEX':[], 'GradCAM':[], 'GradCAM++':[], 'Saliency':[]}
    for rec in records:
        ref = rec['gradcam_pp'].flatten()   # always 784
        for m, key in [('GEMEX','gemex'),('GradCAM','gradcam'),
                       ('GradCAM++','gradcam_pp'),('Saliency','saliency')]:
            hm = _to_28x28(rec[key].flatten()).flatten()  # normalise to 784
            c, _ = spearmanr(hm, ref)
            rk[m].append(float(c) if not np.isnan(c) else 0.0)
    _bar(axes[0], methods, [np.mean(rk[m]) for m in methods],
         [np.std(rk[m]) for m in methods], cols, t,
         'Rank Correlation\n(vs GradCAM++ reference)', 'Spearman r', (0, 1.2))

    # P2: Top-10% pixel precision
    tp = {m: [] for m in methods}
    for rec in records:
        fg = rec['image_grey'].flatten() > 0.15
        k  = max(int(784 * 0.10), 10)  # always 784 pixels after _to_28x28
        for m, key in [('GEMEX','gemex'),('GradCAM','gradcam'),
                       ('GradCAM++','gradcam_pp'),('Saliency','saliency')]:
            hm   = _to_28x28(rec[key].flatten()).flatten()  # always 784
            prec = fg[np.argsort(hm)[::-1][:k]].mean()
            tp[m].append(float(prec))
    _bar(axes[1], methods, [np.mean(tp[m]) for m in methods],
         [np.std(tp[m]) for m in methods], cols, t,
         'Top-10% Pixel Precision\n(fraction in foreground)', 'Precision', (0, 1.2))

    # P3: Deletion AUC (lower = better)
    da = {m: [] for m in methods}
    for rec in records:
        for m, key in [('GEMEX','gemex_del'),('GradCAM','cam_del'),
                       ('GradCAM++','cpp_del'),('Saliency','sal_del')]:
            if key in rec: da[m].append(rec[key])
    if any(da[m] for m in methods):
        _bar(axes[2], methods, [np.mean(da[m]) if da[m] else 0 for m in methods],
             [np.std(da[m])  if da[m] else 0 for m in methods], cols, t,
             'Deletion AUC ↓\n(lower = more faithful)', 'AUC', None,
             lower_is_better=True)

    # P4: Ricci scalar distribution
    riccis = [rec['ricci'] for rec in records if rec['ricci'] > 0]
    if riccis:
        axes[3].hist(riccis, bins=min(10, len(riccis)),
                     color=t['gemex'], alpha=0.80, edgecolor=t['bg'])
        axes[3].axvline(np.mean(riccis), color=t['cam'], lw=1.8,
                        label=f'Mean={np.mean(riccis):.3f}')
        axes[3].set_title('Manifold Curvature (Ricci)\n[GEMEX-exclusive]',
                          fontsize=10, color=t['text'], fontweight='bold')
        axes[3].set_xlabel('Ricci scalar', fontsize=9, color=t['text2'])
        axes[3].legend(fontsize=8.5, framealpha=0.4,
                       facecolor=t['panel'], edgecolor=t['border'],
                       labelcolor=t['text'])
        axes[3].grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)

    fig.suptitle(f'{dataset_name}  ·  Quantitative Comparison',
                 fontsize=12, color=t['text'], fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f"  -> {save_path}")


def _bar(ax, methods, means, stds, cols, t, title, ylabel, ylim,
         lower_is_better=False):
    x    = np.arange(len(methods))
    bars = ax.bar(x, means, color=cols, alpha=0.82,
                  edgecolor='none', width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt='none',
                ecolor=t['text3'], elinewidth=1.2, capsize=3)
    for xi, v in enumerate(means):
        ax.text(xi, v + (max(means) if means else 1)*0.04,
                f'{v:.3f}', ha='center', fontsize=9,
                color=t['text'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right',
                       fontsize=9, color=t['text'])
    ax.set_title(title, fontsize=10, color=t['text'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, color=t['text2'])
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis='y', color=t['grid'], lw=0.5, alpha=0.6)
    if lower_is_better:
        best = min(range(len(means)), key=lambda i: means[i])
        bars[best].set_edgecolor(t['gemex'])
        bars[best].set_linewidth(2.0)

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_dataset(dataset_key, save_dir, n_images, theme, backend, epochs):
    # Ensure output directory exists before any plt.savefig call
    os.makedirs(save_dir, exist_ok=True)
    t = DARK if theme == 'dark' else LIGHT

    # ── Load data ─────────────────────────────────────────────────────
    (X_tr, y_tr, X_te, y_te,
     X_tr_raw, X_te_raw,
     n_ch, classes, ds_name, task) = load_medmnist(dataset_key)

    n_cls = len(classes)
    print(f"  Dataset : {ds_name}")
    print(f"  Task    : {task}")
    print(f"  Classes : {n_cls}   Channels: {n_ch}")

    # ── Scale for sklearn MLP ─────────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # ── Train model ───────────────────────────────────────────────────
    cnn_net = None; device = 'cpu'
    if backend == 'cnn':
        try:
            import torch
            model, cnn_net, device = train_medcnn(
                X_tr_raw, y_tr, n_cls, n_ch, epochs=epochs)
        except ImportError:
            print("  PyTorch not found — using sklearn MLP")
            backend = 'mlp'

    if backend == 'mlp':
        model = train_mlp_sklearn(X_tr_s, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────
    # CNN needs spatial input X_te_raw (N,H,W,C); MLP needs scaled flat X_te_s
    x_eval  = X_te_s if backend == 'mlp' else X_te_raw
    preds   = model.predict(x_eval)
    acc     = np.mean(preds == y_te)
    print(f"  Test accuracy: {acc:.4f}")

    # ── GEMEX setup ───────────────────────────────────────────────────
    # Patch-based GEMEX for images
    # image_patch_size=4 → 28×28 pixels become 7×7=49 patch features
    # Benefits: stronger FIM, higher Ricci scalar, ~10× faster than pixel-level
    # image_patch_size=1 → original pixel-level behaviour
    cfg = GemexConfig(
        n_geodesic_steps    = 10,   # more steps now feasible with 49 features
        n_reference_samples = 20,   # restored — feasible with 49 features
        interaction_order   = 2,    # PTI now meaningful with 49 patch features
        fim_local_n         = 8,    # restored — feasible with 49 features
        fim_local_sigma     = 0.10,
        image_patch_size    = 4,    # 28×28 → 7×7=49 patch features
        verbose             = False)
    # feature_names must match the patch count, not pixel count
    # image_patch_size=4 on 28×28 → 7×7=49 patches
    n_patches = (28 // cfg.image_patch_size) ** 2
    feat = [f'patch_r{i//int(28//cfg.image_patch_size)}_c{i%int(28//cfg.image_patch_size)}'
            for i in range(n_patches)]
    # For CNN + RGB: GEMEX needs a predict_fn that accepts flat grey (784,)
    # patches and internally converts to RGB for the CNN.
    # Strategy: GEMEX works on greyscale patches; CNN receives greyscale
    # replicated to 3 channels (acceptable for attribution purposes).
    if backend == 'cnn' and n_ch > 1:
        def _grey_to_rgb_predict(X_flat):
            # X_flat: (n, 784) greyscale → replicate to (n,3,28,28) for CNN
            X2d = X_flat.reshape(-1, 28, 28).astype(np.float32)
            X_rgb = np.stack([X2d, X2d, X2d], axis=1)  # (n,3,28,28)
            import torch
            with torch.no_grad():
                import torch.nn.functional as F
                t = torch.from_numpy(X_rgb).to(model.device)
                return F.softmax(model.net(t), dim=1).cpu().numpy()
        gemex_predict_fn = _grey_to_rgb_predict
    else:
        gemex_predict_fn = model.predict_proba
    exp  = Explainer(gemex_predict_fn,
                     data_type     = 'image',
                     feature_names = feat,
                     class_names   = classes,
                     config        = cfg)

    # ── Select one image per class (correctly predicted) ──────────────
    selected = {}
    for i, (yi, pi) in enumerate(zip(y_te, preds)):
        if yi not in selected and yi == pi:
            selected[yi] = i
        if len(selected) >= n_cls:
            break
    # Fallback: take any image for missing classes
    for cls in range(n_cls):
        if cls not in selected:
            idxs = np.where(y_te == cls)[0]
            if len(idxs): selected[cls] = idxs[0]

    # ── Explain each selected image ───────────────────────────────────
    records = []
    n_done  = 0
    # x_ref must be in same pixel space as x_inp
    # MLP: scaled (X_tr_s) · CNN: [0,1] normalised (X_tr)
    x_ref   = X_tr_s[:50] if backend == 'mlp' else X_tr[:50]
    # Safety: ensure x_ref is 2D flat (n_samples, 784)
    if x_ref.ndim > 2:
        x_ref = x_ref.reshape(len(x_ref), -1)

    for cls in sorted(selected.keys()):
        if n_done >= n_images:
            break

        idx       = selected[cls]
        x_raw_i   = X_te[idx]             # flat [0,1]
        x_raw_img = X_te_raw[idx]         # spatial (H,W,C) or (H,W)
        x_inp     = X_te_s[idx] if backend == 'mlp' else X_te[idx]
        # x_cnn: what the CNN model actually expects
        # For RGB CNN: spatial (28,28,3) — _to_tensor transposes to (1,3,28,28)
        # For MLP: same as x_inp (flat scaled)
        x_cnn     = x_inp if backend == 'mlp' else x_raw_img

        # Greyscale for display
        if x_raw_img.ndim == 3 and x_raw_img.shape[-1] == 3:
            x_grey = (0.299*x_raw_img[:,:,0] + 0.587*x_raw_img[:,:,1] +
                      0.114*x_raw_img[:,:,2])
        elif x_raw_img.ndim == 3:
            x_grey = x_raw_img[:,:,0]
        else:
            x_grey = x_raw_img.reshape(28, 28)

        pred_cls   = int(model.predict(x_cnn.reshape(1,-1) if x_cnn.ndim == 1 else x_cnn[np.newaxis])[0])
        class_name = classes[cls] if cls < len(classes) else str(cls)
        print(f"\n  [{ds_name}] Image #{idx}  "
              f"true={class_name}  pred={classes[pred_cls]}")

        # GEMEX
        try:
            g_flat, r = compute_gemex(exp, x_inp, x_ref)
            # g_flat may be 49 patches or 784 pixels depending on config
            print(f"    GEMEX  fim={r.fim_quality}  "
                  f"ricci={r.manifold_curvature:.4f}  "
                  f"gsf_shape={g_flat.shape}")
        except Exception as e:
            print(f"    GEMEX failed: {e}")
            g_flat = np.zeros(n_patches)
            r = None

        # GradCAM / GradCAM++ / Saliency
        if cnn_net is not None:
            try:
                # Ensure correct shape for CNN input
                # Use spatial x_raw_img for CNN — it has all channels
                x_flat_np = x_raw_img.flatten()  # (2352,) for RGB, (784,) for grey
                if n_ch > 1:
                    # RGB: (H*W*C,) → (1,C,H,W) via transpose
                    x_cnn = x_raw_img.transpose(2,0,1)[np.newaxis].astype(np.float32)
                else:
                    # Greyscale: (H,W,1) → (1,1,H,W)
                    x_cnn = x_raw_img[:,:,0][np.newaxis,np.newaxis].astype(np.float32).reshape(1, 1, 28, 28).astype(np.float32)
                c_map   = compute_gradcam(cnn_net, x_cnn, pred_cls, device, n_ch)
                cpp_map = compute_gradcam_pp(cnn_net, x_cnn, pred_cls, device, n_ch)
            except Exception as eg:
                print(f'    GradCAM failed: {eg} — using saliency fallback')
                c_map   = compute_saliency(model, x_cnn.flatten(), pred_cls).flatten()
                cpp_map = c_map.copy()
        else:
            c_map   = compute_saliency(model, x_cnn.flatten(), pred_cls).flatten()
            cpp_map = c_map.copy()
        s_map = compute_saliency(model,
                    x_cnn.reshape(1,-1).flatten() if x_cnn.ndim > 1
                    else x_cnn, pred_cls).flatten()

        g_n   = _norm01(g_flat)
        c_n   = _norm01(c_map.flatten())
        cpp_n = _norm01(cpp_map.flatten())
        s_n   = _norm01(s_map.flatten())

        # Deletion AUC
        # For GEMEX patch scores: upsample to pixel space first
        g_n_px  = _to_28x28(g_n).flatten()   # always 784 for deletion
        g_del   = deletion_auc(model, x_cnn.flatten(), g_n_px, pred_cls)
        c_del   = deletion_auc(model, x_cnn.flatten(), c_n, pred_cls)
        cpp_del = deletion_auc(model, x_cnn.flatten(), cpp_n, pred_cls)
        s_del   = deletion_auc(model, x_cnn.flatten(), s_n, pred_cls)

        # Per-image plot
        tag  = ds_name.lower().replace(' ', '_')
        path = f"{save_dir}/{tag}_class{cls}_{theme}.png"
        plot_comparison(
            x_grey, x_raw_img,
            _to_28x28(g_n), c_n.reshape(28,28),
            cpp_n.reshape(28,28), s_n.reshape(28,28),
            r, class_name, cls, pred_cls, idx, ds_name, path, t)
        print(f"    -> {path}")

        records.append({
            'class': cls, 'pred': pred_cls, 'class_name': class_name,
            'image_grey': x_grey, 'image_raw': x_raw_img,
            'gemex': g_n, 'gradcam': c_n, 'gradcam_pp': cpp_n, 'saliency': s_n,
            'ricci': abs(r.manifold_curvature) if r else 0.0,
            'fim_quality': r.fim_quality if r else 'n/a',
            'gemex_del': g_del, 'cam_del': c_del,
            'cpp_del': cpp_del, 'sal_del': s_del,
        })
        n_done += 1

    # ── Quantitative summary ──────────────────────────────────────────
    if records:
        tag  = ds_name.lower().replace(' ', '_')
        plot_quantitative(records, ds_name,
                          f"{save_dir}/{tag}_summary_{theme}.png", t)

    # ── CSV ───────────────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for rec in records:
        fg = rec['image_grey'].flatten() > 0.15
        k  = max(int(784 * 0.10), 10)  # always 784 pixels after _to_28x28
        def tp(key):
            hm = _to_28x28(rec[key].flatten()).flatten()
            return fg[np.argsort(hm)[::-1][:k]].mean()
        def rk(k1, k2):
            h1 = _to_28x28(rec[k1].flatten()).flatten()
            h2 = _to_28x28(rec[k2].flatten()).flatten()
            c, _ = spearmanr(h1, h2)
            return round(float(c) if not np.isnan(c) else 0.0, 4)
        rows.append({
            'dataset'           : ds_name,
            'class'             : rec['class'],
            'class_name'        : rec['class_name'],
            'predicted'         : rec['pred'],
            'ricci_scalar'      : round(rec['ricci'], 4),
            'fim_quality'       : rec['fim_quality'],
            'rk_gemex_gradcam'  : rk('gemex', 'gradcam'),
            'rk_gemex_gradcampp': rk('gemex', 'gradcam_pp'),
            'tp_gemex'          : round(float(tp('gemex')), 4),
            'tp_gradcam'        : round(float(tp('gradcam')), 4),
            'tp_gradcampp'      : round(float(tp('gradcam_pp')), 4),
            'tp_saliency'       : round(float(tp('saliency')), 4),
            'del_auc_gemex'     : round(rec.get('gemex_del', 0.0), 4),
            'del_auc_gradcam'   : round(rec.get('cam_del', 0.0), 4),
            'del_auc_gradcampp' : round(rec.get('cpp_del', 0.0), 4),
            'del_auc_saliency'  : round(rec.get('sal_del', 0.0), 4),
        })
    df = pd.DataFrame(rows)
    tag = ds_name.lower().replace(' ', '_')
    csv_path = f"{save_dir}/{tag}_results_{theme}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")
    print("\n  Results summary:")
    pd.set_option('display.max_columns', None)
    print(df[['class_name','ricci_scalar','rk_gemex_gradcampp',
              'tp_gemex','tp_gradcampp','del_auc_gemex',
              'del_auc_gradcampp']].to_string(index=False))
    return df

# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT — BloodMNIST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GEMEX vs GradCAM — BloodMNIST')
    parser.add_argument('--save-dir', default='./blood_results',
                        help='Output directory (default: ./blood_results)')
    parser.add_argument('--n-images', type=int, default=8,
                        help='Number of images to explain (default: 8)')
    parser.add_argument('--epochs',   type=int, default=20,
                        help='CNN training epochs (default: 20)')
    parser.add_argument('--theme',    default='dark',
                        choices=['dark', 'light'])
    parser.add_argument('--backend',  default='cnn',
                        choices=['mlp', 'cnn'],
                        help='cnn=MedCNN with true GradCAM (recommended); '
                             'mlp=sklearn MLP (no PyTorch needed)')
    args = parser.parse_args()

    print("\n" + "="*62)
    print("  GEMEX vs GradCAM vs GradCAM++ vs Saliency")
    print("  Dataset  : BloodMNIST")
    print("  Task     : blood cell microscopy classification (8 cell types)")
    print("="*62)
    print(f"  backend  : {args.backend.upper()}")
    print(f"  n_images : {args.n_images}")
    print(f"  epochs   : {args.epochs}")
    print(f"  save_dir : {args.save_dir}")
    print(f"  theme    : {args.theme}\n")

    df = run_dataset(
        dataset_key = 'blood',
        save_dir    = args.save_dir,
        n_images    = args.n_images,
        theme       = args.theme,
        backend     = args.backend,
        epochs      = args.epochs)

    print(f"\n  All outputs saved to: {os.path.abspath(args.save_dir)}/")
