# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""gemex.core.config — Central configuration dataclass."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GemexConfig:
    # ── Geodesic integration ─────────────────────────────────────────
    n_geodesic_steps:    int   = 40
    alpha_connection:    float = 0.0     # Amari α ∈ [-1,1]; 0 = Levi-Civita
    n_reference_samples: int   = 80

    # ── FIM estimation ───────────────────────────────────────────────
    fim_epsilon:      float = 1e-3   # base finite-difference step
    fim_epsilon_auto: bool  = True   # True → auto-detect step size per instance
    #   When fim_epsilon_auto=True the FIM tries [fim_epsilon, 5×, 10×, 50×,
    #   100×] until a nonzero gradient is found.  This is essential for
    #   piecewise-constant models (tree ensembles) where the default 1e-3
    #   falls inside a single leaf and finds zero gradient.
    fim_local_avg:    bool  = True   # True → average FIM over a small
    #   neighbourhood of x (radius = fim_local_sigma) in addition to the
    #   kernel-smoothed reference averaging.  Captures boundary geometry
    #   even when x sits deep inside a tree leaf.
    fim_local_sigma:  float = 0.10   # radius for local neighbourhood averaging
    fim_local_n:      int   = 16     # number of neighbourhood perturbations

    # model_type hint — set automatically by Explainer; can be overridden.
    #   'auto'   → detect from model class name
    #   'tree'   → GBM, RandomForest, XGBoost, LightGBM (piecewise-constant)
    #   'smooth' → neural networks, logistic regression, SVM-RBF
    model_type: str = 'auto'

    # ── Interaction ──────────────────────────────────────────────────
    interaction_order: int = 2       # 1=none, 2=pairwise PTI, 3=triplet RCT
    rct_top_k:         int = 10      # top-k features for triplet RCT

    # ── RST decomposition ────────────────────────────────────────────
    rst_n_components: int = 5

    # ── Approximation ────────────────────────────────────────────────
    use_lanczos:       bool = False
    lanczos_threshold: int  = 100

    # ── Image patch support ──────────────────────────────────────────
    # When data_type='image', pixels are aggregated into patches before
    # explanation.  patch_size=1 → current pixel-level behaviour (default).
    # patch_size=4 on 28×28 → 7×7=49 patch features: faster FIM estimation,
    # stronger Ricci scalar, comparable resolution to GradCAM.
    # Tabular and timeseries data are completely unaffected by this setting.
    image_patch_size: int = 1     # 1=pixel-level (default), 4=7×7 patches

    # ── GSF normalisation ────────────────────────────────────────────
    gsf_normalise: bool = False   # True → sum(GSF) = f(x) − f(baseline)

    # ── Output ───────────────────────────────────────────────────────
    language_detail: str  = 'plain'   # 'plain' | 'technical'
    random_state:    int  = 42
    verbose:         bool = False

    def validate(self):
        assert self.n_geodesic_steps >= 5
        assert -1.0 <= self.alpha_connection <= 1.0
        assert self.interaction_order in (1, 2, 3)
        assert self.language_detail in ('plain', 'technical')
        assert self.model_type in ('auto', 'tree', 'smooth')
        return self
