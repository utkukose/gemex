# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
gemex.manifold.fim
------------------
Fisher Information Matrix — the Riemannian metric tensor on the
statistical manifold induced by the model.

Formal definition
-----------------
    G_ij(x) = Σ_y  p(y|x) · (∂ log p(y|x)/∂x_i)(∂ log p(y|x)/∂x_j)

Estimated via central finite differences — no gradient access required.

Design for piecewise-constant models (tree ensembles)
------------------------------------------------------
Standard neural-network FIM estimation uses a small epsilon (≈1e-3)
because smooth models have nonzero gradients everywhere.  Tree ensembles
(GBM, RandomForest, XGBoost, LightGBM) are piecewise-constant: inside
any single leaf the probability is exactly constant, so a small epsilon
finds zero gradient and produces a zero FIM.

Three mechanisms address this:

Correction 1 for version 1.2.0 — Adaptive epsilon
    For each instance, the FIM tests increasing epsilon values
    [base, 5×, 10×, 50×, 100×] until a nonzero gradient is found.
    Once the active epsilon is discovered it is cached per-instance.
    For smooth models the base epsilon (1e-3) is always sufficient
    and no extra evaluations are needed.

Correction 2 for version 1.2.0 — Local neighbourhood averaging
    Even with a larger epsilon, a single perturbation step may still
    land inside the same leaf.  We average the FIM over a small sphere
    of random perturbations around x (radius = fim_local_sigma).
    This captures the geometry of nearby decision boundaries even when
    x itself is deep inside a leaf.

Correction 3 for version 1.2.0 — Model-type routing (via config.model_type)
    When model_type='tree' the default sigma is increased to 0.10
    and the search starts at 5× the base epsilon, avoiding wasted
    evaluations at epsilon values that are known to fail on trees.

These corrections are transparent: they do not change the mathematical
definition of the FIM, only its numerical estimation strategy.

Properties guaranteed by construction
--------------------------------------
- Symmetric:   G = Gᵀ
- PSD:         G ⪰ 0  (enforced by eigenvalue clipping)
- Invariant:   under sufficient statistics (Chentsov, 1982)

References
----------
Rao (1945); Amari & Nagaoka (2000); Nielsen (2020);
Silverman (1986) Density Estimation.
"""

import numpy as np
from typing import Callable, Optional


class FisherInformationMatrix:

    def __init__(self, predict_fn: Callable, config):
        self.predict_fn  = predict_fn
        self.config      = config
        self._cache_x    = None
        self._cache_G    = None
        self._active_eps = None   # cached adaptive epsilon for current instance
        # LRU cache for metric_at() during geodesic integration
        self._lru_keys:   list = []
        self._lru_values: list = []
        self._lru_size         = 32

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self, x: np.ndarray,
                X_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the FIM at x using all three corrections.

        Steps
        -----
        1. Find the active epsilon (Correction 1).
        2. Compute local neighbourhood FIM around x (Correction 2).
        3. Kernel-smooth over reference points if X_ref provided.
        4. Merge local and reference estimates.
        5. Symmetrise and enforce PSD.
        """
        eps = self._find_active_epsilon(x)

        # Local neighbourhood FIM (Correction 2)
        G_local = self._local_neighbourhood_fim(x, eps)

        # Reference kernel-smoothed FIM (existing Fix 2)
        if X_ref is not None and len(X_ref) > 1:
            G_ref = self._kernel_smoothed_fim(x, X_ref, eps)
            # Weighted merge: local geometry 60%, reference context 40%
            G = 0.60 * G_local + 0.40 * G_ref
        else:
            G = G_local

        G = self._make_psd(G)
        self._cache_x = x.copy()
        self._cache_G = G.copy()
        return G

    def metric_at(self, x: np.ndarray) -> np.ndarray:
        """Return FIM at x; use LRU cache when possible."""
        if (self._cache_x is not None
                and np.allclose(x, self._cache_x, atol=1e-8)):
            return self._cache_G
        for key, val in zip(self._lru_keys, self._lru_values):
            if np.allclose(x, key, atol=1e-8):
                return val
        eps = self._active_eps if self._active_eps else self.config.fim_epsilon
        G   = self._make_psd(self._point_fim(x, eps))
        self._lru_push(x, G)
        return G

    def inner_product(self, x: np.ndarray,
                      u: np.ndarray, v: np.ndarray) -> float:
        return float(u @ self.metric_at(x) @ v)

    def norm(self, x: np.ndarray, v: np.ndarray) -> float:
        return float(np.sqrt(max(self.inner_product(x, v, v), 0.0)))

    def ricci_scalar(self, G: np.ndarray) -> float:
        """
        Approximate scalar Ricci curvature from FIM eigenspectrum.
        Returns 0.0 if FIM is degenerate (zero matrix).
        """
        ev  = np.linalg.eigvalsh(G)
        if np.max(ev) < 1e-9:
            return 0.0   # degenerate — return 0 rather than NaN
        ev  = np.maximum(ev, 1e-12)
        log = np.log(ev)
        return float(np.std(log) / (np.abs(np.mean(log)) + 1e-8))

    def geodesic_distance_approx(self, x1: np.ndarray,
                                  x2: np.ndarray) -> float:
        G = self.metric_at(0.5 * (x1 + x2))
        d = x2 - x1
        return float(np.sqrt(max(d @ G @ d, 0.0)))

    def fim_quality(self) -> str:
        """
        Return a human-readable quality label for the current FIM.
        'good'     — trace > 1e-4,  healthy curvature
        'marginal' — trace > 1e-8,  epsilon was increased to find gradient
        'poor'     — trace ≤ 1e-8,  FIM is essentially zero
        """
        if self._cache_G is None:
            return 'unknown'
        tr = np.trace(self._cache_G)
        if tr > 1e-4:
            return 'good'
        elif tr > 1e-8:
            return 'marginal'
        else:
            return 'poor'

    # ------------------------------------------------------------------ #
    #  Correction 1: Adaptive epsilon                                      #
    # ------------------------------------------------------------------ #

    def _find_active_epsilon(self, x: np.ndarray) -> float:
        """
        Find the smallest epsilon that produces a nonzero gradient at x.

        Search order depends on model_type:
          smooth → [base, 5×, 10×, 50×, 100×]
          tree   → [10×, 50×, 100×, 200×, base]  (skip small values)
          auto   → detect from predict_fn behaviour, then choose

        The found epsilon is cached as self._active_eps for reuse in
        metric_at() calls during the geodesic integration.
        """
        base = self.config.fim_epsilon
        mt   = self._resolve_model_type()

        if mt == 'tree':
            candidates = [base * 10, base * 50, base * 100,
                          base * 200, base * 5, base]
        else:
            candidates = [base, base * 5, base * 10,
                          base * 50, base * 100]

        if not self.config.fim_epsilon_auto:
            self._active_eps = base
            return base

        d = len(x)
        n_test = min(d, 5)   # test first 5 features for speed
        for eps in candidates:
            for i in range(n_test):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                pp = self.predict_fn(xp.reshape(1, -1))[0]
                pm = self.predict_fn(xm.reshape(1, -1))[0]
                if np.any(np.abs(pp - pm) > 1e-8):
                    self._active_eps = eps
                    if self.config.verbose and eps > base:
                        print(f"[GEMEX FIM] Adaptive epsilon: {eps:.2e} "
                              f"(base {base:.2e} found zero gradient)")
                    return eps

        # Fallback: use largest candidate
        self._active_eps = candidates[-1]
        return self._active_eps

    # ------------------------------------------------------------------ #
    #  Correction 2 for version 1.2.0: Local neighbourhood averaging       #
    # ------------------------------------------------------------------ #

    def _local_neighbourhood_fim(self, x: np.ndarray,
                                  eps: float) -> np.ndarray:
        """
        Average FIM over a small random neighbourhood around x.

        Samples fim_local_n points from a Gaussian sphere of radius
        fim_local_sigma around x.  This reveals decision-boundary
        geometry even when x is deep inside a tree leaf.
        """
        mt    = self._resolve_model_type()
        sigma = self.config.fim_local_sigma
        # For tree models use a larger neighbourhood
        if mt == 'tree':
            sigma = max(sigma, 0.10)

        n     = self.config.fim_local_n
        rng   = np.random.RandomState(self.config.random_state)
        d     = len(x)

        # Always include the exact point x
        pts   = [x]
        perturbs = rng.randn(n, d)
        norms = np.linalg.norm(perturbs, axis=1, keepdims=True) + 1e-10
        perturbs = perturbs / norms * sigma   # uniform direction, fixed radius
        for i in range(n):
            pts.append(x + perturbs[i])

        Gs = [self._point_fim(pt, eps) for pt in pts]
        # Weight x itself more heavily (centre of neighbourhood)
        weights = np.ones(len(Gs))
        weights[0] = 2.0
        weights /= weights.sum()
        G = np.einsum('i,ijk->jk', weights, np.array(Gs))
        return G

    # ------------------------------------------------------------------ #
    #  Kernel-smoothed reference FIM (existing Fix 2, updated)             #
    # ------------------------------------------------------------------ #

    def _kernel_smoothed_fim(self, x: np.ndarray,
                              X_ref: np.ndarray,
                              eps: float) -> np.ndarray:
        """
        Gaussian kernel-weighted average of point FIMs over X_ref,
        using the adaptive epsilon rather than the config default.
        """
        n_ref  = min(len(X_ref), self.config.n_reference_samples)
        dists  = np.linalg.norm(X_ref - x, axis=1)
        order  = np.argsort(dists)[:n_ref]
        X_sub  = X_ref[order]
        d_sub  = dists[order]

        d_feat = X_sub.shape[1]
        sigma  = np.mean(np.std(X_sub, axis=0)) + 1e-8
        h      = sigma * (n_ref ** (-1.0 / (d_feat + 4))) + 1e-8

        w      = np.exp(-0.5 * (d_sub / h) ** 2)
        w_sum  = w.sum() + 1e-12

        Gs = np.array([self._point_fim(X_sub[i], eps) for i in range(n_ref)])
        return np.einsum('i,ijk->jk', w / w_sum, Gs)

    # ------------------------------------------------------------------ #
    #  Correction 3 for version 1.2.0: Model-type routing                  #
    # ------------------------------------------------------------------ #

    def _resolve_model_type(self) -> str:
        """
        Resolve 'auto' model type by inspecting the predict_fn.
        Result is cached on the config object after first call.
        """
        mt = self.config.model_type
        if mt != 'auto':
            return mt
        # Inspect the class name of the underlying model
        fn = self.predict_fn
        # Unwrap bound methods to get the class
        obj = getattr(fn, '__self__', None)
        cls = type(obj).__name__.lower() if obj else ''
        tree_keywords = ['forest', 'boost', 'tree', 'xgb', 'lgbm',
                         'catboost', 'gradient', 'extra']
        if any(k in cls for k in tree_keywords):
            self.config.model_type = 'tree'
            return 'tree'
        self.config.model_type = 'smooth'
        return 'smooth'

    # ------------------------------------------------------------------ #
    #  Core point FIM (updated to accept epsilon argument)                 #
    # ------------------------------------------------------------------ #

    def _point_fim(self, x: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        """Compute FIM at a single point x using central finite differences."""
        if eps is None:
            eps = self.config.fim_epsilon
        d       = len(x)
        K       = self._n_classes(x)
        log_jac = np.zeros((d, K))
        for i in range(d):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            pp = np.clip(self.predict_fn(xp.reshape(1, -1))[0], 1e-12, 1.)
            pm = np.clip(self.predict_fn(xm.reshape(1, -1))[0], 1e-12, 1.)
            log_jac[i] = (np.log(pp) - np.log(pm)) / (2 * eps)
        p0 = np.clip(self.predict_fn(x.reshape(1, -1))[0], 1e-12, 1.)
        return log_jac @ np.diag(p0) @ log_jac.T

    def _make_psd(self, G: np.ndarray) -> np.ndarray:
        G = 0.5 * (G + G.T)
        ev, evec = np.linalg.eigh(G)
        ev = np.maximum(ev, 1e-10)
        return evec @ np.diag(ev) @ evec.T

    def _n_classes(self, x: np.ndarray) -> int:
        return self.predict_fn(x.reshape(1, -1)).shape[1]

    def _lru_push(self, x: np.ndarray, G: np.ndarray):
        if len(self._lru_keys) >= self._lru_size:
            self._lru_keys.pop(0)
            self._lru_values.pop(0)
        self._lru_keys.append(x.copy())
        self._lru_values.append(G.copy())
