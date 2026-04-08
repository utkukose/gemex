# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
gemex.core.explainer — Model-agnostic GEMEX Explainer.

Queries the model ONLY via predict_proba(X) → (n_samples, n_classes).
No weights, no gradients, no architecture access required.
Compatible with: sklearn, PyTorch, TensorFlow/Keras, XGBoost, any callable.
"""

import numpy as np
import warnings
from typing import Callable, List, Optional

from gemex.core.config        import GemexConfig
from gemex.core.result        import GemexResult
from gemex.manifold.fim       import FisherInformationMatrix
from gemex.geometry.geodesic  import GeodesicSolver
from gemex.geometry.gsf       import GeodesicSensitivityField
from gemex.interaction.pti_rst import ParallelTransportInteraction, RiemannianSaliencyTensor, RiemannCurvatureTriplet
from gemex.explain.fas_btd    import FeatureAttentionSequence, BiasTrapDetector
from gemex.data.adapter       import DataAdapter


class Explainer:
    """
    GEMEX Explainer — Geodesic Entropic Manifold Explainability.

    Parameters
    ----------
    model        : any model with predict_proba(X), or callable f(X)→proba
    data_type    : 'tabular' | 'timeseries' | 'image'
    feature_names: list of str
    class_names  : list of str
    config       : GemexConfig  (defaults if None)
    task         : 'classification' | 'regression'
    compute_fas  : bool  compute Feature Attention Sequence (default True)
    compute_btd  : bool  compute Bias Trap Detection (default True)

    Examples
    --------
    >>> exp = Explainer(rf, data_type="tabular",
    ...                 feature_names=cols, class_names=["No","Yes"])
    >>> result = exp.explain(x, X_reference=X_train)
    >>> print(result.summary())
    >>> result.plot("gsf_bar")
    """

    def __init__(self, model, data_type: str = "tabular",
                 feature_names: Optional[List[str]] = None,
                 class_names:   Optional[List[str]] = None,
                 config:        Optional[GemexConfig] = None,
                 task:          str = "classification",
                 compute_fas:   bool = True,
                 compute_btd:   bool = True):

        assert data_type in ("tabular", "timeseries", "image")
        assert task in ("classification", "regression")

        self.data_type     = data_type
        self.feature_names = feature_names
        self.class_names   = class_names
        self.task          = task
        self.compute_fas   = compute_fas
        self.compute_btd   = compute_btd
        self.config        = (config or GemexConfig()).validate()

        if feature_names and len(feature_names) > self.config.lanczos_threshold:
            self.config.use_lanczos = True

        self._predict_fn = self._wrap_model(model, task)
        self._adapter    = DataAdapter(data_type, self.config)

        # Patch→pixel predict wrapper (image only, patch_size > 1)
        # When data_type='image' and image_patch_size > 1, all geometry
        # (FIM, geodesic, GSF) operates in patch space (49 features for
        # patch_size=4 on 28×28).  But the real model was trained on
        # pixel space (784 features).  This wrapper transparently
        # upsamples every patch vector to pixel space before calling
        # the real model.  Tabular and timeseries are completely unaffected
        # because _patch_sz is always 1 for those data types.
        if (data_type == 'image' and
                getattr(self.config, 'image_patch_size', 1) > 1):
            _raw_predict = self._predict_fn
            _adapter_ref = self._adapter
            def _patch_predict(X_patch):
                # X_patch shape: (n, n_patches) or (1, n_patches)
                X_px = np.array([_adapter_ref.patches_to_pixels(row)
                                 for row in X_patch.reshape(
                                     X_patch.shape[0], -1)])
                return _raw_predict(X_px)
            self._predict_fn = _patch_predict

        # Correction 3: auto-detect model type and store on config
        if self.config.model_type == 'auto':
            cls = type(model).__name__.lower()
            tree_kw = ['forest', 'boost', 'tree', 'xgb', 'lgbm',
                       'catboost', 'gradient', 'extra']
            self.config.model_type = (
                'tree' if any(k in cls for k in tree_kw) else 'smooth')
            if self.config.verbose:
                print(f"[GEMEX] Detected model_type='{self.config.model_type}' "
                      f"from {type(model).__name__}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def explain(self, x: np.ndarray,
                X_reference: Optional[np.ndarray] = None,
                target_class: Optional[int] = None) -> GemexResult:
        """
        Compute a full GEMEX explanation for instance x.

        Parameters
        ----------
        x            : instance to explain
        X_reference  : background dataset (recommended: training set)
        target_class : class index to explain; defaults to predicted class

        Returns
        -------
        GemexResult
        """
        x_flat = self._adapter.flatten(x)
        d      = x_flat.shape[0]

        # Reference distribution
        if X_reference is None:
            warnings.warn("[GEMEX] No X_reference provided. Using zero baseline.")
            X_ref = np.zeros((1, d))
        else:
            X_ref = self._adapter.flatten_batch(X_reference)
            if len(X_ref) > self.config.n_reference_samples:
                rng = np.random.default_rng(self.config.random_state)
                idx = rng.choice(len(X_ref), self.config.n_reference_samples,
                                 replace=False)
                X_ref = X_ref[idx]

        # Prediction
        proba = self._predict_fn(x_flat.reshape(1, -1))[0]
        if target_class is None:
            target_class = int(np.argmax(proba))

        # Sub-modules
        fim    = FisherInformationMatrix(self._predict_fn, self.config)
        G      = fim.compute(x_flat, X_ref)
        _fim_quality = fim.fim_quality()
        solver = GeodesicSolver(self._predict_fn, fim, self.config)
        x_base = X_ref.mean(axis=0)
        path, arc = solver.compute_path(x_base, x_flat)

        gsf_engine = GeodesicSensitivityField(self._predict_fn, fim, self.config)
        # Fix 3 + Fix 4: GSF now returns (scores, uncertainty)
        gsf_scores, gsf_uncertainty = gsf_engine.compute(
            x_flat, x_base, path, target_class)

        # Fix 5: optional completeness normalisation
        # Scale raw GSF so sum(gsf) == f(x) - f(baseline)
        if self.config.gsf_normalise:
            p_x    = float(self._predict_fn(x_flat.reshape(1,-1))[0][target_class])
            p_base = float(self._predict_fn(x_base.reshape(1,-1))[0][target_class])
            delta  = p_x - p_base
            gsf_sum = np.sum(gsf_scores) + 1e-12
            gsf_scores      = gsf_scores      * (delta / gsf_sum)
            gsf_uncertainty = gsf_uncertainty * abs(delta / gsf_sum)

        pti_engine = ParallelTransportInteraction(self._predict_fn, fim, self.config)
        pti, holonomy = pti_engine.compute(x_flat, gsf_scores)

        # Triplet Riemann curvature (only when interaction_order == 3)
        rct = None
        if self.config.interaction_order == 3:
            fnames_rct = (self.feature_names
                          or [f'f{i}' for i in range(d)])[:d]
            rct_engine = RiemannCurvatureTriplet(self._predict_fn, fim, self.config)
            rct = rct_engine.compute(x_flat, gsf_scores, fnames_rct)

        rst_engine = RiemannianSaliencyTensor(self.config)
        rst_ev, rst_evec = rst_engine.compute(gsf_scores, pti, G)

        curvature = fim.ricci_scalar(G)

        # Feature Attention Sequence
        fas = None
        if self.compute_fas:
            fas_engine = FeatureAttentionSequence(fim, self.config)
            fnames     = (self.feature_names
                          or [f"f{i}" for i in range(d)])[:d]
            fas = fas_engine.compute(path, fnames)

        # Bias Trap Detection
        btd = None
        if self.compute_btd and fas is not None:
            btd_engine = BiasTrapDetector(self.config)
            btd = btd_engine.compute(
                gsf_scores, pti, G, fas['dwell_time'],
                fnames)

        return GemexResult(
            instance=x, x_flat=x_flat, x_baseline=x_base,
            prediction=target_class, prediction_proba=proba,
            gsf_scores=gsf_scores, gsf_uncertainty=gsf_uncertainty,
            pti_matrix=pti,
            holonomy_angles=holonomy,
            rst_eigenvalues=rst_ev, rst_eigenvectors=rst_evec,
            fim_matrix=G, geodesic_path=path, geodesic_lengths=arc,
            manifold_curvature=curvature,
            feature_names=self.feature_names,
            class_names=self.class_names,
            data_type=self.data_type, config=self.config,
            fas=fas, btd=btd, rct=rct,
            fim_quality=_fim_quality,
        )

    def explain_batch(self, X: np.ndarray,
                      X_reference: Optional[np.ndarray] = None,
                      target_class: Optional[int] = None) -> List[GemexResult]:
        results = []
        for i, x in enumerate(X):
            if self.config.verbose:
                print(f"[GEMEX] {i+1}/{len(X)}")
            results.append(self.explain(x, X_reference, target_class))
        return results

    # ------------------------------------------------------------------ #
    #  Model wrapper                                                       #
    # ------------------------------------------------------------------ #


    def _select_baseline(self, x: np.ndarray,
                          X_ref: np.ndarray,
                          target_class: int) -> np.ndarray:
        """
        Select a geometrically meaningful baseline for the geodesic path.

        Strategy
        --------
        1. Compute predicted probability for all reference points.
        2. Keep the 20% of reference points whose prediction is furthest
           from p(x) — i.e., points in the opposite prediction region.
        3. Return their mean as the baseline.

        This avoids the degenerate global-mean baseline that, for the
        Heart Disease dataset, produces p(baseline) ≈ 0.02 regardless
        of the instance, forcing every geodesic to cross the full [0,1]
        range and making sign assignment unreliable.

        Falls back to global mean if no opposite-region points exist.
        """
        if X_ref is None or len(X_ref) < 4:
            return X_ref.mean(axis=0) if X_ref is not None else x * 0.0

        # Predict for a random subsample of reference points (for speed)
        rng    = np.random.RandomState(self.config.random_state)
        n_sub  = min(len(X_ref), 40)
        idx    = rng.choice(len(X_ref), n_sub, replace=False)
        X_sub  = X_ref[idx]

        p_x    = float(self._predict_fn(x.reshape(1, -1))[0][target_class])
        p_refs = self._predict_fn(X_sub)[: , target_class]   # (n_sub,)

        # Points whose predicted probability differs most from p(x)
        diffs  = np.abs(p_refs - p_x)
        k      = max(4, n_sub // 5)   # top 20%
        top_k  = np.argsort(diffs)[::-1][:k]

        baseline = X_sub[top_k].mean(axis=0)

        # Sanity check: baseline p should not be too close to p(x)
        p_base = float(self._predict_fn(baseline.reshape(1, -1))[0][target_class])
        if abs(p_base - p_x) < 0.05:
            # Fallback to global mean
            return X_ref.mean(axis=0)

        return baseline

    def _wrap_model(self, model, task: str) -> Callable:
        if hasattr(model, "predict_proba"):
            return model.predict_proba

        if task == "regression" and hasattr(model, "predict"):
            def _reg(X):
                p = model.predict(X).reshape(-1, 1)
                p = (p - p.min()) / (p.max() - p.min() + 1e-10)
                return np.hstack([1 - p, p])
            return _reg

        # PyTorch
        try:
            import torch, torch.nn as nn
            if isinstance(model, nn.Module):
                def _torch(X):
                    model.eval()
                    with torch.no_grad():
                        t = torch.tensor(X, dtype=torch.float32)
                        o = model(t)
                        if o.shape[-1] > 1:
                            return torch.softmax(o, dim=-1).numpy()
                        p = torch.sigmoid(o).numpy().flatten()
                        return np.column_stack([1-p, p])
                return _torch
        except ImportError:
            pass

        # TensorFlow
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                def _tf(X):
                    o = model(X, training=False).numpy()
                    if o.shape[-1] > 1:
                        return o
                    p = o.flatten()
                    return np.column_stack([1-p, p])
                return _tf
        except ImportError:
            pass

        if callable(model):
            return model

        raise ValueError(
            "[GEMEX] Cannot wrap model. Provide predict_proba() or a callable.")
