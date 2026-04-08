# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.2
"""
gemex.geometry.gsf
------------------
Geodesic Sensitivity Field (GSF).

For feature i, the GSF at instance x is:

    GSF_i(x) = ∫₀¹ ⟨γ'(t), eᵢ⟩_G / (‖γ'(t)‖_G · ‖eᵢ‖_G) · ‖γ'(t)‖_G dt

where γ(t) is the geodesic from x_baseline to x on M, and ⟨·,·⟩_G
is the Fisher inner product.

Normalisation design
--------------------
A balanced normalisation prevents one large-G_ii feature from
absorbing all attribution mass. The relative sensitivity factor
√(G_ii / mean(G_jj)) is soft-clipped to [1/3, 3].

Enhancement 1  for version 1.2.0 — Numerically stable sign assignment
---------------------------------------------------
The previous sign integrator used raw log(p(y*|γ(t))), which is
unbounded near 0 and 1.  For near-certain predictions (p < 0.01 or
p > 0.99) this caused the integral to accumulate erratically and
the overall GSF sum to flip sign relative to f(x)−f(baseline).

New implementation:
  (a) Clips probabilities to [0.01, 0.99] before taking log, making
      the log-likelihood derivative bounded everywhere.
  (b) Falls back to the raw prediction-difference direction for
      fully saturated instances (|p(x) − 0.5| > 0.48), where the
      log derivative is unreliable even after clipping.
  (c) Uses the adaptive epsilon from config for the directional
      derivative rather than the base fim_epsilon.

Enhancement 2  for version 1.2.0 — Prediction-confidence-adaptive path weighting
--------------------------------------------------------------
For near-certain predictions the early path steps (near baseline)
carry the geometrically meaningful information — the model is
already saturated near the endpoint and those steps contribute
mostly noise to the integral.

A confidence-based weight w(t) emphasises early steps when the
prediction is extreme:

    conf = clip(|p(x) − 0.5|, 0, 0.5) × 2   ∈ [0, 1]
    w(t) = (1 − conf) + conf × exp(−3t)

For uncertain predictions (conf ≈ 0) weights are uniform.
For certain predictions (conf ≈ 1) early steps are weighted ~e^3
more than late steps.  Weights are normalised so ∫w(t)dt = 1.

Enhancement 3  for version 1.2.0 — Increased fim_local_n for tree models
------------------------------------------------------
Handled in config.py / fim.py.  The GSF integrator is unchanged.

Properties
----------
- Reparameterisation-invariant
- Balanced: comparable magnitudes across features
- Signed: numerically stable for saturated predictions
- Uncertainty: geometric, per-feature, path-integrated

References
----------
Amari & Nagaoka (2000) Ch.2;
Peters et al. (2017) Elements of Causal Inference, Ch.6.
"""

import numpy as np
from typing import Callable


class GeodesicSensitivityField:

    def __init__(self, predict_fn: Callable, fim, config):
        self.predict_fn = predict_fn
        self.fim        = fim
        self.config     = config

    def compute(self, x: np.ndarray, x_baseline: np.ndarray,
                geodesic_path: np.ndarray,
                target_class: int):
        """
        Compute GSF scores and per-feature curvature uncertainty.

        Returns
        -------
        gsf             : (d,)  signed, balanced attribution scores
        gsf_uncertainty : (d,)  curvature-weighted confidence widths
        """
        d        = x.shape[0]
        n_steps  = len(geodesic_path)
        tangents = np.gradient(geodesic_path, axis=0)

        # ── Prediction confidence (Enhancement 2 for version 1.2.0) ──
        p_x_raw = float(np.clip(
            self.predict_fn(x.reshape(1, -1))[0][target_class], 1e-6, 1.))
        p_b_raw = float(np.clip(
            self.predict_fn(x_baseline.reshape(1, -1))[0][target_class], 1e-6, 1.))
        conf    = min(abs(p_x_raw - 0.5) * 2.0, 1.0)   # 0=uncertain, 1=certain

        # Confidence-adaptive step weights: emphasise early path when certain
        ts      = np.linspace(0, 1, n_steps)
        raw_w   = (1.0 - conf) + conf * np.exp(-3.0 * ts)
        weights = raw_w / (raw_w.sum() + 1e-12)   # normalised, sum = 1

        # ── Main integration loop ────────────────────────────────────
        gsf_raw      = np.zeros(d)
        fim_diag_sum = np.zeros(d)
        ricci_steps  = np.zeros(n_steps)

        for step in range(n_steps):
            x_t   = geodesic_path[step]
            v_t   = tangents[step]
            G_t   = self.fim.metric_at(x_t)
            speed = np.sqrt(max(float(v_t @ G_t @ v_t), 1e-12))
            w_t   = weights[step]

            ricci_steps[step] = self.fim.ricci_scalar(G_t)

            for i in range(d):
                e_i    = np.zeros(d); e_i[i] = 1.0
                inner  = float(v_t @ G_t @ e_i)
                g_ii   = max(float(G_t[i, i]), 1e-12)
                e_norm = np.sqrt(g_ii)
                cosine_i        = inner / (speed * e_norm + 1e-12)
                gsf_raw[i]      += cosine_i * w_t    # weighted integral
                fim_diag_sum[i] += g_ii * w_t

        # ── Balanced FIM-diagonal normalisation ──────────────────────
        mean_diag = np.mean(fim_diag_sum) + 1e-12
        rel_sens  = np.sqrt(fim_diag_sum / mean_diag)
        rel_sens  = np.clip(rel_sens, 1.0 / 3.0, 3.0)
        gsf       = gsf_raw * rel_sens

        # Normalise to max |GSF| = 1 before sign assignment
        gsf_max = np.max(np.abs(gsf)) + 1e-12
        gsf     = gsf / gsf_max

        # ── Enhancement 1: stable sign assignment ───────────────────
        gsf = self._apply_sign_stable(
            gsf, geodesic_path, tangents, target_class,
            weights, p_x_raw, p_b_raw)

        # ── Restore scale ────────────────────────────────────────────
        # Scale so |sum(GSF)| ≈ |f(x) − f(baseline)|
        delta = abs(p_x_raw - p_b_raw) + 1e-12
        gsf   = gsf * delta

        # ── Enhancement 4: curvature uncertainty (unchanged) ─────────
        gsf_uncertainty = self._compute_uncertainty(
            geodesic_path, tangents, ricci_steps, weights)

        return gsf, gsf_uncertainty

    # --------------------------------------------------------------------- #
    #  Enhancement 1 for version 1.2.0: Numerically stable sign assignment    #
    # --------------------------------------------------------------------- #

    def _apply_sign_stable(self, gsf: np.ndarray,
                            path: np.ndarray,
                            tangents: np.ndarray,
                            target_class: int,
                            weights: np.ndarray,
                            p_x: float,
                            p_b: float) -> np.ndarray:
        """
        Assign signs to GSF scores using a numerically stable approach.

        Strategy
        --------
        (a) Fully saturated instances (|p(x)−0.5| > 0.48):
            Use the raw prediction-difference direction.
            The sign of (p(x) − p(baseline)) × (x_i − baseline_i)
            tells us whether feature i moves toward the prediction.
            The log-likelihood derivative is unreliable here.

        (b) All other instances:
            Integrate the bounded log-likelihood directional derivative
            with probabilities clipped to [0.01, 0.99] to prevent
            blow-up near the saturation boundaries.
        """
        d     = len(gsf)
        out   = gsf.copy()

        # Case (a): near-saturated — use raw direction
        if abs(p_x - 0.5) > 0.48:
            x_end    = path[-1]
            x_start  = path[0]
            diff     = x_end - x_start
            pred_dir = np.sign(p_x - p_b)   # did prediction go up or down?
            for i in range(d):
                feat_dir = np.sign(diff[i])
                s = pred_dir * feat_dir if feat_dir != 0 else 1.0
                out[i] = abs(gsf[i]) * (s if s != 0 else 1.0)
            return out

        # Case (b): bounded log-likelihood sign integral
        # Use adaptive epsilon if available, else base epsilon
        eps       = getattr(self.config, '_active_eps', None) or \
                    self.config.fim_epsilon
        n_steps   = len(path)
        integrals = np.zeros(d)
        # Sample uniformly — up to 20 steps for speed
        step_ids  = range(0, n_steps, max(1, n_steps // 20))

        for step in step_ids:
            x_t = path[step]
            v_t = tangents[step]
            w_t = weights[step]
            for i in range(d):
                if abs(v_t[i]) < 1e-12:
                    continue
                xp = x_t.copy(); xp[i] += eps
                xm = x_t.copy(); xm[i] -= eps
                # Clip to [0.01, 0.99] — prevents log blow-up (Enhancement 1)
                pp_i = float(np.clip(
                    self.predict_fn(xp.reshape(1, -1))[0][target_class],
                    0.01, 0.99))
                pm_i = float(np.clip(
                    self.predict_fn(xm.reshape(1, -1))[0][target_class],
                    0.01, 0.99))
                dlog_p = (np.log(pp_i) - np.log(pm_i)) / (2.0 * eps)
                integrals[i] += dlog_p * v_t[i] * w_t

        for i in range(d):
            s = np.sign(integrals[i])
            if s != 0:
                out[i] = abs(gsf[i]) * s
        return out

    # ------------------------------------------------------------------ #
    #  Curvature-weighted uncertainty                                      #
    # ------------------------------------------------------------------ #

    def _compute_uncertainty(self, path: np.ndarray,
                              tangents: np.ndarray,
                              ricci_steps: np.ndarray,
                              weights: np.ndarray) -> np.ndarray:
        """
        Per-feature curvature-weighted confidence width:

            σ_i = ∫₀¹ κ(γ(t)) · |⟨γ'(t), eᵢ⟩_G| / (‖γ'‖_G · ‖eᵢ‖_G) · w(t) dt
        """
        d       = path.shape[1]
        n_steps = len(path)
        unc     = np.zeros(d)

        for step in range(n_steps):
            x_t   = path[step]
            v_t   = tangents[step]
            G_t   = self.fim.metric_at(x_t)
            speed = np.sqrt(max(float(v_t @ G_t @ v_t), 1e-12))
            kappa = ricci_steps[step]
            w_t   = weights[step]
            if kappa < 1e-12:
                continue

            for i in range(d):
                e_i    = np.zeros(d); e_i[i] = 1.0
                inner  = abs(float(v_t @ G_t @ e_i))
                e_norm = np.sqrt(max(float(G_t[i, i]), 1e-12))
                unc[i] += kappa * (inner / (speed * e_norm + 1e-12)) * w_t

        return unc
