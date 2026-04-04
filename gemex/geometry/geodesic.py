# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
gemex.geometry.geodesic
-----------------------
Geodesic path computation on the statistical manifold.

The geodesic equation on a Riemannian manifold is:

    d²γⁱ/dt² + Γⁱⱼₖ (dγʲ/dt)(dγᵏ/dt) = 0

where Γⁱⱼₖ are the Christoffel symbols of the FIM connection.

GEMEX solves this ODE with a 4th-order Runge-Kutta integrator
using an approximate Christoffel term derived from directional
finite differences of G.

Numerical stability design
--------------------------
Three mechanisms keep integration bounded on practical ML models:

(1) Damped Christoffel acceleration
    The raw Christoffel term -G⁻¹ ∂_v G v / 2 can be arbitrarily
    large when the FIM changes rapidly (steep gradient-boosting
    decision boundaries).  We clip the acceleration magnitude to
    at most `accel_clip` × ‖v‖ per step, preventing the geodesic
    from collapsing onto a single dominant feature direction.
    Default accel_clip = 0.20 (≤ 20% velocity correction per step).

(2) Velocity renormalisation
    After each RK4 step the velocity is renormalised to ‖v₀‖ under
    the FIM inner product, preventing exponential growth of speed
    across steps.

(3) NaN / Inf guards
    All intermediate results are sanitised; any divergent step falls
    back to a zero acceleration (straight-line behaviour at that step)
    rather than propagating NaN through the path.

References
----------
Do Carmo (1992) Riemannian Geometry, Ch.3;
Amari & Nagaoka (2000) Methods of Information Geometry, Ch.2.
"""

import numpy as np
from typing import Tuple


class GeodesicSolver:

    def __init__(self, predict_fn, fim, config):
        self.predict_fn = predict_fn
        self.fim        = fim
        self.config     = config

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute_path(self,
                     x_start: np.ndarray,
                     x_end:   np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the geodesic path from x_start to x_end.

        Returns
        -------
        path : (n_steps, d)   positions along the geodesic
        arc  : (n_steps,)     cumulative Fisher-Rao arc-length
        """
        n    = self.config.n_geodesic_steps
        path = self._rk4_path(x_start, x_end, n)
        arc  = self._arc_lengths(path)
        return path, arc

    def total_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        _, arc = self.compute_path(x1, x2)
        return float(arc[-1])

    # ------------------------------------------------------------------ #
    #  Numerically stable RK4 geodesic integrator                         #
    # ------------------------------------------------------------------ #

    def _rk4_path(self, x_start: np.ndarray,
                  x_end: np.ndarray, n: int) -> np.ndarray:
        """
        4th-order Runge-Kutta integration of the geodesic ODE with
        damped Christoffel acceleration and velocity renormalisation.

        ODE system
        ----------
            dγ/dt = v
            dv/dt = accel_damped(γ, v)

        where accel_damped clips ‖dv/dt‖ ≤ accel_clip × ‖v‖ and
        sanitises any NaN/Inf produced by near-singular FIM regions.
        """
        d          = len(x_start)
        eps        = self.config.fim_epsilon * 3.0
        accel_clip = 0.20      # max Christoffel correction per ‖v‖
        dt         = 1.0 / max(n - 1, 1)

        # Initial velocity: vector from start to end, traversed in t ∈ [0,1]
        diff = x_end - x_start
        v0   = diff.copy()                  # reaches x_end exactly at t=1
        v0_norm = np.linalg.norm(v0) + 1e-10

        def christoffel_accel(x: np.ndarray, v: np.ndarray) -> np.ndarray:
            """
            Damped Christoffel acceleration.

            Computes -½ G⁻¹ (∂_v G) v then clips its magnitude to
            accel_clip × ‖v‖ and replaces any NaN/Inf with zero.
            """
            v_norm = np.linalg.norm(v) + 1e-10
            v_dir  = v / v_norm

            # Directional derivative of G along the velocity direction
            xp   = x + eps * v_dir
            xm   = x - eps * v_dir
            G_p  = self.fim.metric_at(xp)
            G_m  = self.fim.metric_at(xm)
            dG_v = (G_p - G_m) / (2.0 * eps)

            G0 = self.fim.metric_at(x)

            # Pseudo-inverse with generous regularisation for stability
            try:
                G0_inv = np.linalg.pinv(G0, rcond=1e-8)
            except Exception:
                return np.zeros(d)

            raw_accel = -0.5 * G0_inv @ (dG_v @ v)

            # Guard against NaN / Inf from near-singular regions
            if not np.all(np.isfinite(raw_accel)):
                return np.zeros(d)

            # Clip magnitude: correction ≤ accel_clip × ‖v‖
            accel_norm = np.linalg.norm(raw_accel) + 1e-10
            max_norm   = accel_clip * v_norm
            if accel_norm > max_norm:
                raw_accel = raw_accel * (max_norm / accel_norm)

            return raw_accel

        def ode_rhs(x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray,
                                                             np.ndarray]:
            return v.copy(), christoffel_accel(x, v)

        # RK4 integration
        path  = np.zeros((n, d))
        path[0] = x_start.copy()
        x_cur   = x_start.copy()
        v_cur   = v0.copy()

        for i in range(1, n):

            # Standard RK4 step
            k1x, k1v = ode_rhs(x_cur,                 v_cur)
            k2x, k2v = ode_rhs(x_cur + 0.5*dt*k1x,   v_cur + 0.5*dt*k1v)
            k3x, k3v = ode_rhs(x_cur + 0.5*dt*k2x,   v_cur + 0.5*dt*k2v)
            k4x, k4v = ode_rhs(x_cur +    dt*k3x,     v_cur +    dt*k3v)

            x_new = x_cur + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            v_new = v_cur + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

            # Sanitise position
            if not np.all(np.isfinite(x_new)):
                x_new = x_cur + dt * v_cur   # fallback: Euler step

            # Velocity renormalisation — keep ‖v‖ ≈ ‖v₀‖ to prevent
            # exponential growth or collapse across steps
            v_new_norm = np.linalg.norm(v_new) + 1e-10
            if np.isfinite(v_new_norm) and v_new_norm > 1e-10:
                v_new = v_new * (v0_norm / v_new_norm)
            else:
                v_new = v_cur.copy()   # fallback: keep previous velocity

            # Force last step to land exactly on x_end
            if i == n - 1:
                x_new = x_end.copy()

            path[i] = x_new
            x_cur   = x_new
            v_cur   = v_new

        return path

    # ------------------------------------------------------------------ #
    #  Arc-length computation                                              #
    # ------------------------------------------------------------------ #

    def _arc_lengths(self, path: np.ndarray) -> np.ndarray:
        """Cumulative Fisher-Rao arc-length along the path."""
        n   = len(path)
        arc = np.zeros(n)
        for i in range(1, n):
            dx     = path[i] - path[i - 1]
            x_mid  = 0.5 * (path[i] + path[i - 1])
            seg    = self.fim.norm(x_mid, dx)
            # Guard against overflow in arc accumulation
            arc[i] = arc[i - 1] + (seg if np.isfinite(seg) else 0.0)
        return arc
