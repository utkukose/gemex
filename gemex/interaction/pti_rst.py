# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
gemex.interaction.pti_rst
--------------------------
Parallel Transport Interaction (PTI) matrix,
Riemannian Curvature Triplet (RCT) tensor, and
Riemannian Saliency Tensor (RST).

PTI — Holonomy-based feature interaction
-----------------------------------------
For each pair (i,j), PTI[i,j] estimates the holonomy angle
of a small square loop in the (xᵢ, xⱼ) subspace at x:

    Holonomy(i,j) ≈ eᵢᵀ [(G(x+εeⱼ)-G(x-εeⱼ))/(2ε)] eⱼ / (G_ii·G_jj)^½

A non-zero holonomy means the manifold is curved in the (i,j)
plane — the features interact non-linearly in the model's geometry.

RCT — Riemannian Curvature Triplet (interaction_order=3)
----------------------------------------------------------
For each triplet (i,j,k), RCT[i,j,k] estimates the Riemann
curvature tensor component R(eᵢ,eⱼ)eₖ at x:

    RCT(i,j,k) ≈ eₖᵀ [∂ᵢG·G⁻¹·∂ⱼG − ∂ⱼG·G⁻¹·∂ᵢG] eₖ / ‖eₖ‖²_G

A non-zero RCT(i,j,k) means features i and j interact
*differently* depending on feature k — a true three-way
modulation that cannot be reduced to any combination of
pairwise PTI values.  This is not detectable by SHAP,
LIME, or any perturbation/attribution method.

RST — Riemannian Saliency Tensor
----------------------------------
    T = α (gsf⊗gsf) + β Sym(PTI) + γ FIM
Eigen-decomposition of T gives principal explanation directions.

References
----------
Do Carmo (1992) Riemannian Geometry, Ch.4;
Amari & Nagaoka (2000) Methods of Information Geometry, Ch.3.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ParallelTransportInteraction:

    def __init__(self, predict_fn, fim, config):
        self.predict_fn = predict_fn
        self.fim        = fim
        self.config     = config

    def compute(self, x: np.ndarray,
                gsf_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d   = len(x)
        pti = np.zeros((d, d))
        if self.config.interaction_order == 1:
            return pti, pti.copy()

        top_k = min(d, 20)
        top   = np.argsort(np.abs(gsf_scores))[::-1][:top_k]

        for a, i in enumerate(top):
            for j in top[a + 1:]:
                angle     = self._holonomy(x, i, j)
                pti[i, j] = angle
                pti[j, i] = angle

        return pti, pti.copy()

    def _holonomy(self, x: np.ndarray, i: int, j: int) -> float:
        eps = self.config.fim_epsilon * 5
        d   = len(x)

        def G_at(xi_off, xj_off):
            xp = x.copy(); xp[i] += xi_off; xp[j] += xj_off
            return self.fim.metric_at(xp)

        G00 = G_at(0,   0)
        G10 = G_at(eps, 0)
        Gm0 = G_at(-eps,0)
        G01 = G_at(0, eps)
        G0m = G_at(0,-eps)

        # Mixed second partial via five-point stencil
        dG = (G10 - Gm0 - G01 + G0m) / (2 * eps ** 2)

        e_i = np.zeros(d); e_i[i] = 1.0
        e_j = np.zeros(d); e_j[j] = 1.0
        raw = float(e_i @ dG @ e_j)
        nrm = np.sqrt(max(G00[i, i], 1e-12) * max(G00[j, j], 1e-12))
        return raw / (nrm + 1e-12)


# ======================================================================= #
#  Riemannian Curvature Triplet (RCT)                                      #
# ======================================================================= #

class RiemannCurvatureTriplet:
    """
    Computes the Riemann curvature tensor component R(eᵢ,eⱼ)eₖ
    for the top-k features ranked by |GSF|.

    The curvature is approximated via finite differences of the FIM:

        ∂ᵢG ≈ [G(x+εeᵢ) − G(x−εeᵢ)] / (2ε)

        R_approx(i,j,k) = eₖᵀ [∂ᵢG · G⁻¹ · ∂ⱼG
                               − ∂ⱼG · G⁻¹ · ∂ᵢG] eₖ
                          / G_kk(x)

    Interpretation
    --------------
    +  synergistic modulation: features i,j amplify each other's
       effect on k's contribution to the prediction
    −  antagonistic modulation: features i,j suppress each other's
       effect on k
    ≈0 features i,j,k are geometrically independent at this point
    """

    def __init__(self, predict_fn, fim, config):
        self.predict_fn = predict_fn
        self.fim        = fim
        self.config     = config

    def compute(self, x: np.ndarray,
                gsf_scores: np.ndarray,
                feature_names: Optional[List[str]] = None
                ) -> Dict:
        """
        Compute triplet curvature for top-k features.

        Returns
        -------
        dict with keys:
            tensor      : dict mapping (i,j,k) → float
            top_triplets: list of (name_i, name_j, name_k, value)
                          sorted by |value| descending
            feature_names: list of str
        """
        d      = len(x)
        eps    = self.config.fim_epsilon * 5
        top_k  = min(d, self.config.rct_top_k)
        top    = np.argsort(np.abs(gsf_scores))[::-1][:top_k]
        fnames = feature_names or [f'f{i}' for i in range(d)]

        # Pre-compute ∂ᵢG for all top features
        G0  = self.fim.metric_at(x)
        G0_inv = self._safe_inv(G0)
        dG  = {}
        for i in top:
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            dG[i] = (self.fim.metric_at(xp) - self.fim.metric_at(xm)) / (2 * eps)

        # Compute R(eᵢ,eⱼ)eₖ for all unique ordered triplets (i<j, k in top)
        tensor   = {}
        records  = []

        for a_idx, i in enumerate(top):
            for b_idx in range(a_idx + 1, len(top)):
                j = top[b_idx]
                # Commutator: [∂ᵢG·G⁻¹·∂ⱼG − ∂ⱼG·G⁻¹·∂ᵢG]
                comm = dG[i] @ G0_inv @ dG[j] - dG[j] @ G0_inv @ dG[i]
                for k in top:
                    e_k   = np.zeros(d); e_k[k] = 1.0
                    raw   = float(e_k @ comm @ e_k)
                    g_kk  = max(float(G0[k, k]), 1e-12)
                    value = raw / g_kk
                    key   = (int(i), int(j), int(k))
                    tensor[key] = value
                    records.append((fnames[i], fnames[j], fnames[k], value))

        records.sort(key=lambda r: abs(r[3]), reverse=True)

        return dict(
            tensor=tensor,
            top_triplets=records,
            feature_names=fnames,
            top_indices=top.tolist(),
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_inv(G: np.ndarray) -> np.ndarray:
        """Pseudo-inverse via eigendecomposition (handles near-singular FIM)."""
        ev, evec = np.linalg.eigh(G)
        ev_inv   = np.where(np.abs(ev) > 1e-10, 1.0 / ev, 0.0)
        return evec @ np.diag(ev_inv) @ evec.T

    def narrative(self, rct: Dict, n: int = 5) -> str:
        lines = [
            "Riemannian Curvature Triplets (RCT)",
            "=" * 42,
            "Each triplet (i, j → k) means: features i and j",
            "jointly modulate k's contribution — irreducible to pairs.",
            "",
        ]
        for fi, fj, fk, val in rct['top_triplets'][:n]:
            kind = "synergistic" if val > 0 else "antagonistic"
            lines.append(
                f"  ({fi} × {fj}) → {fk}:  {val:+.4f}  [{kind}]"
            )
        return "\n".join(lines)


class RiemannianSaliencyTensor:

    def __init__(self, config, alpha=0.50, beta=0.30, gamma=0.20):
        self.config = config
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma

    def compute(self, gsf: np.ndarray,
                pti: np.ndarray,
                fim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d     = len(gsf)
        n_c   = min(self.config.rst_n_components, d)
        eps   = 1e-12

        g_n  = gsf / (np.linalg.norm(gsf) + eps)
        T_g  = np.outer(g_n, g_n)

        T_p  = 0.5 * (pti + pti.T)
        ps   = np.max(np.abs(T_p)) + eps
        T_p /= ps

        ts   = np.trace(fim) / d + eps
        T_f  = fim / ts

        T    = self.alpha * T_g + self.beta * T_p + self.gamma * T_f
        T    = 0.5 * (T + T.T)

        ev, evec = np.linalg.eigh(T)
        idx  = np.argsort(np.abs(ev))[::-1]
        ev   = ev[idx][:n_c]
        evec = evec[:, idx][:, :n_c].T   # (n_c, d)
        return ev, evec

    def explained_variance_ratio(self, eigenvalues: np.ndarray) -> np.ndarray:
        total = np.sum(np.abs(eigenvalues)) + 1e-12
        return np.abs(eigenvalues) / total
