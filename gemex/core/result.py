# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""gemex.core.result — GemexResult returned by Explainer.explain()."""

import numpy as np
from typing import List, Optional, Dict, Any


class GemexResult:
    """
    Full GEMEX explanation for a single instance.

    Attributes (geometric)
    ----------------------
    gsf_scores       (n_features,)            Geodesic Sensitivity Field
    gsf_uncertainty  (n_features,)            Curvature-weighted per-feature confidence width
    pti_matrix       (n_features, n_features) Parallel Transport Interaction
    rst_eigenvalues  (n_components,)          RST eigenspectrum
    rst_eigenvectors (n_components, n_features) RST principal directions
    fim_matrix       (n_features, n_features) Fisher Information Matrix
    geodesic_path    (n_steps, n_features)    Path from baseline to instance
    geodesic_lengths (n_steps,)               Cumulative Fisher-Rao arc-length
    manifold_curvature float                  Approximate scalar Ricci curvature

    Attributes (extra analyses)
    ---------------------------
    fas   dict | None   Feature Attention Sequence (if compute_fas=True)
    btd   dict | None   Bias Trap Detection (if compute_btd=True)
    rct   dict | None   Riemannian Curvature Triplets (if interaction_order=3)
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'rct'):
            self.rct = None
        if not hasattr(self, 'gsf_uncertainty'):
            self.gsf_uncertainty = np.zeros_like(self.gsf_scores)
        if not hasattr(self, 'fim_quality'):
            self.fim_quality = 'unknown'
        self._narratives: Dict[str, str] = {}

    def top_features(self, n: int = 5) -> List[tuple]:
        names = self.feature_names or [f"f{i}" for i in range(len(self.gsf_scores))]
        return sorted(zip(names, self.gsf_scores),
                      key=lambda x: abs(x[1]), reverse=True)[:n]

    def top_interactions(self, n: int = 5) -> List[tuple]:
        names = self.feature_names or [f"f{i}" for i in range(len(self.gsf_scores))]
        d = len(names)
        pairs = [(names[i], names[j], self.pti_matrix[i, j])
                 for i in range(d) for j in range(i + 1, d)]
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:n]

    def top_triplets(self, n: int = 5) -> List[tuple]:
        """
        Return the n strongest Riemannian Curvature Triplets (RCT).

        Each entry is (feat_i, feat_j, feat_k, R_value) where:
          feat_i, feat_j : the interaction-plane features (the loop)
          feat_k         : the probe feature being modulated
          R_value > 0    : synergistic three-way modulation
          R_value < 0    : antagonistic three-way modulation

        Returns [] if RCT was not computed (interaction_order < 3).
        """
        if self.rct is None:
            return []
        return self.rct['top_triplets'][:n]

    def uncertainty_level(self) -> str:
        c = abs(self.manifold_curvature)
        return 'high' if c > 0.5 else 'moderate' if c > 0.1 else 'low'

    def confidence_score(self) -> float:
        return float(np.clip(1.0 / (1.0 + abs(self.manifold_curvature)), 0., 1.))

    def explained_variance_ratio(self) -> np.ndarray:
        ev = np.abs(self.rst_eigenvalues)
        return ev / (ev.sum() + 1e-12)

    def summary(self) -> str:
        cls  = (self.class_names[self.prediction]
                if self.class_names else str(self.prediction))
        prob = self.prediction_proba[self.prediction]
        top  = self.top_features(5)
        lines = [
            "GEMEX Explanation Summary",
            "=" * 42,
            f"Prediction : {cls}  (p={prob:.3f})",
            f"Uncertainty: {self.uncertainty_level()}  "
            f"(curvature={self.manifold_curvature:.3f})",
            f"Geodesic length: {self.geodesic_lengths[-1]:.4f}",
        f"FIM quality    : {self.fim_quality}"
        + ("" if self.fim_quality == "good" else
           "  [WARNING: low — try fim_epsilon=0.05 or interaction_order=1]"),
            "",
            "Top features (GSF):",
        ]
        for name, score in top:
            direction = "to" if score >= 0 else "against"
            lines.append(f"  [{direction}] {name:<26} {score:+.4f}")
        lines += ["", "Top interactions (holonomy):"]
        for fi, fj, angle in self.top_interactions(3):
            kind = "synergy" if angle > 0 else "antagonism"
            lines.append(f"  {fi} x {fj}: {angle:+.4f}  ({kind})")
        if self.rct is not None:
            lines += ["", "Top triplets (Riemann curvature):"]
            for fi, fj, fk, val in self.top_triplets(3):
                kind = "synergistic" if val > 0 else "antagonistic"
                lines.append(f"  ({fi} x {fj}) -> {fk}: {val:+.4f}  [{kind}]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        cn = (self.class_names[self.prediction]
              if self.class_names else str(self.prediction))
        out = dict(
            prediction=int(self.prediction),
            predicted_class=cn,
            prediction_proba=self.prediction_proba.tolist(),
            gsf_scores=self.gsf_scores.tolist(),
            top_features=[(n, float(s)) for n, s in self.top_features(5)],
            top_interactions=[(a, b, float(v)) for a, b, v in self.top_interactions(3)],
            rst_eigenvalues=self.rst_eigenvalues.tolist(),
            explained_variance_ratio=self.explained_variance_ratio().tolist(),
            manifold_curvature=float(self.manifold_curvature),
            uncertainty_level=self.uncertainty_level(),
            confidence_score=float(self.confidence_score()),
            geodesic_length=float(self.geodesic_lengths[-1]),
        )
        if self.rct is not None:
            out['top_triplets'] = [
                (fi, fj, fk, float(v))
                for fi, fj, fk, v in self.top_triplets(5)
            ]
        return out

    def plot(self, kind: str = "gsf_bar", **kwargs):
        from gemex.viz.dispatcher import VizDispatcher
        return VizDispatcher(result=self, config=self.config).plot(kind, **kwargs)

    def __repr__(self):
        cls  = (self.class_names[self.prediction]
                if self.class_names else str(self.prediction))
        prob = self.prediction_proba[self.prediction]
        top  = self.top_features(3)
        top_str = ", ".join(f"{n}={v:+.3f}" for n, v in top)
        rct_str = (f", rct={len(self.rct['top_triplets'])}triplets"
                   if self.rct else "")
        return (f"GemexResult(prediction='{cls}' p={prob:.3f}, "
                f"top=[{top_str}], "
                f"curvature={self.manifold_curvature:.3f}, "
                f"uncertainty='{self.uncertainty_level()}'{rct_str})")
