# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.0
"""
GEMEX — Geodesic Entropic Manifold Explainability
==================================================
Version : 1.2.0
License : MIT

A novel, model-agnostic XAI library grounded in Riemannian
information geometry and differential geometry.

Theoretical foundations
-----------------------
GEMEX treats a trained model f: X → Δ(Y) as defining a statistical
manifold M equipped with the Fisher Information Metric (FIM).
Every prediction is a point on this manifold; explanations emerge
from the intrinsic geometry of M rather than from linear surrogates,
perturbation sampling, or game-theoretic axioms.

Three core objects
------------------
GSF  — Geodesic Sensitivity Field
       Path-integrated curvature of the prediction surface along
       each feature direction on the Riemannian manifold.
PTI  — Parallel Transport Interaction matrix
       Holonomy angles measuring true nonlinear feature interactions.
RST  — Riemannian Saliency Tensor
       Rank-2 tensor fusing GSF and PTI; eigen-decomposition gives
       principal explanation directions and explained variance.

Two additional analyses
-----------------------
FAS  — Feature Attention Sequence
       Which feature the model attends to at each geodesic step.
BTD  — Bias Trap Detector
       Geometric signals for spurious attention and over-reliance.

Quick start
-----------
>>> from gemex import Explainer, GemexConfig
>>> cfg = GemexConfig(n_geodesic_steps=40, interaction_order=2)
>>> exp = Explainer(model, data_type="tabular",
...                 feature_names=cols, class_names=labels, config=cfg)
>>> result = exp.explain(x_instance, X_reference=X_train)
>>> result.plot("gsf_bar")
>>> result.plot("beeswarm", batch_results=batch)
>>> print(result.summary())
"""

from gemex.core.config    import GemexConfig
from gemex.core.explainer import Explainer
from gemex.core.result    import GemexResult

__version__  = "1.2.0"
__author__   = "Dr. Utku Kose"
__orcid__    = "https://orcid.org/0000-0002-9652-6415"
__all__      = ["Explainer", "GemexResult", "GemexConfig"]
