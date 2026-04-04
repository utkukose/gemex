# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
gemex.explain.fas_btd
---------------------
Feature Attention Sequence (FAS) and Bias Trap Detector (BTD).

FAS — Feature Attention Sequence
----------------------------------
At each geodesic step t, the leading eigenvector of G(γ(t))
identifies which feature direction the manifold is most stretched
along.  The sequence of dominant features across all steps forms
the "attention sequence" of the model.

    FAS(t) = argmax_i  [v₁(G(γ(t)))]²_i

Dwell time:   fraction of geodesic steps feature i dominates.
Transitions:  steps where the dominant feature changes.

BTD — Bias Trap Detector
--------------------------
Three geometric bias signals:

HAT  Holonomy Asymmetry Test
     BiasScore_HAT(i) = Σⱼ |PTI(i,j)| / (|GSF(i)| + ε)
     High → feature entangles many others without direct effect
     → possible confounder.

MCA  Manifold Curvature Asymmetry
     BiasScore_MCA(i) = |G_ii − mean(G_jj)| / mean(G_jj)
     High → model is unusually sensitive to feature i
     → possible over-reliance.

GDI  Geodesic Dominance Inconsistency
     BiasScore_GDI(i) = dwell(i) / (|GSF(i)|/max_GSF + ε)
     High → attended throughout path but low direct effect
     → possible spurious correlation.

Combined:  BiasRisk(i) = 0.4·HAT + 0.35·MCA + 0.25·GDI  (normalised).

References
----------
Amari (2016) Ch.4 (α-connection asymmetry);
Peters et al. (2017) Elements of Causal Inference.
"""

import numpy as np
from typing import List, Dict


# ======================================================================= #
#  Feature Attention Sequence                                              #
# ======================================================================= #

class FeatureAttentionSequence:

    def __init__(self, fim, config):
        self.fim    = fim
        self.config = config

    def compute(self, geodesic_path: np.ndarray,
                feature_names: List[str]) -> Dict:
        n_steps, n_feat = geodesic_path.shape
        d = min(len(feature_names), n_feat)

        sequence  = []
        dominance = []

        for step in range(n_steps):
            x_t = geodesic_path[step]
            G_t = self._local_fim(x_t, step, geodesic_path)
            ev, evec = np.linalg.eigh(G_t)
            idx  = np.argsort(ev)[::-1]
            lead = np.abs(evec[:, idx[0]])[:d]
            dom_feat  = int(np.argmax(lead))
            dom_ratio = ev[idx[0]] / (np.sum(np.abs(ev)) + 1e-12)
            sequence.append(dom_feat)
            dominance.append(float(dom_ratio))

        sequence  = np.array(sequence)
        dominance = np.array(dominance)

        dwell = np.bincount(sequence, minlength=d) / n_steps

        trans_steps = [s for s in range(1, n_steps)
                       if sequence[s] != sequence[s - 1]]
        trans_feats = [(feature_names[min(sequence[s-1], d-1)],
                        feature_names[min(sequence[s],   d-1)])
                       for s in trans_steps]

        mid = n_steps // 2
        ec  = np.bincount(sequence[:mid],  minlength=d)
        lc  = np.bincount(sequence[mid:],  minlength=d)
        early = [feature_names[i] for i in np.argsort(ec)[::-1][:3]]
        late  = [feature_names[i] for i in np.argsort(lc)[::-1][:3]]

        return dict(sequence=sequence, dominance=dominance,
                    dwell_time=dwell,
                    transition_steps=trans_steps,
                    transition_features=trans_feats,
                    early_features=early, late_features=late,
                    feature_names=list(feature_names))

    def _local_fim(self, x_t, step, path):
        G = self.fim.metric_at(x_t)
        n = len(path)
        if 0 < step < n - 1:
            tang = path[step + 1] - path[step - 1]
            tn   = np.linalg.norm(tang) + 1e-12
            G    = G + 0.12 * np.outer(tang / tn, tang / tn)
        return G

    def narrative(self, fas: Dict) -> str:
        dwell = fas['dwell_time']
        names = fas['feature_names']
        top   = sorted(zip(names, dwell), key=lambda x: x[1], reverse=True)[:3]
        n_tr  = len(fas['transition_steps'])
        lines = [
            "Feature Attention Sequence",
            "=" * 42,
            f"Attention switches along geodesic: {n_tr}",
            "",
            "Early path focus (near baseline):",
        ] + [f"  → {f}" for f in fas['early_features']] + [
            "",
            "Late path focus (near decision):",
        ] + [f"  → {f}" for f in fas['late_features']] + [
            "",
            "Overall dwell time (top 3):",
        ] + [f"  {n:<24} {d*100:.1f}%" for n, d in top]
        return "\n".join(lines)


# ======================================================================= #
#  Bias Trap Detector                                                      #
# ======================================================================= #

class BiasTrapDetector:

    def __init__(self, config, alpha=0.40, beta=0.35, gamma=0.25):
        self.config = config
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma

    def compute(self, gsf: np.ndarray, pti: np.ndarray,
                fim: np.ndarray, dwell: np.ndarray,
                feature_names: List[str]) -> Dict:
        d   = len(feature_names)
        eps = 1e-10
        mg  = np.max(np.abs(gsf)) + eps

        # HAT
        hat = np.array([
            (np.sum(np.abs(pti[i, :])) - abs(pti[i, i]))
            / (abs(gsf[i]) + eps) for i in range(d)])
        hat /= hat.max() + eps

        # MCA
        fd  = np.diag(fim)
        mca = np.array([
            abs(fd[i] - np.mean(np.delete(fd, i)))
            / (np.mean(np.delete(fd, i)) + eps) for i in range(d)])
        mca /= mca.max() + eps

        # GDI
        gdi = dwell / (np.abs(gsf) / mg + eps)
        gdi /= gdi.max() + eps

        bias = (self.alpha * hat + self.beta * mca + self.gamma * gdi)
        bias /= bias.max() + eps

        levels = ['high' if b > 0.70 else 'moderate' if b > 0.40 else 'low'
                  for b in bias]

        top_idx = np.argsort(bias)[::-1][:3]
        top_b   = [(feature_names[i], float(bias[i]), levels[i])
                   for i in top_idx]

        err = dwell * (1 - np.abs(gsf) / mg)
        err /= err.max() + eps
        err_idx = np.argsort(err)[::-1][:3]
        err_v   = [(feature_names[i], float(err[i])) for i in err_idx]

        return dict(hat_scores=hat, mca_scores=mca, gdi_scores=gdi,
                    bias_risk=bias, risk_level=levels,
                    top_biased=top_b, error_vulnerable=err_v,
                    feature_names=list(feature_names))

    def narrative(self, btd: Dict) -> str:
        lines = ["Bias Trap Analysis", "=" * 42]
        rc = {'high': '[HIGH]', 'moderate': '[MOD]', 'low': '[LOW]'}
        for name, score, level in btd['top_biased']:
            i = btd['feature_names'].index(name)
            lines += [
                f"\n  {rc[level]}  {name}  (risk={score:.3f})",
                f"    HAT={btd['hat_scores'][i]:.3f}  "
                f"MCA={btd['mca_scores'][i]:.3f}  "
                f"GDI={btd['gdi_scores'][i]:.3f}",
            ]
        lines += ["\nError-vulnerable features (distribution shift risk):"]
        for name, score in btd['error_vulnerable']:
            lines.append(f"  {name:<24}  {score:.3f}")
        return "\n".join(lines)
