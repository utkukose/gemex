# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
tests/test_gemex.py
--------------------
GEMEX test suite — run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=200, n_features=6,
                                n_informative=4, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def model(data):
    Xtr, _, ytr, _ = data
    return RandomForestClassifier(n_estimators=30, random_state=42).fit(Xtr, ytr)


@pytest.fixture(scope="module")
def feature_names():
    return [f"F{i}" for i in range(6)]


@pytest.fixture(scope="module")
def explainer(model, feature_names):
    from gemex import Explainer, GemexConfig
    cfg = GemexConfig(n_geodesic_steps=8, fim_epsilon=8e-3,
                      n_reference_samples=20, interaction_order=2,
                      rst_n_components=3, verbose=False)
    return Explainer(model, data_type="tabular",
                     feature_names=feature_names,
                     class_names=["C0", "C1"], config=cfg)


@pytest.fixture(scope="module")
def result(explainer, data):
    Xtr, Xte, _, _ = data
    return explainer.explain(Xte[0], X_reference=Xtr)


# ── Config ────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        from gemex import GemexConfig
        cfg = GemexConfig()
        assert cfg.n_geodesic_steps == 40
        assert cfg.interaction_order == 2

    def test_validate_passes(self):
        from gemex import GemexConfig
        GemexConfig(n_geodesic_steps=10, alpha_connection=0.5).validate()

    def test_validate_fails(self):
        from gemex import GemexConfig
        with pytest.raises(AssertionError):
            GemexConfig(n_geodesic_steps=3).validate()


# ── FIM ───────────────────────────────────────────────────────────────

class TestFIM:
    def test_shape(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        _, Xte, _, _ = data
        fim = FisherInformationMatrix(model.predict_proba, GemexConfig(fim_epsilon=8e-3))
        G = fim.compute(Xte[0])
        assert G.shape == (6, 6)

    def test_symmetric(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        _, Xte, _, _ = data
        fim = FisherInformationMatrix(model.predict_proba, GemexConfig(fim_epsilon=8e-3))
        G = fim.compute(Xte[0])
        np.testing.assert_allclose(G, G.T, atol=1e-10)

    def test_psd(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        _, Xte, _, _ = data
        fim = FisherInformationMatrix(model.predict_proba, GemexConfig(fim_epsilon=8e-3))
        G = fim.compute(Xte[0])
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals >= -1e-9)

    def test_ricci_scalar(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        _, Xte, _, _ = data
        fim = FisherInformationMatrix(model.predict_proba, GemexConfig(fim_epsilon=8e-3))
        G = fim.compute(Xte[0])
        assert isinstance(fim.ricci_scalar(G), float)


# ── Geodesic ──────────────────────────────────────────────────────────

class TestGeodesic:
    def test_path_shape(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        from gemex.geometry.geodesic import GeodesicSolver
        Xtr, Xte, _, _ = data
        cfg = GemexConfig(n_geodesic_steps=10, fim_epsilon=8e-3)
        fim = FisherInformationMatrix(model.predict_proba, cfg)
        fim.compute(Xte[0])
        solver = GeodesicSolver(model.predict_proba, fim, cfg)
        path, arc = solver.compute_path(Xtr[0], Xte[0])
        assert path.shape == (10, 6)
        assert arc.shape  == (10,)

    def test_arc_monotone(self, model, data):
        from gemex import GemexConfig
        from gemex.manifold.fim import FisherInformationMatrix
        from gemex.geometry.geodesic import GeodesicSolver
        Xtr, Xte, _, _ = data
        cfg = GemexConfig(n_geodesic_steps=10, fim_epsilon=8e-3)
        fim = FisherInformationMatrix(model.predict_proba, cfg)
        fim.compute(Xte[0])
        solver = GeodesicSolver(model.predict_proba, fim, cfg)
        _, arc = solver.compute_path(Xtr[0], Xte[0])
        assert np.all(np.diff(arc) >= -1e-12)


# ── Result ────────────────────────────────────────────────────────────

class TestResult:
    def test_gsf_shape(self, result, feature_names):
        assert result.gsf_scores.shape == (len(feature_names),)

    def test_gsf_finite(self, result):
        assert np.all(np.isfinite(result.gsf_scores))

    def test_pti_shape(self, result, feature_names):
        d = len(feature_names)
        assert result.pti_matrix.shape == (d, d)

    def test_pti_symmetric(self, result):
        np.testing.assert_allclose(result.pti_matrix, result.pti_matrix.T, atol=1e-10)

    def test_rst_eigenvalues(self, result):
        assert len(result.rst_eigenvalues) == 3

    def test_rst_eigenvectors(self, result, feature_names):
        assert result.rst_eigenvectors.shape == (3, len(feature_names))

    def test_proba_sums_to_one(self, result):
        np.testing.assert_allclose(result.prediction_proba.sum(), 1.0, atol=1e-6)

    def test_uncertainty_level(self, result):
        assert result.uncertainty_level() in ("low", "moderate", "high")

    def test_confidence_range(self, result):
        assert 0.0 <= result.confidence_score() <= 1.0

    def test_top_features_sorted(self, result):
        top = result.top_features(4)
        scores = [abs(s) for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict_keys(self, result):
        d = result.to_dict()
        for k in ["prediction", "predicted_class", "gsf_scores",
                  "manifold_curvature", "uncertainty_level",
                  "confidence_score", "geodesic_length"]:
            assert k in d, f"Missing key: {k}"

    def test_summary_string(self, result):
        s = result.summary()
        assert isinstance(s, str) and len(s) > 50

    def test_repr(self, result):
        r = repr(result)
        assert "GemexResult" in r


# ── FAS and BTD ───────────────────────────────────────────────────────

class TestFASBTD:
    def test_fas_present(self, result):
        assert result.fas is not None
        assert "sequence" in result.fas
        assert "dwell_time" in result.fas

    def test_fas_sequence_length(self, result):
        assert len(result.fas["sequence"]) == result.geodesic_path.shape[0]

    def test_fas_dwell_sums_to_one(self, result):
        np.testing.assert_allclose(
            result.fas["dwell_time"].sum(), 1.0, atol=1e-6)

    def test_btd_present(self, result):
        assert result.btd is not None
        assert "bias_risk" in result.btd

    def test_btd_risk_levels(self, result):
        for lv in result.btd["risk_level"]:
            assert lv in ("low", "moderate", "high")


# ── Batch ─────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_length(self, explainer, data):
        Xtr, Xte, _, _ = data
        batch = explainer.explain_batch(Xte[:3], X_reference=Xtr)
        assert len(batch) == 3

    def test_all_have_gsf(self, explainer, data):
        Xtr, Xte, _, _ = data
        batch = explainer.explain_batch(Xte[:2], X_reference=Xtr)
        for r in batch:
            assert hasattr(r, "gsf_scores")
            assert np.all(np.isfinite(r.gsf_scores))
