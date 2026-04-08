# Changelog

All notable changes to GEMEX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.2] — 2026

### Added

- Added four new examples.
- Added new plot screenshots.

### Fixed

- Fixed some plot wiring.

### Changed

- Folder renamed from `gemex-1.2.1` to `gemex-1.2.2`
- Updated version info texts in the related GEMEX files.
- Updated readme considering new additions and explanations.

---

## [1.2.1] — 2026

### Fixed

- Removed some unnecessary codes in the examples.
- Fixed README GitHub image URLs for visibility in PyPI.

### Changed

- Folder renamed from `gemex-1.2.0` to `gemex-1.2.1`
- Updated version info texts in the related GEMEX files.

---

## [1.2.0] — 2026

### Added

Initial release of GEMEX.

**Patch-based image explanation**
- New `image_patch_size` config parameter (default=1, backward compatible)
- Setting `image_patch_size=4` on 28×28 images produces 7×7=49 patch features:
  ~6× faster FIM estimation, Ricci scalar improves from ~0.02 to ~0.35–0.52
- `DataAdapter._image_to_patches()` handles greyscale/RGB/flat inputs
- `DataAdapter.patches_to_pixels()` upsamples patch vectors to pixel space
- Patch→pixel predict wrapper in `Explainer.__init__` — model calls remain
  in pixel space while all geometry operates in patch space
- `image_trio` plot updated to use real upsampled GSF when available
- Tabular and timeseries data completely unaffected by this setting

**MedMNIST comparative study**
- Five standalone scripts: `09_pneumoniamnist.py`, `10_pathmnist.py`,
  `11_dermamnist.py`, `12_organamnist.py`, `13_bloodmnist.py`
- MedCNN architecture: ResNet-inspired, BatchNorm, Dropout,
  CosineAnnealingLR, 20 epochs
- Methods: GEMEX GeodesicCAM, GradCAM, GradCAM++, Vanilla Saliency
- Metrics: rank correlation, top-10% pixel precision, deletion AUC, Ricci scalar

**New example scripts**
- `01_tabular_heart_diabetes.py` — real UCI medical data, all GEMEX plots
- `02_comparative_study.py` — GEMEX vs SHAP vs LIME vs ELI5 with
  metric tables, radar charts, attribution comparison, summary heatmap
- `03_timeseries_ecg_har.py` — ECG5000 and UCI HAR with temporal
  attribution profiles and Ricci scalar by class

### Fixed

**Enhancement 1 — Numerically stable sign assignment** (`gsf.py`)
- Saturated predictions (`|p(x)−0.5| > 0.48`): use raw prediction-difference
  direction instead of log-likelihood derivative (avoids blow-up at boundaries)
- Other cases: clip probabilities to `[0.01, 0.99]` before log computation
- Root cause: previous sign integral was unbounded near probability = 0 or 1,
  causing GSF sums to invert sign relative to `f(x)−f(baseline)`

**Enhancement 2 — Confidence-adaptive path weighting** (`gsf.py`)
- Integration weight `w(t) = (1−conf) + conf × exp(−3t)` where
  `conf = |p(x)−0.5| × 2`
- Emphasises early path steps for confident predictions where the model is
  already saturated near the endpoint
- Weights normalised so geodesic integral remains unit-preserving

**Enhancement 3 — Increased neighbourhood defaults** (`config.py`)
- `fim_local_n` default increased from 8 to 16
- `fim_local_sigma` default increased from 0.05 to 0.10
- Improves FIM quality on smooth models with no meaningful speed cost

**DermaMNIST DataClass bug** (`11_dermamnist.py`)
- Fixed `getattr(medmnist, 'dermamnist'.capitalize().replace('mnist','MNIST'))`
  producing `'Dermamnist'` instead of `'DermaMNIST'`
- Replaced with explicit name mapping for all five MedMNIST datasets

**GradCAM tensor detach** (`09–13_*.py`)
- Added `.detach()` before `.cpu().numpy()` on activation and gradient tensors
- Prevented `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Size mismatch in plot_quantitative** (`09–13_*.py`)
- GEMEX stores 49-patch scores; GradCAM stores 784-pixel scores
- All comparison functions now call `_to_28x28()` to normalise to 784 pixels
  before spearmanr, top-k precision, and deletion AUC calculations

### Changed

- Folder renamed from `gemex-1.0.0` to `gemex-1.2.0`
- Removed obsolete examples: `03_pytorch_model.py`, `05_image_explanation.py`,
  `08_medmnist_gemex_vs_gradcam.py` (superseded by per-dataset scripts)
- MEDMNIST_DATASETS registry in each dataset script reduced to single entry
  (no longer contains all five datasets)

---

## [1.1.0] — 2026

### Added

**Adaptive epsilon FIM estimation** (`fim.py`, `config.py`)
- Auto-detection tests `[base, 5×, 10×, 50×, 100×]` epsilon values
- Tree models start at 10× base epsilon
- `fim_quality()` method returning 'good'/'marginal'/'poor'

**Local neighbourhood averaging** (`fim.py`)
- 16 random perturbations on sphere radius 0.10 around x
- 60% local / 40% reference weighting
- Prevents zero-FIM failure inside tree leaves

**Model-type routing** (`explainer.py`, `config.py`)
- Auto-detects 'tree' vs 'smooth' from class name
- Tree keywords: forest, boost, tree, xgb, lgbm, catboost, gradient, extra

**Full-path sign assignment** (`gsf.py`, Fix 3)
- Replaces midpoint heuristic with full-path log-likelihood integral
- Samples every other step for efficiency

**Curvature-weighted uncertainty** (`gsf.py`, Fix 4)
- Per-feature confidence bands weighted by Ricci scalar along path
- Stored on `GemexResult.gsf_uncertainty`

**GSF normalisation** (`config.py`, Fix 5)
- Optional `gsf_normalise=True` forces `sum(GSF) = f(x) − f(baseline)`

### Fixed

**RK4 geodesic integrator** (`geodesic.py`, Fix 1)
- Replaced Euler integrator with RK4
- Damped Christoffel acceleration (clip=0.20)
- Velocity renormalisation and NaN guards

**Kernel-smoothed FIM** (`fim.py`, Fix 2)
- Silverman bandwidth estimation
- LRU cache (32 entries) for repeated FIM queries

---

## [1.0.0] — 2026

### Added

Early draft release of GEMEX.

**Core geometric objects**
- FIM (Fisher Information Matrix) estimation
- GSF (Geodesic Sensitivity Field) attribution
- PTI (Parallel Transport Interaction) pairwise holonomy
- RCT (Riemannian Curvature Triplet) three-way interactions
- RST (Riemannian Saliency Tensor) eigendecomposition
- FAS (Feature Attention Sequence) geodesic dwell time
- BTD (Bias Trap Detector)

**13 visualisation types** with dark/light themes

**Data type support**: tabular, timeseries, image

**Documentation**: README, CITATION.cff, LICENSE, pyproject.toml
