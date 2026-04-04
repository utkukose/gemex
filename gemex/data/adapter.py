# Copyright (c) 2026 Dr. Utku Kose
# Suleyman Demirel University, Turkey | University of North Dakota, USA
# VelTech University, India | Universidad Panamericana, Mexico
# ORCID: 0000-0002-9652-6415 | utkukose@gmail.com | www.utkukose.com
# Licensed under the MIT License — see LICENSE for details.
#
# GEMEX: Geodesic Entropic Manifold Explainability v1.2.1
"""
gemex.data.adapter — Input preprocessing for tabular / timeseries / image.

Patch-based image support
--------------------------
For image data (data_type='image'), pixel features can be aggregated
into spatial patches before explanation using config.image_patch_size.

  image_patch_size = 1  (default) — pixel-level, current behaviour
  image_patch_size = 4  — 28×28 → 7×7 = 49 patch features
  image_patch_size = 2  — 28×28 → 14×14 = 196 patch features
  image_patch_size = 7  — 28×28 → 4×4 = 16 patch features (coarse)

Benefits of patch_size > 1:
  · FIM estimation is stronger (fewer features, better coverage)
  · Ricci scalar is larger and more meaningful
  · Computation ~(patch_size²)× faster
  · Spatial coherence comparable to GradCAM without conv layer access

Impact on other data types: NONE.
  Tabular and timeseries inputs are completely unaffected.
  The patch parameter is only applied when data_type='image'.

Patch attribution → pixel space:
  When patch_size > 1, GSF scores (n_patches,) are stored on the result.
  The image_trio plot and the MedMNIST example both upsample patch
  attributions back to pixel space for display via bilinear interpolation.
  All other plots (tabular, timeseries) work on their own feature vectors
  unchanged.
"""
import numpy as np


class DataAdapter:

    def __init__(self, data_type: str, config):
        self.data_type  = data_type
        self.config     = config
        # Cache spatial dimensions for later upsampling
        self._img_h     = None
        self._img_w     = None
        self._patch_sz  = getattr(config, 'image_patch_size', 1)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def flatten(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocess a single instance.

        Tabular / timeseries : flatten to 1-D float vector (unchanged).
        Image (patch_size=1) : flatten to pixel vector (unchanged).
        Image (patch_size>1) : aggregate pixels into patches, then flatten.
        """
        x = np.squeeze(x)

        if self.data_type == 'image':
            return self._image_to_patches(x)

        # Tabular and timeseries — identical to original behaviour
        return x.flatten().astype(float)

    def flatten_batch(self, X: np.ndarray) -> np.ndarray:
        """Preprocess a batch of instances."""
        return np.array([self.flatten(x) for x in X])

    def get_patch_info(self):
        """
        Return (patch_size, img_h, img_w) for use by plots that need to
        upsample patch attributions back to pixel space.
        Returns (1, None, None) for non-image data or pixel-level image.
        """
        return self._patch_sz, self._img_h, self._img_w

    def patches_to_pixels(self, x_patch: np.ndarray) -> np.ndarray:
        """
        Upsample patch feature vector back to pixel space.
        Used by the predict_fn wrapper so the real model always
        receives the pixel-level input it was trained on.

        patch vector (n_patches,) → pixel vector (H*W,)

        With patch_size=1 this is a no-op.
        """
        ps = self._patch_sz
        if ps == 1:
            return x_patch
        h = self._img_h or 28
        w = self._img_w or 28
        n_ph = int(np.ceil(h / ps))
        n_pw = int(np.ceil(w / ps))
        # Reshape patches to 2D grid
        patch_grid = x_patch.reshape(n_ph, n_pw)
        # Repeat each patch value ps times in each spatial dimension
        pixel_grid = np.repeat(np.repeat(patch_grid, ps, axis=0),
                               ps, axis=1)
        # Crop to original image size
        pixel_grid = pixel_grid[:h, :w]
        return pixel_grid.flatten()

    # ------------------------------------------------------------------ #
    #  Image → patch aggregation                                           #
    # ------------------------------------------------------------------ #

    def _image_to_patches(self, x: np.ndarray) -> np.ndarray:
        """
        Convert an image array to a patch feature vector.

        Handles input shapes:
          (H, W)       — greyscale
          (H, W, C)    — RGB / multi-channel
          (H*W,)       — already flat greyscale
          (H*W*C,)     — already flat RGB

        With patch_size=1 the output is identical to a simple flatten.
        With patch_size=p the image is divided into (H/p × W/p) patches
        and each patch is represented by its mean pixel value.

        The spatial dimensions are cached so the image_trio plot can
        upsample patch attributions back to the original pixel grid.
        """
        ps = self._patch_sz

        # ── Resolve spatial dimensions ────────────────────────────────
        if x.ndim == 1:
            # Flat input — infer spatial dims
            n = len(x)
            if n == 784:    h, w = 28, 28
            elif n == 1024: h, w = 32, 32
            elif n == 4096: h, w = 64, 64
            else:
                side = int(np.sqrt(n))
                h, w = side, side
            x_2d = x.reshape(h, w)
        elif x.ndim == 2:
            h, w   = x.shape
            x_2d   = x
        elif x.ndim == 3:
            h, w   = x.shape[0], x.shape[1]
            # Convert multi-channel to greyscale for FIM estimation
            if x.shape[2] == 3:
                x_2d = (0.299*x[:,:,0] + 0.587*x[:,:,1] +
                        0.114*x[:,:,2]).astype(float)
            else:
                x_2d = x[:,:,0].astype(float)
        else:
            return x.flatten().astype(float)

        # Cache for upsampling later
        self._img_h = h
        self._img_w = w

        if ps == 1:
            # Pixel-level — no aggregation, identical to original
            return x_2d.flatten().astype(float)

        # ── Patch aggregation ─────────────────────────────────────────
        # Ensure dimensions are divisible by patch_size (pad if needed)
        ph = int(np.ceil(h / ps)) * ps
        pw = int(np.ceil(w / ps)) * ps
        if ph != h or pw != w:
            x_pad = np.zeros((ph, pw), dtype=float)
            x_pad[:h, :w] = x_2d
        else:
            x_pad = x_2d.astype(float)

        # Compute mean of each ps×ps block
        n_ph   = ph // ps
        n_pw   = pw // ps
        patches = np.zeros((n_ph, n_pw), dtype=float)
        for i in range(n_ph):
            for j in range(n_pw):
                patches[i, j] = x_pad[i*ps:(i+1)*ps,
                                       j*ps:(j+1)*ps].mean()

        return patches.flatten()

    # ------------------------------------------------------------------ #
    #  Legacy downsampling (kept for backward compatibility)               #
    # ------------------------------------------------------------------ #

    def _downsample(self, img, target):
        """Original pixel downsampler — kept for backward compatibility."""
        h, w, c = img.shape
        fh, fw  = max(1, h//target), max(1, w//target)
        nh, nw  = h//fh, w//fw
        out = np.zeros((nh, nw, c))
        for ch in range(c):
            for i in range(nh):
                for j in range(nw):
                    out[i,j,ch] = img[i*fh:(i+1)*fh,
                                      j*fw:(j+1)*fw, ch].mean()
        return out
