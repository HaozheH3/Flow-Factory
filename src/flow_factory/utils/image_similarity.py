# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# see LICENSE or http://www.apache.org/licenses/LICENSE-2.0

"""
Image similarity helpers (SSIM), aligned with ``visual_shortcut_reward/run_similarity.py``:

grayscale SSIM via ``skimage.metrics.structural_similarity``, resizing the second image to
match the first when shapes differ (same idea as ``cv2.resize`` there, without requiring OpenCV).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .image import standardize_image_batch


def _rgb_pil_to_gray_uint8(pil_image: Image.Image) -> np.ndarray:
    return np.asarray(pil_image.convert("RGB").convert("L"), dtype=np.uint8)


def ssim_grayscale_pair(gen_gray: np.ndarray, cond_gray: np.ndarray) -> float:
    """SSIM on 2D uint8 arrays; resizes ``cond_gray`` to ``gen_gray`` shape if needed."""
    from skimage.metrics import structural_similarity as ssim

    if gen_gray.ndim != 2 or cond_gray.ndim != 2:
        raise ValueError(
            f"expected 2D grayscale arrays, got gen_gray.ndim={getattr(gen_gray, 'ndim', None)} "
            f"cond_gray.ndim={getattr(cond_gray, 'ndim', None)}"
        )
    if gen_gray.shape != cond_gray.shape:
        cond_img = Image.fromarray(cond_gray, mode="L").resize(
            (gen_gray.shape[1], gen_gray.shape[0]),
            Image.Resampling.BILINEAR,
        )
        cond_gray = np.asarray(cond_img, dtype=np.uint8)
    v, _ = ssim(gen_gray, cond_gray, full=True)
    return float(v)


def max_ssim_against_conditions(
    generated: Union[Image.Image, torch.Tensor, np.ndarray],
    condition_images: Sequence[Union[Image.Image, torch.Tensor, np.ndarray]],
) -> Tuple[float, List[float]]:
    """
    Returns:
        (max_ssim, per_condition_ssims). If ``condition_images`` is empty, returns (-1.0, []).
    """
    if condition_images is None or len(condition_images) == 0:
        return -1.0, []

    pils = standardize_image_batch(generated, output_type="pil")
    if isinstance(pils, list):
        gen_pil = pils[0]
    else:
        raise TypeError(f"expected generated image to standardize to a list of PIL, got {type(pils)}")
    gen_gray = _rgb_pil_to_gray_uint8(gen_pil)

    per: List[float] = []
    for cond in condition_images:
        c_pils = standardize_image_batch(cond, output_type="pil")
        if isinstance(c_pils, list):
            c_pil = c_pils[0]
        else:
            c_pil = c_pils
        c_gray = _rgb_pil_to_gray_uint8(c_pil)
        per.append(ssim_grayscale_pair(gen_gray, c_gray))
    return max(per), per
