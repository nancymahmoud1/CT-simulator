"""
core/projection.py
------------------
Forward projection (Radon transform) — sinogram computation.
No Qt dependencies.
"""

import numpy as np
from skimage.transform import radon


# def compute_sinogram(phantom: np.ndarray, theta: np.ndarray) -> np.ndarray:
#     """
#     Compute the Radon-transform sinogram of a 2-D phantom.

#     Parameters
#     ----------
#     phantom : np.ndarray, shape (H, W)
#         Normalized 2-D image.
#     theta : np.ndarray
#         Projection angles in degrees (e.g. np.linspace(0, 180, n_angles)).

#     Returns
#     -------
#     sinogram : np.ndarray, shape (n_detectors, n_angles)
#         Clean sinogram.
#     """
    # return radon(phantom, theta=theta)
import numpy as np


def rotate_image(image, angle):
    """
    Rotate image using backward mapping (nearest neighbor)
    """
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    h, w = image.shape
    cx, cy = w // 2, h // 2

    rotated = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            # shift to center
            x_shift = x - cx
            y_shift = y - cy

            # inverse rotation
            x_orig = cos_a * x_shift + sin_a * y_shift
            y_orig = -sin_a * x_shift + cos_a * y_shift

            x_orig += cx
            y_orig += cy

            # nearest neighbor
            x_orig = int(round(x_orig))
            y_orig = int(round(y_orig))

            if 0 <= x_orig < w and 0 <= y_orig < h:
                rotated[y, x] = image[y_orig, x_orig]

    return rotated


def compute_sinogram(phantom, theta):
    """
    Compute sinogram without using skimage
    """
    h, w = phantom.shape
    sinogram = []

    for angle in theta:
        # 1. rotate image
        rotated = rotate_image(phantom, angle)

        # 2. sum along columns
        projection = np.sum(rotated, axis=0)

        sinogram.append(projection)

    # convert to array (detectors × angles)
    sinogram = np.array(sinogram).T

    return sinogram