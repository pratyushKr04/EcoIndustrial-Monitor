"""
Preprocessing Module — Normalization, alignment, clipping, and patch extraction.
"""

import numpy as np
from typing import Tuple, List


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range per-band.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Normalized image in [0, 1].
    """
    image = image.astype(np.float32)

    # If already in [0, 1]
    if image.max() <= 1.0:
        return image

    # Sentinel-2 reflectance: 0–10000
    if image.max() > 1.0:
        image = image / 10000.0

    return np.clip(image, 0.0, 1.0)


def align_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two images to the same spatial dimensions by cropping to
    the minimum common size.

    Parameters
    ----------
    img1 : np.ndarray
        First image (H1, W1, C).
    img2 : np.ndarray
        Second image (H2, W2, C).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aligned images with the same (H, W).
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1_aligned = img1[:h, :w, :]
    img2_aligned = img2[:h, :w, :]

    if img1.shape != img1_aligned.shape or img2.shape != img2_aligned.shape:
        print(f"[PREPROC] Aligned images from {img1.shape}/{img2.shape} → ({h}, {w}, {img1.shape[2]})")

    return img1_aligned, img2_aligned


def clip_image_to_polygon(image: np.ndarray, image_bounds: tuple,
                          polygon) -> np.ndarray:
    """
    Clip an image array to a polygon using coordinate-based masking.

    This is a simplified version that uses bounding-box clipping when
    rasterio is not available or the image is already a numpy array
    without geotransform metadata.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C).
    image_bounds : tuple
        (minx, miny, maxx, maxy) geographic bounds of the image.
    polygon : shapely.geometry.Polygon
        Polygon to clip to.

    Returns
    -------
    np.ndarray
        Clipped image region.
    """
    minx, miny, maxx, maxy = image_bounds
    h, w = image.shape[:2]

    # Compute pixel coordinates for the polygon's bounding box
    poly_bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    px_minx = int(max(0, (poly_bounds[0] - minx) / (maxx - minx) * w))
    px_maxx = int(min(w, (poly_bounds[2] - minx) / (maxx - minx) * w))
    px_miny = int(max(0, (1.0 - (poly_bounds[3] - miny) / (maxy - miny)) * h))
    px_maxy = int(min(h, (1.0 - (poly_bounds[1] - miny) / (maxy - miny)) * h))

    # Ensure valid crop region
    if px_maxx <= px_minx or px_maxy <= px_miny:
        print("[PREPROC] Warning: polygon does not overlap with image bounds.")
        return np.zeros((1, 1, image.shape[2]), dtype=image.dtype)

    clipped = image[px_miny:px_maxy, px_minx:px_maxx, :]
    return clipped


def extract_patches(image: np.ndarray, size: int = 256,
                    stride: int = 128) -> List[np.ndarray]:
    """
    Extract overlapping patches from an image using a sliding window.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C).
    size : int
        Patch size (square).
    stride : int
        Stride between patches.

    Returns
    -------
    List[np.ndarray]
        List of patches, each of shape (size, size, C).
    """
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patch = image[y:y + size, x:x + size, :]
            patches.append(patch)

    # If no patches could be extracted, pad the image and take 1 patch
    if len(patches) == 0 and h > 0 and w > 0:
        padded = pad_image(image, size)
        patches.append(padded[:size, :size, :])

    return patches


def extract_patch_pairs(img1: np.ndarray, img2: np.ndarray,
                        mask: np.ndarray, size: int = 256,
                        stride: int = 128) -> Tuple[List, List, List]:
    """
    Extract aligned patch triplets (img1_patch, img2_patch, mask_patch)
    from two images and a corresponding mask.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images of shape (H, W, C).
    mask : np.ndarray
        Mask of shape (H, W).
    size : int
        Patch size.
    stride : int
        Stride between patches.

    Returns
    -------
    Tuple of lists: (patches_1, patches_2, mask_patches)
    """
    h, w = img1.shape[:2]
    patches_1, patches_2, mask_patches = [], [], []

    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            p1 = img1[y:y + size, x:x + size, :]
            p2 = img2[y:y + size, x:x + size, :]
            pm = mask[y:y + size, x:x + size]
            patches_1.append(p1)
            patches_2.append(p2)
            mask_patches.append(pm)

    return patches_1, patches_2, mask_patches


def pad_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pad an image to at least target_size in both dimensions.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C).
    target_size : int
        Minimum size for H and W.

    Returns
    -------
    np.ndarray
        Padded image.
    """
    h, w = image.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    if pad_h == 0 and pad_w == 0:
        return image

    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

    return padded


def reconstruct_from_patches(patches: List[np.ndarray],
                              original_shape: Tuple[int, int],
                              size: int = 256,
                              stride: int = 128) -> np.ndarray:
    """
    Reconstruct an image from overlapping patches by averaging.

    Parameters
    ----------
    patches : List[np.ndarray]
        List of prediction patches (size, size, 1).
    original_shape : Tuple[int, int]
        (H, W) of the original image.
    size : int
        Patch size.
    stride : int
        Stride used during extraction.

    Returns
    -------
    np.ndarray
        Reconstructed image of shape (H, W).
    """
    h, w = original_shape
    output = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            if idx >= len(patches):
                break
            patch = patches[idx]
            if patch.ndim == 3:
                patch = patch[:, :, 0]
            output[y:y + size, x:x + size] += patch
            counts[y:y + size, x:x + size] += 1.0
            idx += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    output = output / counts

    return output
