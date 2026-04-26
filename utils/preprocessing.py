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
                          polygon, padding: float = 0.05,
                          mask_outside: bool = True) -> np.ndarray:
    """
    Clip an image array to a polygon with tight cropping and optional masking.

    Steps:
      1. Convert polygon bounds to pixel coordinates
      2. Add a small padding buffer (default 5% of polygon size)
      3. Crop the image to the padded bounding box
      4. Optionally mask (zero out) pixels that fall outside the polygon

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, C).
    image_bounds : tuple
        (minx, miny, maxx, maxy) geographic bounds of the image.
    polygon : shapely.geometry.Polygon
        Polygon to clip to.
    padding : float
        Fractional padding around the polygon bounding box (0.05 = 5%).
        Set to 0.0 for the tightest possible crop.
    mask_outside : bool
        If True, zero out pixels that fall outside the polygon boundary.
        This ensures only the industrial zone pixels are non-zero.

    Returns
    -------
    np.ndarray
        Clipped (and optionally masked) image region.
    """
    minx, miny, maxx, maxy = image_bounds
    h, w = image.shape[:2]

    # Geographic extent of the full image
    geo_w = maxx - minx
    geo_h = maxy - miny

    if geo_w <= 0 or geo_h <= 0:
        return np.zeros((1, 1, image.shape[2]), dtype=image.dtype)

    # Polygon bounding box
    pb = polygon.bounds  # (minx, miny, maxx, maxy)

    # Add padding (fraction of the polygon's own size)
    poly_w = pb[2] - pb[0]
    poly_h = pb[3] - pb[1]
    pad_x = poly_w * padding
    pad_y = poly_h * padding

    # Padded polygon bounds (clamped to image bounds)
    padded_minx = max(minx, pb[0] - pad_x)
    padded_maxx = min(maxx, pb[2] + pad_x)
    padded_miny = max(miny, pb[1] - pad_y)
    padded_maxy = min(maxy, pb[3] + pad_y)

    # Convert to pixel coordinates
    px_left  = int((padded_minx - minx) / geo_w * w)
    px_right = int((padded_maxx - minx) / geo_w * w)
    px_top   = int((1.0 - (padded_maxy - miny) / geo_h) * h)
    px_bot   = int((1.0 - (padded_miny - miny) / geo_h) * h)

    # Clamp to image dimensions
    px_left  = max(0, min(px_left, w - 1))
    px_right = max(px_left + 1, min(px_right, w))
    px_top   = max(0, min(px_top, h - 1))
    px_bot   = max(px_top + 1, min(px_bot, h))

    # Crop the image
    clipped = image[px_top:px_bot, px_left:px_right, :].copy()

    if not mask_outside:
        return clipped

    # Apply polygon mask: zero out pixels outside the actual polygon
    clip_h, clip_w = clipped.shape[:2]
    if clip_h == 0 or clip_w == 0:
        return clipped

    # Build a fast rasterized mask using vectorized coordinate grid
    # Create arrays of geographic coords for every pixel center
    xs = np.linspace(padded_minx, padded_maxx, clip_w, endpoint=False)
    xs += (padded_maxx - padded_minx) / clip_w / 2  # Center of pixel
    ys = np.linspace(padded_maxy, padded_miny, clip_h, endpoint=False)
    ys -= (padded_maxy - padded_miny) / clip_h / 2  # Center of pixel

    # Use matplotlib.path for fast vectorized point-in-polygon test
    from matplotlib.path import Path

    try:
        if polygon.geom_type == 'MultiPolygon':
            exterior_coords = list(polygon.geoms[0].exterior.coords)
        else:
            exterior_coords = list(polygon.exterior.coords)
    except Exception:
        return clipped  # Can't extract coords, skip masking

    poly_path = Path(exterior_coords)

    # Create meshgrid of all pixel coordinates
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    # Vectorized point-in-polygon test (very fast)
    mask = poly_path.contains_points(points).reshape(clip_h, clip_w)

    # If the mask caught nothing (possible edge case), skip masking
    if not mask.any():
        return clipped

    # Apply mask: zero out non-polygon pixels
    mask_3d = np.expand_dims(mask, axis=2).astype(clipped.dtype)
    clipped = clipped * mask_3d

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
