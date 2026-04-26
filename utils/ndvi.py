"""
NDVI Module — Compute NDVI and generate vegetation/change masks.

These masks serve as automatically-generated labels for model training
(weakly supervised — no manual annotation required).
"""

import numpy as np


def compute_ndvi(image: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red + epsilon)

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W, 4) with bands [B2, B3, B4, B8].
        Band order: Blue(0), Green(1), Red(2), NIR(3).

    Returns
    -------
    np.ndarray
        NDVI array of shape (H, W), values in [-1, 1].
    """
    red = image[..., 2].astype(np.float32)
    nir = image[..., 3].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi


def vegetation_mask(ndvi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Create a binary vegetation mask from NDVI values.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array of shape (H, W).
    threshold : float
        NDVI threshold above which a pixel is classified as vegetation.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W), dtype uint8.
        1 = vegetation, 0 = non-vegetation.
    """
    return (ndvi > threshold).astype(np.uint8)


def change_mask(ndvi_t1: np.ndarray, ndvi_t2: np.ndarray,
                threshold: float = 0.2) -> np.ndarray:
    """
    Create a binary change mask from two NDVI arrays.

    Pixels where |NDVI_t1 - NDVI_t2| > threshold are marked as changed.

    Parameters
    ----------
    ndvi_t1 : np.ndarray
        NDVI array at time t1 (H, W).
    ndvi_t2 : np.ndarray
        NDVI array at time t2 (H, W).
    threshold : float
        Absolute difference threshold for change detection.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W), dtype uint8.
        1 = change detected, 0 = no change.
    """
    diff = np.abs(ndvi_t1.astype(np.float32) - ndvi_t2.astype(np.float32))
    return (diff > threshold).astype(np.uint8)


def compute_vegetation_percentage(veg_mask: np.ndarray,
                                   valid_mask: np.ndarray = None) -> float:
    """
    Compute the percentage of vegetation pixels in a mask.

    Parameters
    ----------
    veg_mask : np.ndarray
        Binary vegetation mask (H, W). 1 = vegetation, 0 = non-vegetation.
    valid_mask : np.ndarray, optional
        Binary mask (H, W) indicating which pixels are inside the actual
        polygon boundary. If provided, only those pixels are counted.
        This prevents masked-out (black) regions from inflating the
        denominator and artificially lowering the vegetation percentage.

    Returns
    -------
    float
        Vegetation percentage in [0, 1].
    """
    if veg_mask.size == 0:
        return 0.0

    if valid_mask is not None:
        total_valid = float(valid_mask.sum())
        if total_valid == 0:
            return 0.0
        # Only count vegetation pixels that are also inside the polygon
        veg_inside = float((veg_mask * valid_mask).sum())
        return veg_inside / total_valid

    return float(veg_mask.sum()) / float(veg_mask.size)


def compute_change_area(chg_mask: np.ndarray, pixel_area_m2: float = 100.0) -> float:
    """
    Compute the total area of change in square meters.

    Parameters
    ----------
    chg_mask : np.ndarray
        Binary change mask.
    pixel_area_m2 : float
        Area of a single pixel in m² (default 100 for 10m Sentinel-2).

    Returns
    -------
    float
        Total change area in m².
    """
    return float(chg_mask.sum()) * pixel_area_m2


def compute_ndwi(image: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR + epsilon)

    Water: NDWI > 0  |  Land/Vegetation: NDWI < 0

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W, 4) with bands [B2, B3, B4, B8].
        Band order: Blue(0), Green(1), Red(2), NIR(3).

    Returns
    -------
    np.ndarray
        NDWI array of shape (H, W), values in [-1, 1].
    """
    green = image[..., 1].astype(np.float32)
    nir   = image[..., 3].astype(np.float32)

    ndwi = (green - nir) / (green + nir + 1e-6)
    return np.clip(ndwi, -1.0, 1.0)


def is_mostly_water(image: np.ndarray, water_threshold: float = 0.0,
                    max_water_ratio: float = 0.5) -> bool:
    """
    Check if an image crop is dominated by water pixels.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, 4).
    water_threshold : float
        NDWI above this is classified as water (default 0.0).
    max_water_ratio : float
        If more than this fraction of pixels are water, return True.

    Returns
    -------
    bool
        True if the crop is mostly water.
    """
    if image.size == 0:
        return True

    ndwi = compute_ndwi(image)
    water_pixels = np.sum(ndwi > water_threshold)
    total_pixels = ndwi.size

    water_ratio = water_pixels / total_pixels
    return water_ratio > max_water_ratio
