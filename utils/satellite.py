"""
Satellite Module — Fetch Sentinel-2 imagery.

Option A (Primary): Google Earth Engine
Option B (Fallback): Load local GeoTIFF files
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S2_BANDS, S2_SCALE, GEE_PROJECT


def initialize_gee():
    """Initialize Google Earth Engine with the configured project ID."""
    try:
        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
        except Exception:
            print("[SAT] Trying GEE authentication flow...")
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
        print("[SAT] Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"[SAT] GEE initialization failed: {e}")
        print(f"[SAT] Make sure GEE_PROJECT in config.py is set correctly.")
        print(f"[SAT] Your project ID looks like: 'ee-yourname' or 'your-gcp-project'")
        print(f"[SAT] Find it at: https://code.earthengine.google.com/")
        return False


def fetch_sentinel_image(roi_polygon, start_date: str, end_date: str) -> np.ndarray:
    """
    Fetch a Sentinel-2 median composite from Google Earth Engine.

    Parameters
    ----------
    roi_polygon : shapely.geometry.Polygon
        The Region of Interest.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 4) with bands [B2, B3, B4, B8],
        values scaled to [0, 1].
    """
    import ee
    import geemap

    # Convert shapely polygon to ee.Geometry
    coords = list(roi_polygon.exterior.coords)
    ee_roi = ee.Geometry.Polygon([[[lon, lat] for lon, lat in coords]])

    print(f"[SAT] Fetching Sentinel-2 images from {start_date} to {end_date}...")

    # Load Sentinel-2 Surface Reflectance collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    count = collection.size().getInfo()
    print(f"[SAT] Found {count} images after cloud filtering.")

    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 images found for the given ROI and date range "
            f"({start_date} to {end_date}). Try a wider date range or different ROI."
        )

    # Create median composite and select bands
    composite = collection.median().select(S2_BANDS)

    # Clip to ROI
    composite = composite.clip(ee_roi)

    # Download in tiles to stay under GEE's 50MB request limit
    print("[SAT] Downloading image in tiles (GEE 50MB limit workaround)...")
    image_array = _download_tiled(composite, ee_roi, roi_polygon)

    if image_array is None or image_array.size == 0:
        raise RuntimeError("[SAT] Failed to download image from GEE.")

    # Handle potential NaN values
    image_array = np.nan_to_num(image_array, nan=0.0)

    # Scale reflectance values to [0, 1] (Sentinel-2 SR values are 0–10000)
    image_array = image_array.astype(np.float32) / 10000.0
    image_array = np.clip(image_array, 0.0, 1.0)

    print(f"[SAT] Downloaded image shape: {image_array.shape}")
    return image_array


def _download_tiled(composite, ee_roi, roi_polygon,
                    tile_deg: float = 0.08) -> np.ndarray:
    """
    Download a GEE image by splitting the ROI into tiles and stitching.

    tile_deg = 0.08 degrees ≈ ~9km at equator → ~16MB per tile at 10m res.
    Well under GEE's 50MB limit.

    Parameters
    ----------
    composite : ee.Image
        The GEE image to download.
    ee_roi : ee.Geometry
        The full ROI geometry.
    roi_polygon : shapely.geometry.Polygon
        Shapely polygon of the ROI (for bounds).
    tile_deg : float
        Tile size in degrees.

    Returns
    -------
    np.ndarray
        Stitched image array (H, W, C).
    """
    import ee
    import geemap

    minx, miny, maxx, maxy = roi_polygon.bounds

    # Build tile grid
    xs = list(_frange(minx, maxx, tile_deg))
    ys = list(_frange(miny, maxy, tile_deg))

    total_tiles = len(xs) * len(ys)
    print(f"[SAT] Splitting into {total_tiles} tiles ({len(xs)} cols × {len(ys)} rows)...")

    row_arrays = []

    for row_idx, y0 in enumerate(ys):
        y1 = min(y0 + tile_deg, maxy)
        col_arrays = []

        for col_idx, x0 in enumerate(xs):
            x1 = min(x0 + tile_deg, maxx)

            tile_num = row_idx * len(xs) + col_idx + 1
            print(f"[SAT]   Tile {tile_num}/{total_tiles}...", end="\r")

            tile_geom = ee.Geometry.Rectangle([x0, y0, x1, y1])

            try:
                tile_array = geemap.ee_to_numpy(
                    composite,
                    region=tile_geom,
                    scale=S2_SCALE,
                )
            except Exception as e:
                print(f"\n[SAT]   Tile {tile_num} failed: {e}. Filling with zeros.")
                # Estimate tile shape and fill with zeros
                tile_w = max(1, int((x1 - x0) / (S2_SCALE / 111320)))
                tile_h = max(1, int((y1 - y0) / (S2_SCALE / 111320)))
                tile_array = np.zeros((tile_h, tile_w, len(S2_BANDS)), dtype=np.float32)

            if tile_array is None:
                tile_w = max(1, int((x1 - x0) / (S2_SCALE / 111320)))
                tile_h = max(1, int((y1 - y0) / (S2_SCALE / 111320)))
                tile_array = np.zeros((tile_h, tile_w, len(S2_BANDS)), dtype=np.float32)

            col_arrays.append(tile_array)

        # Horizontally concatenate columns for this row
        # Ensure same height before concatenating
        row_h = max(t.shape[0] for t in col_arrays)
        padded_cols = []
        for t in col_arrays:
            if t.shape[0] < row_h:
                pad = np.zeros((row_h - t.shape[0], t.shape[1], t.shape[2]), dtype=t.dtype)
                t = np.concatenate([t, pad], axis=0)
            padded_cols.append(t)

        row_array = np.concatenate(padded_cols, axis=1)
        row_arrays.append(row_array)

    print()  # newline after tile progress

    # Vertically concatenate rows
    # Ensure same width before concatenating
    max_w = max(r.shape[1] for r in row_arrays)
    padded_rows = []
    for r in row_arrays:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], r.shape[2]), dtype=r.dtype)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)

    full_image = np.concatenate(padded_rows, axis=0)
    return full_image


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    val = start
    while val < stop:
        yield val
        val += step


def load_local_image(path: str) -> np.ndarray:
    """
    Load a local GeoTIFF image as a numpy array.

    Parameters
    ----------
    path : str
        Path to the .tif file.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, C).
    """
    import rasterio

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    with rasterio.open(path) as src:
        # Read all bands: shape (C, H, W)
        image = src.read()
        # Transpose to (H, W, C)
        image = np.transpose(image, (1, 2, 0)).astype(np.float32)

    # Normalize to [0, 1] if values are in Sentinel-2 range
    if image.max() > 1.0:
        image = image / 10000.0
        image = np.clip(image, 0.0, 1.0)

    print(f"[SAT] Loaded local image: {path}, shape: {image.shape}")
    return image
