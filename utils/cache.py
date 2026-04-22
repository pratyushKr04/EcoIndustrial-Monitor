"""
Cache Module — Save/load satellite images and OSM polygons locally.

Cache layout:
  data/cache/
    satellite/
      chennai_india_2024-01-01_2024-03-31.npy
      mumbai_india_2023-01-01_2023-03-31.npy
    osm/
      chennai_india_industrial.gpkg
      mumbai_india_industrial.gpkg
    roi/
      chennai_india_bounds.npy

Calling `delete data/cache/` forces a complete re-download.
"""

import os
import re
import numpy as np
import geopandas as gpd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CACHE_DIR


def _sanitize(name: str) -> str:
    """Turn a city name like 'São Paulo, Brazil' into 'sao_paulo_brazil'."""
    name = name.lower().strip()
    # Replace accented chars with ASCII equivalents
    replacements = {
        "ã": "a", "á": "a", "à": "a", "â": "a",
        "é": "e", "è": "e", "ê": "e",
        "í": "i", "ì": "i", "î": "i",
        "ó": "o", "ò": "o", "ô": "o", "õ": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ñ": "n", "ç": "c",
    }
    for orig, repl in replacements.items():
        name = name.replace(orig, repl)
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


# ── Satellite image cache ──────────────────────────────────────

def get_satellite_cache_path(city_name: str, start_date: str, end_date: str) -> str:
    """Return the .npy cache file path for a satellite image."""
    folder = os.path.join(DATA_CACHE_DIR, "satellite")
    os.makedirs(folder, exist_ok=True)
    key = f"{_sanitize(city_name)}_{start_date}_{end_date}"
    return os.path.join(folder, f"{key}.npy")


def load_satellite_cache(city_name: str, start_date: str, end_date: str):
    """Load a cached satellite image if it exists, else return None."""
    path = get_satellite_cache_path(city_name, start_date, end_date)
    if os.path.exists(path):
        print(f"[CACHE] Loading satellite image from cache: {os.path.basename(path)}")
        return np.load(path)
    return None


def save_satellite_cache(image: np.ndarray, city_name: str,
                         start_date: str, end_date: str):
    """Save a satellite image to the cache."""
    path = get_satellite_cache_path(city_name, start_date, end_date)
    np.save(path, image)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[CACHE] Saved satellite image: {os.path.basename(path)} ({size_mb:.1f} MB)")


# ── OSM polygon cache ─────────────────────────────────────────

def get_osm_cache_path(city_name: str) -> str:
    """Return the .gpkg cache file path for OSM polygons."""
    folder = os.path.join(DATA_CACHE_DIR, "osm")
    os.makedirs(folder, exist_ok=True)
    key = _sanitize(city_name)
    return os.path.join(folder, f"{key}_industrial.gpkg")


def load_osm_cache(city_name: str):
    """Load cached OSM polygons if they exist, else return None."""
    path = get_osm_cache_path(city_name)
    if os.path.exists(path):
        print(f"[CACHE] Loading OSM polygons from cache: {os.path.basename(path)}")
        gdf = gpd.read_file(path)
        if not gdf.empty:
            return gdf
    return None


def save_osm_cache(gdf: gpd.GeoDataFrame, city_name: str):
    """Save OSM polygons to the cache."""
    if gdf.empty:
        return
    path = get_osm_cache_path(city_name)
    # Keep only geometry column to avoid serialization issues
    gdf_clean = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    gdf_clean.to_file(path, driver="GPKG")
    print(f"[CACHE] Saved OSM polygons: {os.path.basename(path)} ({len(gdf)} polygons)")


# ── ROI bounds cache ──────────────────────────────────────────

def get_roi_cache_path(city_name: str) -> str:
    """Return cache path for ROI bounds."""
    folder = os.path.join(DATA_CACHE_DIR, "roi")
    os.makedirs(folder, exist_ok=True)
    key = _sanitize(city_name)
    return os.path.join(folder, f"{key}_bounds.npy")


def load_roi_bounds_cache(city_name: str):
    """Load cached ROI bounds (minx, miny, maxx, maxy) if they exist."""
    path = get_roi_cache_path(city_name)
    if os.path.exists(path):
        return tuple(np.load(path))
    return None


def save_roi_bounds_cache(bounds: tuple, city_name: str):
    """Save ROI bounds to cache."""
    path = get_roi_cache_path(city_name)
    np.save(path, np.array(bounds))


# ── Cache summary ─────────────────────────────────────────────

def print_cache_summary():
    """Print a summary of what's currently cached."""
    if not os.path.exists(DATA_CACHE_DIR):
        print("[CACHE] No cache directory found.")
        return

    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(DATA_CACHE_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            total_size += os.path.getsize(fpath)
            file_count += 1

    total_mb = total_size / (1024 * 1024)
    print(f"[CACHE] Cache: {file_count} files, {total_mb:.1f} MB total")
    print(f"[CACHE] Location: {os.path.abspath(DATA_CACHE_DIR)}")
    print(f"[CACHE] Delete this folder to force re-download.")
