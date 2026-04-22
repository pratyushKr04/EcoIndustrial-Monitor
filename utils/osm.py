"""
OSM Module — Fetch industrial polygons from OpenStreetMap.

NO file input. NO hardcoded GeoJSON. Everything is fetched live via osmnx.
"""

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MIN_POLYGON_AREA


def fetch_industrial_polygons(roi_polygon, min_area: float = None) -> gpd.GeoDataFrame:
    """
    Fetch industrial land-use and building polygons from OpenStreetMap
    within the given ROI polygon.

    Parameters
    ----------
    roi_polygon : shapely.geometry.Polygon
        The Region of Interest polygon.
    min_area : float, optional
        Minimum polygon area in degrees² to keep. Defaults to config value.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of industrial polygons.
    """
    if min_area is None:
        min_area = MIN_POLYGON_AREA

    tags = {
        "landuse": "industrial",
        "building": "industrial",
    }

    print("[OSM] Fetching industrial polygons from OpenStreetMap...")

    try:
        gdf = ox.features_from_polygon(roi_polygon, tags)
    except Exception as e:
        print(f"[OSM] Warning: Could not fetch OSM features: {e}")
        print("[OSM] Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if gdf.empty:
        print("[OSM] No industrial features found in the ROI.")
        return gdf

    # Keep only Polygon and MultiPolygon geometries
    valid_types = ["Polygon", "MultiPolygon"]
    gdf = gdf[gdf.geometry.type.isin(valid_types)].copy()

    if gdf.empty:
        print("[OSM] No polygon geometries after filtering.")
        return gdf

    # Filter by minimum area — reproject to metric CRS first for accuracy
    gdf_metric = gdf.to_crs(epsg=3857)
    gdf = gdf[gdf_metric.geometry.area > min_area].copy()

    # Reset index for clean iteration
    gdf = gdf.reset_index(drop=True)

    print(f"[OSM] Found {len(gdf)} industrial polygons.")
    return gdf


def get_industrial_polygons_for_roi(roi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convenience wrapper: extract the polygon from a ROI GeoDataFrame
    and fetch industrial polygons.

    Parameters
    ----------
    roi_gdf : gpd.GeoDataFrame
        GeoDataFrame with at least one polygon (from roi.py).

    Returns
    -------
    gpd.GeoDataFrame
        Industrial polygons.
    """
    roi_polygon = roi_gdf.geometry.iloc[0]
    return fetch_industrial_polygons(roi_polygon)
