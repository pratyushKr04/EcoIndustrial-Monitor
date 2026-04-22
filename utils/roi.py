"""
ROI Module — Convert user input into a Shapely polygon.

Supports three input types:
  - "place"   : Place name geocoded via Nominatim (e.g., "Chennai, India")
  - "bbox"    : Bounding box [minx, miny, maxx, maxy]
  - "polygon" : List of (lon, lat) coordinate tuples
"""

import geopandas as gpd
from shapely.geometry import box, Polygon
from geopy.geocoders import Nominatim
import osmnx as ox


def get_roi_polygon(config: dict) -> gpd.GeoDataFrame:
    """
    Convert a user-provided ROI configuration into a GeoDataFrame
    containing a single polygon.

    Parameters
    ----------
    config : dict
        Must contain:
          - "type"  : one of "place", "bbox", "polygon"
          - "value" : depends on type

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with one polygon row, CRS EPSG:4326.
    """
    roi_type = config["type"].lower()
    value = config["value"]

    if roi_type == "place":
        polygon = _geocode_place(value)
    elif roi_type == "bbox":
        polygon = _bbox_to_polygon(value)
    elif roi_type == "polygon":
        polygon = _coords_to_polygon(value)
    else:
        raise ValueError(
            f"Unknown ROI type '{roi_type}'. Must be 'place', 'bbox', or 'polygon'."
        )

    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    print(f"[ROI] Resolved ROI polygon -- bounds: {polygon.bounds}")
    return gdf


def _geocode_place(place_name: str) -> Polygon:
    """Geocode a place name to a polygon using osmnx."""
    try:
        gdf = ox.geocode_to_gdf(place_name)
        polygon = gdf.geometry.iloc[0]
        if polygon.geom_type == "Point":
            # Buffer point to a small region (~5 km)
            polygon = polygon.buffer(0.05)
        print(f"[ROI] Geocoded '{place_name}' -> {polygon.geom_type}")
        return polygon
    except Exception as e:
        print(f"[ROI] osmnx geocode failed, falling back to geopy: {e}")
        return _geocode_place_fallback(place_name)


def _geocode_place_fallback(place_name: str) -> Polygon:
    """Fallback geocoding using geopy Nominatim."""
    geolocator = Nominatim(user_agent="env_monitor_system")
    location = geolocator.geocode(place_name, geometry="wkt")

    if location is None:
        raise ValueError(f"Could not geocode '{place_name}'.")

    # If bounding box is available, use it
    if hasattr(location, "raw") and "boundingbox" in location.raw:
        bb = location.raw["boundingbox"]
        # Nominatim returns [south, north, west, east]
        south, north, west, east = [float(x) for x in bb]
        polygon = box(west, south, east, north)
    else:
        # Buffer around point
        polygon = box(
            location.longitude - 0.05,
            location.latitude - 0.05,
            location.longitude + 0.05,
            location.latitude + 0.05,
        )

    print(f"[ROI] Fallback geocoded '{place_name}' -> bounds: {polygon.bounds}")
    return polygon


def _bbox_to_polygon(bbox: list) -> Polygon:
    """Convert [minx, miny, maxx, maxy] to a Shapely box polygon."""
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 values: [minx, miny, maxx, maxy]")
    return box(*bbox)


def _coords_to_polygon(coords: list) -> Polygon:
    """Convert a list of (lon, lat) tuples to a Shapely Polygon."""
    if len(coords) < 3:
        raise ValueError("Polygon must have at least 3 coordinate pairs.")
    return Polygon(coords)
