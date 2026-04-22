"""
Configuration for the Environmental Monitoring System.
Only the ROI config is required from the user — everything else is automatic.
"""

# ============================================================
# GOOGLE EARTH ENGINE PROJECT
# Find this at: https://code.earthengine.google.com/
# It looks like: "ee-yourname" or "your-gcloud-project-id"
# ============================================================
GEE_PROJECT = "ee-pratyushyashkumar04"   # <-- CHANGE THIS

# ============================================================
# USER INPUT — This is the ONLY thing the user needs to change
# ============================================================
ROI_CONFIG = {
    "type": "place",       # "place" | "bbox" | "polygon"
    "value": "Bangalore, India"
}

# ============================================================
# TRAINING CITIES — Multi-country training for global generalization
# Diverse climates, vegetation, and industrial patterns.
# Add or remove cities as needed. The pipeline geocodes any place name.
# ============================================================
TRAINING_CITIES = [
    # ── India (tropical / subtropical) ──
    "Chennai, India",
    "Mumbai, India",
    "Delhi, India",

    # ── East & Southeast Asia ──
    "Jakarta, Indonesia",

    # ── Europe (temperate) ──
    "Essen, Germany",             # Ruhr industrial belt
    "Manchester, United Kingdom",

    # ── Middle East (arid) ──
    "Jubail, Saudi Arabia",       # Massive industrial city

    # ── Africa ──

    # ── Americas ──
    "São Paulo, Brazil",
]

# ============================================================
# DATE RANGES for Sentinel-2 imagery
# t1 = baseline period, t2 = current/comparison period
# ============================================================
T1_START = "2023-01-01"
T1_END   = "2023-03-31"
T2_START = "2024-01-01"
T2_END   = "2024-03-31"

# ============================================================
# NDVI THRESHOLDS
# ============================================================
NDVI_VEG_THRESHOLD    = 0.3   # NDVI > this → vegetation
NDVI_CHANGE_THRESHOLD = 0.2   # |NDVI_t1 - NDVI_t2| > this → change

# ============================================================
# VIOLATION RULES
# ============================================================
VEG_VIOLATION_THRESHOLD = 0.3  # vegetation % < this → violation

# ============================================================
# PATCH EXTRACTION
# ============================================================
PATCH_SIZE   = 256
PATCH_STRIDE = 128

# ============================================================
# TRAINING CONFIG
# ============================================================
BATCH_SIZE     = 1
EPOCHS         = 50
LEARNING_RATE  = 1e-4
VAL_SPLIT      = 0.2

# Mixed precision (float16) — halves GPU memory usage but can cause
# crashes (SIGSEGV) on some TensorFlow + GPU combinations.
# Set to True only if you've verified it works on your hardware.
USE_MIXED_PRECISION = False

# ============================================================
# MODEL SAVE PATHS
# ============================================================
VEG_MODEL_PATH    = "models/saved/vegetation_model.h5"
CHANGE_MODEL_PATH = "models/saved/change_model.h5"

# ============================================================
# OUTPUT PATHS
# ============================================================
REPORT_DIR = "outputs/reports"
MAP_DIR    = "outputs/maps"

# ============================================================
# DATA CACHE — Downloaded satellite images and OSM polygons are
# stored here so they don't need to be re-downloaded every run.
# Delete this folder to force a fresh download.
# ============================================================
DATA_CACHE_DIR = "data/cache"

# ============================================================
# OSM SETTINGS
# ============================================================
# Minimum industrial polygon area in m² (EPSG:3857).
# 10,000 m² = 1 hectare — filters out tiny OSM tags like individual buildings.
MIN_POLYGON_AREA = 10000  # m²

# Minimum pixels (H*W) after clipping a polygon — skip if smaller.
# 64×64 = 4,096 pixels ≈ 640m × 640m at 10m/pixel.
# Lowered from 128×128 to capture more industrial zones —
# the model's pad_image() handles sub-256 inputs gracefully.
MIN_CLIP_PIXELS = 64 * 64

# Maximum fraction of water pixels allowed in a crop.
# Crops with more than 50% water (detected via NDWI) are skipped.
MAX_WATER_RATIO = 0.5

# ============================================================
# SENTINEL-2 BANDS (B2=Blue, B3=Green, B4=Red, B8=NIR)
# ============================================================
S2_BANDS = ["B2", "B3", "B4", "B8"]
S2_SCALE = 10  # meters per pixel
