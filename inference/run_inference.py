"""
Inference Pipeline — Run trained models on industrial polygons.

Full loop: ROI → OSM polygons → Fetch images → Clip → Predict → Report.
"""

import os
import sys

# Force legacy Keras 2 — Keras 3's _cast_seed causes SIGSEGV on some GPUs.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ROI_CONFIG, T1_START, T1_END, T2_START, T2_END,
    PATCH_SIZE, PATCH_STRIDE,
    VEG_VIOLATION_THRESHOLD,
    VEG_MODEL_PATH, CHANGE_MODEL_PATH,
    REPORT_DIR, MAP_DIR, MIN_CLIP_PIXELS, MAX_WATER_RATIO,
)
from utils.roi import get_roi_polygon
from utils.osm import get_industrial_polygons_for_roi
from utils.satellite import fetch_sentinel_image, initialize_gee
from utils.ndvi import (
    compute_ndvi, vegetation_mask, change_mask,
    compute_vegetation_percentage, compute_change_area,
    is_mostly_water,
)
from utils.preprocessing import (
    normalize_image, align_images, clip_image_to_polygon,
    extract_patches, reconstruct_from_patches, pad_image,
)
from models.unet import combined_loss as veg_loss
from models.siamese_unet import combined_loss as chg_loss


def load_models():
    """Load trained vegetation and change detection models."""
    veg_model = None
    chg_model = None

    if os.path.exists(VEG_MODEL_PATH):
        print(f"[INFER] Loading vegetation model from {VEG_MODEL_PATH}")
        veg_model = tf.keras.models.load_model(
            VEG_MODEL_PATH,
            custom_objects={"combined_loss": veg_loss},
        )
    else:
        print(f"[INFER] WARNING: Vegetation model not found at {VEG_MODEL_PATH}")
        print("[INFER] Will use NDVI-based vegetation detection as fallback.")

    if os.path.exists(CHANGE_MODEL_PATH):
        print(f"[INFER] Loading change model from {CHANGE_MODEL_PATH}")
        chg_model = tf.keras.models.load_model(
            CHANGE_MODEL_PATH,
            custom_objects={"combined_loss": chg_loss},
        )
    else:
        print(f"[INFER] WARNING: Change model not found at {CHANGE_MODEL_PATH}")
        print("[INFER] Will use NDVI-based change detection as fallback.")

    return veg_model, chg_model


def predict_vegetation(model, image: np.ndarray) -> np.ndarray:
    """
    Predict vegetation mask using the U-Net model.
    Falls back to NDVI thresholding if model is None.

    Parameters
    ----------
    model : tf.keras.Model or None
    image : np.ndarray (H, W, 4)

    Returns
    -------
    np.ndarray (H, W) binary mask
    """
    if model is None:
        # Fallback: NDVI-based
        ndvi = compute_ndvi(image)
        return vegetation_mask(ndvi)

    original_h, original_w = image.shape[:2]

    # Pad and extract patches
    padded = pad_image(image, PATCH_SIZE)
    patches = extract_patches(padded, size=PATCH_SIZE, stride=PATCH_STRIDE)

    if len(patches) == 0:
        ndvi = compute_ndvi(image)
        return vegetation_mask(ndvi)

    # Predict each patch
    pred_patches = []
    batch = np.array(patches, dtype=np.float32)
    predictions = model.predict(batch, verbose=0)

    for pred in predictions:
        pred_patches.append((pred > 0.5).astype(np.uint8))

    # Reconstruct
    result = reconstruct_from_patches(
        pred_patches,
        (padded.shape[0], padded.shape[1]),
        size=PATCH_SIZE,
        stride=PATCH_STRIDE,
    )

    return (result[:original_h, :original_w] > 0.5).astype(np.uint8)


def predict_change(model, img_t1: np.ndarray,
                   img_t2: np.ndarray) -> np.ndarray:
    """
    Predict change mask using the Siamese U-Net model.
    Falls back to NDVI difference thresholding if model is None.

    Parameters
    ----------
    model : tf.keras.Model or None
    img_t1, img_t2 : np.ndarray (H, W, 4)

    Returns
    -------
    np.ndarray (H, W) binary mask
    """
    if model is None:
        # Fallback: NDVI-based
        ndvi_t1 = compute_ndvi(img_t1)
        ndvi_t2 = compute_ndvi(img_t2)
        return change_mask(ndvi_t1, ndvi_t2)

    original_h, original_w = img_t1.shape[:2]

    # Pad both images
    padded_t1 = pad_image(img_t1, PATCH_SIZE)
    padded_t2 = pad_image(img_t2, PATCH_SIZE)

    patches_t1 = extract_patches(padded_t1, size=PATCH_SIZE, stride=PATCH_STRIDE)
    patches_t2 = extract_patches(padded_t2, size=PATCH_SIZE, stride=PATCH_STRIDE)

    if len(patches_t1) == 0:
        ndvi_t1, ndvi_t2 = compute_ndvi(img_t1), compute_ndvi(img_t2)
        return change_mask(ndvi_t1, ndvi_t2)

    # Predict
    batch_t1 = np.array(patches_t1, dtype=np.float32)
    batch_t2 = np.array(patches_t2, dtype=np.float32)
    predictions = model.predict([batch_t1, batch_t2], verbose=0)

    pred_patches = [(pred > 0.5).astype(np.uint8) for pred in predictions]

    # Reconstruct
    result = reconstruct_from_patches(
        pred_patches,
        (padded_t1.shape[0], padded_t1.shape[1]),
        size=PATCH_SIZE,
        stride=PATCH_STRIDE,
    )

    return (result[:original_h, :original_w] > 0.5).astype(np.uint8)


def generate_maps(region_id, img_t1, img_t2, veg_pred, chg_pred,
                  veg_pct, chg_area, violation, map_dir):
    """
    Generate a 5-panel before/after map:
      [Before RGB | After RGB | Veg Mask | Change Mask | Change Overlay]
    """
    os.makedirs(map_dir, exist_ok=True)

    def to_rgb(img):
        """Convert 4-band image to brightened RGB for display."""
        rgb = img[:, :, [2, 1, 0]]          # B4=Red, B3=Green, B2=Blue
        rgb = np.clip(rgb * 3.5, 0, 1)      # Brightness boost for Sentinel-2
        return rgb

    rgb_t1 = to_rgb(img_t1)
    rgb_t2 = to_rgb(img_t2)

    # Change overlay: T2 with red pixels where change detected
    overlay = rgb_t2.copy()
    if chg_pred.sum() > 0:
        overlay[chg_pred == 1, 0] = 1.0   # Red channel → full
        overlay[chg_pred == 1, 1] *= 0.3  # Green → dim
        overlay[chg_pred == 1, 2] *= 0.3  # Blue → dim

    status = "🔴 NON-COMPLIANT" if violation else "🟢 COMPLIANT"
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (rgb_t1,   "Before (T1)",       None),
        (rgb_t2,   "After (T2)",         None),
        (veg_pred, "Vegetation Mask",   "Greens"),
        (chg_pred, "Change Mask",       "Reds"),
        (overlay,  "Change Overlay",    None),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(data, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")

    fig.suptitle(
        f"Region {region_id}  |  Vegetation: {veg_pct:.1%}  |  "
        f"Change: {chg_area:,.0f} m²  |  {status}",
        color="white", fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=0.5)

    filepath = os.path.join(map_dir, f"region_{region_id:04d}_maps.png")
    plt.savefig(filepath, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[INFER]   Saved map: {os.path.basename(filepath)}")
    return filepath


def run_inference(roi_config=None):
    """
    Run the full inference pipeline.

    Parameters
    ----------
    roi_config : dict, optional
        ROI configuration. If None, uses config.ROI_CONFIG.

    Returns
    -------
    list[dict]
        Compliance results for each industrial polygon.
    """
    if roi_config is None:
        roi_config = ROI_CONFIG

    print("=" * 60)
    print("  ENVIRONMENTAL MONITORING — INFERENCE PIPELINE")
    print("=" * 60)

    # Step 1: Get ROI
    roi_gdf = get_roi_polygon(roi_config)
    roi_polygon = roi_gdf.geometry.iloc[0]

    # Step 2: Fetch industrial polygons from OSM
    industrial_gdf = get_industrial_polygons_for_roi(roi_gdf)

    if industrial_gdf.empty:
        print("[INFER] No industrial polygons found. Cannot proceed.")
        return []

    # Step 3: Initialize GEE and fetch satellite images
    gee_available = initialize_gee()
    if not gee_available:
        print("[INFER] ERROR: GEE unavailable. Cannot fetch images.")
        return []

    print(f"[INFER] Fetching t1 image ({T1_START} to {T1_END})...")
    image_t1 = fetch_sentinel_image(roi_polygon, T1_START, T1_END)

    print(f"[INFER] Fetching t2 image ({T2_START} to {T2_END})...")
    image_t2 = fetch_sentinel_image(roi_polygon, T2_START, T2_END)

    # Normalize
    image_t1 = normalize_image(image_t1)
    image_t2 = normalize_image(image_t2)

    # Align
    image_t1, image_t2 = align_images(image_t1, image_t2)

    # Step 4: Load models
    veg_model, chg_model = load_models()

    # Step 5: Get image bounds for clipping
    roi_bounds = roi_polygon.bounds  # (minx, miny, maxx, maxy)

    # Step 6: Process each industrial polygon
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MAP_DIR, exist_ok=True)

    results = []
    total = len(industrial_gdf)
    saved_count = 0

    for idx, row in industrial_gdf.iterrows():
        poly = row.geometry
        region_id = idx + 1

        print(f"\n[INFER] Processing region {region_id}/{total}...", end=" ")

        # Clip both images to this polygon
        img1_clip = clip_image_to_polygon(image_t1, roi_bounds, poly)
        img2_clip = clip_image_to_polygon(image_t2, roi_bounds, poly)

        # Skip if too small or mostly empty (no real data)
        h, w = img1_clip.shape[:2]
        if h * w < MIN_CLIP_PIXELS:
            print(f"too small ({h}×{w} px), skipping.")
            continue
        if not np.any(img1_clip > 0) or not np.any(img2_clip > 0):
            print("blank image, skipping.")
            continue

        # Skip water-dominated regions (ports, harbours, coastal areas)
        if is_mostly_water(img2_clip, max_water_ratio=MAX_WATER_RATIO):
            print(f"({h}×{w} px) mostly water, skipping.")
            continue
        print(f"({h}×{w} px)")

        # Align clipped images
        img1_clip, img2_clip = align_images(img1_clip, img2_clip)

        # Derive valid pixel mask (pixels inside the polygon boundary)
        # Masked-out pixels are zeroed across all bands, so any band > 0 = valid
        valid_mask = np.any(img2_clip > 0, axis=2).astype(np.uint8)

        # Run vegetation model on t2
        veg_pred = predict_vegetation(veg_model, img2_clip)

        # Run change model on (t1, t2)
        chg_pred = predict_change(chg_model, img1_clip, img2_clip)

        # Compute metrics (using valid_mask so only polygon pixels count)
        veg_pct  = compute_vegetation_percentage(veg_pred, valid_mask=valid_mask)
        chg_area = compute_change_area(chg_pred)
        violation = veg_pct < VEG_VIOLATION_THRESHOLD

        # Extract geographic coordinates for interactive map
        centroid = poly.centroid
        bounds = poly.bounds  # (minx, miny, maxx, maxy)

        # Extract polygon boundary as coordinate list for Leaflet
        try:
            if poly.geom_type == 'MultiPolygon':
                coords = list(poly.geoms[0].exterior.coords)
            else:
                coords = list(poly.exterior.coords)
            # Leaflet uses [lat, lng] order
            poly_coords = [[round(c[1], 6), round(c[0], 6)] for c in coords]
        except Exception:
            poly_coords = []

        result = {
            "region_id": int(region_id),
            "size_px": [int(h), int(w)],
            "vegetation_percent": round(veg_pct, 4),
            "change_area_m2": round(chg_area, 2),
            "violation": bool(violation),
            "status": "NON-COMPLIANT" if violation else "COMPLIANT",
            "lat": round(centroid.y, 6),
            "lon": round(centroid.x, 6),
            "bounds": [
                [round(bounds[1], 6), round(bounds[0], 6)],  # [south, west]
                [round(bounds[3], 6), round(bounds[2], 6)],  # [north, east]
            ],
            "polygon": poly_coords,
        }
        results.append(result)
        saved_count += 1

        icon = "🔴" if violation else "🟢"
        print(f"[INFER]   {icon} Veg: {veg_pct:.1%} | "
              f"Change: {chg_area:,.0f} m² | {result['status']}")

        # 5-panel before/after map
        generate_maps(
            region_id, img1_clip, img2_clip,
            veg_pred, chg_pred,
            veg_pct, chg_area, violation,
            MAP_DIR,
        )

    # Compute summary statistics
    total_processed = len(results)
    violations = sum(1 for r in results if r["violation"])
    avg_veg = (sum(r["vegetation_percent"] for r in results) / total_processed
               if total_processed else 0)
    total_change = sum(r["change_area_m2"] for r in results)

    # Save JSON report with summary
    report = {
        "summary": {
            "total_regions": total_processed,
            "compliant": total_processed - violations,
            "violations": violations,
            "violation_rate": round(violations / total_processed, 4) if total_processed else 0,
            "avg_vegetation_percent": round(avg_veg, 4),
            "total_change_area_m2": round(total_change, 2),
            "roi": ROI_CONFIG.get("value", "Unknown"),
            "t1_period": f"{T1_START} to {T1_END}",
            "t2_period": f"{T2_START} to {T2_END}",
        },
        "regions": results,
    }
    report_path = os.path.join(REPORT_DIR, "compliance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFER] Compliance report saved: {report_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {total_processed} regions processed")
    print(f"  🟢 Compliant:     {total_processed - violations}")
    print(f"  🔴 Non-compliant: {violations}")
    print(f"  📊 Avg vegetation: {avg_veg:.1%}")
    print(f"  🔄 Total change:  {total_change:,.0f} m²")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    run_inference()
