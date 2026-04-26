"""
Training Pipeline — Vegetation Model (U-Net).

End-to-end: Fetch data → Generate NDVI labels → Extract patches → Train U-Net.
"""

import faulthandler
faulthandler.enable()  # Print traceback on SIGSEGV instead of silent crash

import os
import sys

# Force TensorFlow to use legacy Keras 2 instead of Keras 3.
# Keras 3's random seed _cast_seed uses floor_mod which causes SIGSEGV
# on certain TF + GPU combinations (e.g. L40S).
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Allow GPU memory to grow incrementally instead of grabbing it all at once
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ROI_CONFIG, T2_START, T2_END,
    NDVI_VEG_THRESHOLD, PATCH_SIZE, PATCH_STRIDE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT,
    VEG_MODEL_PATH, MIN_CLIP_PIXELS, MAX_WATER_RATIO,
    TRAINING_CITIES, USE_MIXED_PRECISION,
)
from utils.roi import get_roi_polygon
from utils.osm import get_industrial_polygons_for_roi
from utils.satellite import fetch_sentinel_image, initialize_gee
from utils.ndvi import compute_ndvi, vegetation_mask, is_mostly_water
from utils.preprocessing import (
    normalize_image, extract_patches, pad_image, clip_image_to_polygon,
)
from models.unet import get_vegetation_model


def has_valid_content(image: np.ndarray, min_content_ratio: float = 0.1) -> bool:
    """Return True if the image has enough non-zero pixels to be useful."""
    if image.size == 0:
        return False
    valid = np.sum(np.any(image > 0, axis=-1))
    return (valid / (image.shape[0] * image.shape[1])) >= min_content_ratio


def prepare_vegetation_data(image: np.ndarray,
                             industrial_gdf,
                             roi_bounds: tuple) -> tuple:
    """
    Generate training patches ONLY from industrial polygon crops.

    Instead of sliding-window patching the whole city image, we clip to
    each OSM industrial polygon and extract patches from those crops.
    This ensures all training data comes from actual industrial areas.

    Parameters
    ----------
    image : np.ndarray  (H, W, 4) — full ROI satellite image
    industrial_gdf     — GeoDataFrame of industrial polygons
    roi_bounds         — (minx, miny, maxx, maxy) bounds of the full image
    """
    image = normalize_image(image)

    X_patches, Y_patches = [], []
    skipped = 0

    for idx, row in industrial_gdf.iterrows():
        poly = row.geometry

        # Clip image to this industrial polygon (no masking for training patches)
        crop = clip_image_to_polygon(image, roi_bounds, poly, mask_outside=False)

        # Skip if too small or mostly empty
        h, w = crop.shape[:2]
        if h * w < MIN_CLIP_PIXELS or not has_valid_content(crop):
            skipped += 1
            continue

        # Skip water-dominated crops
        if is_mostly_water(crop, max_water_ratio=MAX_WATER_RATIO):
            skipped += 1
            continue

        # Compute NDVI label on the crop
        ndvi = compute_ndvi(crop)
        veg_mask_crop = vegetation_mask(ndvi, threshold=NDVI_VEG_THRESHOLD)

        # Pad if smaller than patch size
        crop = pad_image(crop, PATCH_SIZE)
        veg_mask_crop = pad_image(veg_mask_crop, PATCH_SIZE)

        # Extract patches
        x_pats = extract_patches(crop, size=PATCH_SIZE, stride=PATCH_STRIDE)
        y_pats = extract_patches(
            veg_mask_crop[..., np.newaxis] if veg_mask_crop.ndim == 2 else veg_mask_crop,
            size=PATCH_SIZE, stride=PATCH_STRIDE,
        )
        X_patches.extend(x_pats)
        Y_patches.extend(y_pats)

    print(f"[TRAIN-VEG] Industrial polygons used: {len(industrial_gdf) - skipped} "
          f"(skipped {skipped} too-small/empty)")
    print(f"[TRAIN-VEG] Total patches extracted: {len(X_patches)}")

    if not X_patches:
        return np.array([]), np.array([])

    X = np.array(X_patches, dtype=np.float32)
    Y = np.array(Y_patches, dtype=np.float32)
    if Y.ndim == 3:
        Y = Y[..., np.newaxis]

    return X, Y



def train_vegetation_model(X: np.ndarray, Y: np.ndarray) -> tf.keras.Model:
    """
    Train the vegetation U-Net model.

    Parameters
    ----------
    X : np.ndarray
        Training images (N, 256, 256, 4).
    Y : np.ndarray
        Training masks (N, 256, 256, 1).

    Returns
    -------
    tf.keras.Model
        Trained model.
    """
    # Train/validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=VAL_SPLIT, random_state=42
    )

    print(f"[TRAIN-VEG] Training samples:   {len(X_train)}")
    print(f"[TRAIN-VEG] Validation samples: {len(X_val)}")

    # Optionally enable mixed precision (disabled by default — can crash)
    if USE_MIXED_PRECISION:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[TRAIN-VEG] Mixed precision (float16) enabled.")
        except Exception as e:
            print(f"[TRAIN-VEG] Mixed precision failed: {e}, using float32.")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("[TRAIN-VEG] Using float32 precision.")

    # Build model
    model = get_vegetation_model(
        input_shape=(PATCH_SIZE, PATCH_SIZE, 4),
        learning_rate=LEARNING_RATE,
    )

    # Callbacks
    os.makedirs(os.path.dirname(VEG_MODEL_PATH), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            VEG_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print(f"[TRAIN-VEG] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Print final metrics
    val_loss = min(history.history["val_loss"])
    print(f"[TRAIN-VEG] Best validation loss: {val_loss:.4f}")

    # Save training history and plots
    _save_training_history(history, "vegetation")

    return model


def _save_training_history(history, model_name: str):
    """Save training metrics as JSON and generate plots."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = os.path.join("outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save raw history as JSON
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    json_path = os.path.join(metrics_dir, f"{model_name}_history.json")
    with open(json_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"[TRAIN] Saved metrics: {json_path}")

    # Generate plots
    epochs = range(1, len(hist_data["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"{model_name.title()} Model — Training Metrics",
                 color="white", fontsize=16, fontweight="bold")

    plot_configs = [
        ("loss", "val_loss", "Loss", "#ff6b6b", "#ffd93d"),
        ("accuracy", "val_accuracy", "Accuracy", "#6bcb77", "#4d96ff"),
        ("iou_metric", "val_iou_metric", "IoU", "#ff9a3c", "#a855f7"),
        ("precision_metric", "val_precision_metric", "Precision", "#38bdf8", "#fb7185"),
    ]

    for ax, (train_key, val_key, title, tc, vc) in zip(axes.flat, plot_configs):
        ax.set_facecolor("#16213e")
        if train_key in hist_data:
            ax.plot(epochs, hist_data[train_key], color=tc, linewidth=2, label="Train")
        if val_key in hist_data:
            ax.plot(epochs, hist_data[val_key], color=vc, linewidth=2,
                    linestyle="--", label="Validation")
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel(title, color="#aaa")
        ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(metrics_dir, f"{model_name}_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[TRAIN] Saved plots: {plot_path}")


def run_vegetation_training():
    """Full pipeline: fetch data from multiple cities → train vegetation model.

    Uses local cache — satellite images and OSM polygons are downloaded once
    and reused on subsequent runs.
    """
    from utils.cache import (
        load_satellite_cache, save_satellite_cache,
        load_osm_cache, save_osm_cache,
        load_roi_bounds_cache, save_roi_bounds_cache,
        print_cache_summary,
    )

    print("=" * 60)
    print("  VEGETATION MODEL TRAINING PIPELINE")
    print(f"  Training on {len(TRAINING_CITIES)} cities")
    print("=" * 60)

    print_cache_summary()

    # Initialize GEE (only needed if any city is not cached)
    gee_initialized = False

    # Aggregate patches from all cities
    all_X, all_Y = [], []

    for city_idx, city_name in enumerate(TRAINING_CITIES, 1):
        print(f"\n{'─' * 50}")
        print(f"[TRAIN-VEG] City {city_idx}/{len(TRAINING_CITIES)}: {city_name}")
        print(f"{'─' * 50}")

        try:
            # ── Load or fetch ROI ──
            roi_bounds = load_roi_bounds_cache(city_name)
            if roi_bounds is not None:
                # Reconstruct polygon from bounds for OSM fetch fallback
                from shapely.geometry import box
                roi_polygon = box(*roi_bounds)
            else:
                city_config = {"type": "place", "value": city_name}
                roi_gdf = get_roi_polygon(city_config)
                roi_polygon = roi_gdf.geometry.iloc[0]
                roi_bounds = roi_polygon.bounds
                save_roi_bounds_cache(roi_bounds, city_name)

            # ── Load or fetch OSM polygons ──
            industrial_gdf = load_osm_cache(city_name)
            if industrial_gdf is None:
                roi_gdf_for_osm = get_roi_polygon({"type": "place", "value": city_name})
                industrial_gdf = get_industrial_polygons_for_roi(roi_gdf_for_osm)
                if not industrial_gdf.empty:
                    save_osm_cache(industrial_gdf, city_name)

            if industrial_gdf is None or industrial_gdf.empty:
                print(f"[TRAIN-VEG]   No industrial polygons found, skipping.")
                continue
            print(f"[TRAIN-VEG]   {len(industrial_gdf)} industrial polygons.")

            # ── Load or fetch satellite image ──
            image = load_satellite_cache(city_name, T2_START, T2_END)
            if image is None:
                if not gee_initialized:
                    gee_initialized = initialize_gee()
                    if not gee_initialized:
                        print("[TRAIN-VEG] ERROR: GEE not available.")
                        return None
                image = fetch_sentinel_image(roi_polygon, T2_START, T2_END)
                save_satellite_cache(image, city_name, T2_START, T2_END)

            # ── Extract patches from industrial zones ──
            X, Y = prepare_vegetation_data(image, industrial_gdf, roi_bounds)

            if len(X) > 0:
                all_X.append(X)
                all_Y.append(Y)
                print(f"[TRAIN-VEG]   ✓ {len(X)} patches from {city_name}")
            else:
                print(f"[TRAIN-VEG]   No valid patches from {city_name}.")

        except Exception as e:
            print(f"[TRAIN-VEG]   ERROR processing {city_name}: {e}")
            print(f"[TRAIN-VEG]   Skipping this city and continuing...")
            continue

    # Combine all patches
    if not all_X:
        print("\n[TRAIN-VEG] ERROR: No training data from any city!")
        return None

    X_combined = np.concatenate(all_X, axis=0)
    Y_combined = np.concatenate(all_Y, axis=0)

    print(f"\n{'=' * 60}")
    print(f"  TOTAL TRAINING DATA: {len(X_combined)} patches from {len(all_X)} cities")
    print(f"{'=' * 60}")

    # Shuffle the combined dataset
    indices = np.random.RandomState(42).permutation(len(X_combined))
    X_combined = X_combined[indices]
    Y_combined = Y_combined[indices]

    # Train
    model = train_vegetation_model(X_combined, Y_combined)

    print(f"[TRAIN-VEG] Model saved to: {VEG_MODEL_PATH}")
    print("=" * 60)
    print("  VEGETATION MODEL TRAINING COMPLETE")
    print("=" * 60)

    return model


if __name__ == "__main__":
    run_vegetation_training()
