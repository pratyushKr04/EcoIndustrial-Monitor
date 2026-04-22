"""
Training Pipeline — Change Detection Model (Siamese U-Net).

End-to-end: Fetch t1+t2 data → Generate NDVI change labels → Extract patch pairs → Train.
"""

import faulthandler
faulthandler.enable()

import os
import sys

# Force legacy Keras 2 — Keras 3's _cast_seed causes SIGSEGV on some GPUs.
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
    ROI_CONFIG, T1_START, T1_END, T2_START, T2_END,
    NDVI_CHANGE_THRESHOLD, PATCH_SIZE, PATCH_STRIDE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT,
    CHANGE_MODEL_PATH, MIN_CLIP_PIXELS, MAX_WATER_RATIO,
    TRAINING_CITIES, USE_MIXED_PRECISION,
)
from utils.roi import get_roi_polygon
from utils.osm import get_industrial_polygons_for_roi
from utils.satellite import fetch_sentinel_image, initialize_gee
from utils.ndvi import compute_ndvi, change_mask, is_mostly_water
from utils.preprocessing import (
    normalize_image, align_images, extract_patch_pairs,
    pad_image, clip_image_to_polygon,
)
from models.siamese_unet import get_change_model


def has_valid_content(image: np.ndarray, min_content_ratio: float = 0.1) -> bool:
    """Return True if the image has enough non-zero pixels to be useful."""
    if image.size == 0:
        return False
    valid = np.sum(np.any(image > 0, axis=-1))
    return (valid / (image.shape[0] * image.shape[1])) >= min_content_ratio


def prepare_change_data(image_t1: np.ndarray,
                        image_t2: np.ndarray,
                        industrial_gdf,
                        roi_bounds: tuple) -> tuple:
    """
    Generate training patch pairs ONLY from industrial polygon crops.

    Parameters
    ----------
    image_t1, image_t2 : np.ndarray  (H, W, 4)
    industrial_gdf     — GeoDataFrame of industrial polygons
    roi_bounds         — (minx, miny, maxx, maxy)
    """
    image_t1 = normalize_image(image_t1)
    image_t2 = normalize_image(image_t2)
    image_t1, image_t2 = align_images(image_t1, image_t2)

    X1_patches, X2_patches, Y_patches = [], [], []
    skipped = 0

    for idx, row in industrial_gdf.iterrows():
        poly = row.geometry

        crop1 = clip_image_to_polygon(image_t1, roi_bounds, poly)
        crop2 = clip_image_to_polygon(image_t2, roi_bounds, poly)

        h, w = crop1.shape[:2]
        if h * w < MIN_CLIP_PIXELS or not has_valid_content(crop1):
            skipped += 1
            continue

        # Skip water-dominated crops
        if is_mostly_water(crop1, max_water_ratio=MAX_WATER_RATIO):
            skipped += 1
            continue

        crop1, crop2 = align_images(crop1, crop2)

        # NDVI change mask as label
        ndvi1 = compute_ndvi(crop1)
        ndvi2 = compute_ndvi(crop2)
        chg_mask_crop = change_mask(ndvi1, ndvi2, threshold=NDVI_CHANGE_THRESHOLD)

        # Pad
        crop1 = pad_image(crop1, PATCH_SIZE)
        crop2 = pad_image(crop2, PATCH_SIZE)
        chg_mask_crop = pad_image(chg_mask_crop, PATCH_SIZE)

        p1, p2, pm = extract_patch_pairs(
            crop1, crop2, chg_mask_crop,
            size=PATCH_SIZE, stride=PATCH_STRIDE,
        )
        X1_patches.extend(p1)
        X2_patches.extend(p2)
        Y_patches.extend(pm)

    print(f"[TRAIN-CHG] Industrial polygons used: {len(industrial_gdf) - skipped} "
          f"(skipped {skipped} too-small/empty)")
    print(f"[TRAIN-CHG] Total patch pairs extracted: {len(X1_patches)}")

    if not X1_patches:
        return np.array([]), np.array([]), np.array([])

    X1 = np.array(X1_patches, dtype=np.float32)
    X2 = np.array(X2_patches, dtype=np.float32)
    Y  = np.array(Y_patches,  dtype=np.float32)
    if Y.ndim == 3:
        Y = Y[..., np.newaxis]

    return X1, X2, Y


def train_change_model(X1: np.ndarray, X2: np.ndarray,
                       Y: np.ndarray) -> tf.keras.Model:
    """
    Train the Siamese U-Net change detection model.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Image patch pairs (N, 256, 256, 4).
    Y : np.ndarray
        Change masks (N, 256, 256, 1).

    Returns
    -------
    tf.keras.Model
        Trained model.
    """
    # Train/validation split (same indices for both inputs)
    indices = np.arange(len(X1))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=42
    )

    X1_train, X1_val = X1[train_idx], X1[val_idx]
    X2_train, X2_val = X2[train_idx], X2[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    print(f"[TRAIN-CHG] Training samples:   {len(train_idx)}")
    print(f"[TRAIN-CHG] Validation samples: {len(val_idx)}")

    # Optionally enable mixed precision (disabled by default — can crash)
    if USE_MIXED_PRECISION:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[TRAIN-CHG] Mixed precision (float16) enabled.")
        except Exception as e:
            print(f"[TRAIN-CHG] Mixed precision failed: {e}, using float32.")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("[TRAIN-CHG] Using float32 precision.")

    # Build model
    model = get_change_model(
        input_shape=(PATCH_SIZE, PATCH_SIZE, 4),
        learning_rate=LEARNING_RATE,
    )

    # ── tf.data.Dataset: stream batches CPU→GPU, never load all at once ──
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({"input_t1": X1_train, "input_t2": X2_train}, Y_train)
        )
        .shuffle(buffer_size=len(train_idx), seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({"input_t1": X1_val, "input_t2": X2_val}, Y_val)
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Callbacks
    os.makedirs(os.path.dirname(CHANGE_MODEL_PATH), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CHANGE_MODEL_PATH,
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
    print(f"[TRAIN-CHG] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Print final metrics
    val_loss = min(history.history["val_loss"])
    print(f"[TRAIN-CHG] Best validation loss: {val_loss:.4f}")

    # Save training history and plots
    _save_training_history(history, "change_detection")

    return model


def _save_training_history(history, model_name: str):
    """Save training metrics as JSON and generate plots."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = os.path.join("outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    json_path = os.path.join(metrics_dir, f"{model_name}_history.json")
    with open(json_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"[TRAIN] Saved metrics: {json_path}")

    epochs = range(1, len(hist_data["loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"{model_name.replace('_', ' ').title()} Model — Training Metrics",
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


def run_change_training():
    """Full pipeline: fetch t1+t2 data from multiple cities → train change model.

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
    print("  CHANGE DETECTION MODEL TRAINING PIPELINE")
    print(f"  Training on {len(TRAINING_CITIES)} cities")
    print("=" * 60)

    print_cache_summary()

    # Initialize GEE only if needed (deferred)
    gee_initialized = False

    # Aggregate patches from all cities
    all_X1, all_X2, all_Y = [], [], []

    for city_idx, city_name in enumerate(TRAINING_CITIES, 1):
        print(f"\n{'─' * 50}")
        print(f"[TRAIN-CHG] City {city_idx}/{len(TRAINING_CITIES)}: {city_name}")
        print(f"{'─' * 50}")

        try:
            # ── Load or fetch ROI ──
            roi_bounds = load_roi_bounds_cache(city_name)
            if roi_bounds is not None:
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
                print(f"[TRAIN-CHG]   No industrial polygons found, skipping.")
                continue
            print(f"[TRAIN-CHG]   {len(industrial_gdf)} industrial polygons.")

            # ── Load or fetch satellite images (both t1 and t2) ──
            image_t1 = load_satellite_cache(city_name, T1_START, T1_END)
            image_t2 = load_satellite_cache(city_name, T2_START, T2_END)

            need_download = image_t1 is None or image_t2 is None
            if need_download and not gee_initialized:
                gee_initialized = initialize_gee()
                if not gee_initialized:
                    print("[TRAIN-CHG] ERROR: GEE not available.")
                    return None

            if image_t1 is None:
                image_t1 = fetch_sentinel_image(roi_polygon, T1_START, T1_END)
                save_satellite_cache(image_t1, city_name, T1_START, T1_END)
            if image_t2 is None:
                image_t2 = fetch_sentinel_image(roi_polygon, T2_START, T2_END)
                save_satellite_cache(image_t2, city_name, T2_START, T2_END)

            # ── Extract patch pairs from industrial zones ──
            X1, X2, Y = prepare_change_data(
                image_t1, image_t2, industrial_gdf, roi_bounds
            )

            if len(X1) > 0:
                all_X1.append(X1)
                all_X2.append(X2)
                all_Y.append(Y)
                print(f"[TRAIN-CHG]   ✓ {len(X1)} patch pairs from {city_name}")
            else:
                print(f"[TRAIN-CHG]   No valid patch pairs from {city_name}.")

        except Exception as e:
            print(f"[TRAIN-CHG]   ERROR processing {city_name}: {e}")
            print(f"[TRAIN-CHG]   Skipping this city and continuing...")
            continue

    # Combine all patches
    if not all_X1:
        print("\n[TRAIN-CHG] ERROR: No training data from any city!")
        return None

    X1_combined = np.concatenate(all_X1, axis=0)
    X2_combined = np.concatenate(all_X2, axis=0)
    Y_combined  = np.concatenate(all_Y,  axis=0)

    print(f"\n{'=' * 60}")
    print(f"  TOTAL TRAINING DATA: {len(X1_combined)} patch pairs from {len(all_X1)} cities")
    print(f"{'=' * 60}")

    # Shuffle the combined dataset
    indices = np.random.RandomState(42).permutation(len(X1_combined))
    X1_combined = X1_combined[indices]
    X2_combined = X2_combined[indices]
    Y_combined  = Y_combined[indices]

    # Train
    model = train_change_model(X1_combined, X2_combined, Y_combined)

    print(f"[TRAIN-CHG] Model saved to: {CHANGE_MODEL_PATH}")
    print("=" * 60)
    print("  CHANGE DETECTION MODEL TRAINING COMPLETE")
    print("=" * 60)

    return model


if __name__ == "__main__":
    run_change_training()
