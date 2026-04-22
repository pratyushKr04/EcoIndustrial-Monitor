"""
Environmental Monitoring System — Main Entry Point.

Usage:
    python main.py                          # Uses config.py defaults
    python main.py --roi "Mumbai, India"    # Override ROI
    python main.py --mode train             # Train models only
    python main.py --mode infer             # Run inference only
    python main.py --mode all               # Train + Infer (default)
"""

import argparse
import io
import os
import sys

# Fix Windows console encoding — cp1252 can't print Unicode arrows/emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Force legacy Keras 2 — Keras 3 causes SIGSEGV on some TF+GPU combos.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROI_CONFIG


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Computer Vision-Based Environmental Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                            # Full pipeline with defaults
  python main.py --roi "Mumbai, India"      # Custom region
  python main.py --mode train               # Train models only
  python main.py --mode infer               # Inference only
  python main.py --mode all                 # Train + Inference
        """,
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Region of Interest (place name). Overrides config.py.",
    )
    parser.add_argument(
        "--roi-type",
        type=str,
        choices=["place", "bbox", "polygon"],
        default="place",
        help="Type of ROI input (default: place).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "all"],
        default="all",
        help="Pipeline mode: 'train', 'infer', or 'all' (default: all).",
    )
    parser.add_argument(
        "--t1-start", type=str, default=None, help="t1 start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--t1-end", type=str, default=None, help="t1 end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--t2-start", type=str, default=None, help="t2 start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--t2-end", type=str, default=None, help="t2 end date (YYYY-MM-DD)."
    )

    return parser.parse_args()


def update_config(args):
    """Update config.py values based on command-line arguments."""
    import config

    if args.roi:
        config.ROI_CONFIG = {
            "type": args.roi_type,
            "value": args.roi,
        }
        print(f"[MAIN] ROI overridden: {config.ROI_CONFIG}")

    if args.t1_start:
        config.T1_START = args.t1_start
    if args.t1_end:
        config.T1_END = args.t1_end
    if args.t2_start:
        config.T2_START = args.t2_start
    if args.t2_end:
        config.T2_END = args.t2_end


def run_training():
    """
    Run both training pipelines in SEPARATE PROCESSES.

    TensorFlow never releases VRAM back to the OS within the same process,
    even after clear_session(). Running each training step as a subprocess
    guarantees 100% GPU memory is freed between runs when the process exits.
    """
    import subprocess

    print("\n" + "=" * 60)
    print("  PHASE 1: MODEL TRAINING")
    print("=" * 60 + "\n")

    project_root = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # ── Step 1: Train vegetation model in its own process ──
    print("[MAIN] Launching vegetation model training (separate process)...")
    veg_script = os.path.join(project_root, "training", "run_veg.py")
    result = subprocess.run(
        [sys.executable, veg_script],
        cwd=project_root,
        env=env,
    )
    if result.returncode != 0:
        print(f"[MAIN] ERROR: Vegetation training failed (exit code {result.returncode}).")
        return
    print("[MAIN] Vegetation training process finished. GPU memory fully released.\n")

    # ── Step 2: Train change model in its own process ──
    print("[MAIN] Launching change detection model training (separate process)...")
    chg_script = os.path.join(project_root, "training", "run_chg.py")
    result = subprocess.run(
        [sys.executable, chg_script],
        cwd=project_root,
        env=env,
    )
    if result.returncode != 0:
        print(f"[MAIN] ERROR: Change detection training failed (exit code {result.returncode}).")
        return
    print("[MAIN] Change detection training process finished.\n")



def run_inference_pipeline(roi_config=None):
    """Run the inference pipeline."""
    print("\n" + "=" * 60)
    print("  PHASE 2: INFERENCE")
    print("=" * 60 + "\n")

    from inference.run_inference import run_inference

    results = run_inference(roi_config)
    return results


def main():
    """Main entry point."""
    args = parse_args()
    update_config(args)

    import config

    print("=" * 60)
    print("  ENVIRONMENTAL MONITORING SYSTEM")
    print(f"  ROI: {config.ROI_CONFIG['value']}")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    # Create output directories
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    if args.mode in ("train", "all"):
        run_training()

    if args.mode in ("infer", "all"):
        results = run_inference_pipeline(config.ROI_CONFIG)

        if results:
            print(f"\n[MAIN] Pipeline complete. {len(results)} regions analyzed.")
        else:
            print("\n[MAIN] Pipeline complete. No regions processed.")

    print("\n[MAIN] Done.")


if __name__ == "__main__":
    main()
