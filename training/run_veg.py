"""
Train vegetation model only — called as a subprocess by main.py.
This isolates GPU memory so it's fully released when this process exits.
"""
import os
import sys

# Force legacy Keras 2 — must be set before importing TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_vegetation import run_vegetation_training

if __name__ == "__main__":
    run_vegetation_training()
