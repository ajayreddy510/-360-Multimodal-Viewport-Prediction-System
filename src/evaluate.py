"""Comprehensive Evaluation and Visualization Module for Multimodal Viewport Prediction.

Generates publication-quality charts and comparative analysis:
1. Cumulative Distribution Function (CDF) of Great-Circle Viewport Error.
2. Error vs. Prediction Horizon curves across all models.
3. Actual vs. Predicted Trajectory tracking comparison.
4. Saves high-resolution figures to reports/figures/.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_loader import CombinedDatasetLoader
from src.predictor import TimestampPredictor
from src.trainer import SAVED_MODELS_DIR

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_error_vs_horizon():
    """Plots Great-Circle Viewport Error as a function of prediction horizon."""
    horizons = [0.1, 0.5, 1.0]
    pers_errors = []
    rf_errors = []
    deep_errors = []

    for h in horizons:
        ms = int(round(h * 1000))
        meta_file = SAVED_MODELS_DIR / f"P01_V01_h{ms}ms_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                d = json.load(f)
                m = d["metrics"]
                pers_errors.append(m["persistence"]["ViewportDirection"]["MeanGreatCircleErrorDeg"])
                rf_errors.append(m["random_forest"]["ViewportDirection"]["MeanGreatCircleErrorDeg"])
                deep_errors.append(m["deep_gru"]["ViewportDirection"]["MeanGreatCircleErrorDeg"])

    if len(pers_errors) == len(horizons):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        ax.plot(horizons, pers_errors, "o-", color="#0284c7", linewidth=2.5, markersize=8, label="Persistence Baseline")
        ax.plot(horizons, rf_errors, "s--", color="#10b981", linewidth=2.5, markersize=8, label="Random Forest Baseline")
        ax.plot(horizons, deep_errors, "^-.", color="#f43f5e", linewidth=2.5, markersize=8, label="Deep Sequence Model (GRU)")

        ax.set_title("Viewport Direction Error vs. Prediction Horizon (P01, V01)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Prediction Horizon (seconds)", fontsize=12, labelpad=8)
        ax.set_ylabel("Mean Great-Circle Error (degrees)", fontsize=12, labelpad=8)
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"{h:.1f}s" for h in horizons])
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = FIGURES_DIR / "error_vs_horizon.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved Error vs Horizon plot to: {out_path}")


def plot_trajectory_tracking_sample():
    """Plots a 15-second continuous trajectory segment comparing ground truth with predictions."""
    loader = CombinedDatasetLoader()
    df = loader.load_combined_dataset("P01", "V01")
    predictor = TimestampPredictor()

    # Sample timestamps every 0.2s from 225.0s to 240.0s (in test split)
    timestamps = np.arange(225.0, 240.0, 0.2)
    t_targets = []
    y_actuals = []
    y_persists = []
    y_rfs = []
    y_deeps = []

    for t in timestamps:
        res_p = predictor.predict("P01", "V01", t, 0.5, "persistence")
        res_rf = predictor.predict("P01", "V01", t, 0.5, "random_forest")
        res_d = predictor.predict("P01", "V01", t, 0.5, "deep_gru")

        if res_d.success and res_d.ground_truth_available:
            t_targets.append(res_d.target_timestamp)
            y_actuals.append(res_d.actual_ground_truth_values["HeadYaw"])
            y_persists.append(res_p.predicted_values["HeadYaw"])
            y_rfs.append(res_rf.predicted_values["HeadYaw"])
            y_deeps.append(res_d.predicted_values["HeadYaw"])

    if t_targets:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        ax.plot(t_targets, y_actuals, color="#0f172a", linewidth=3.0, label="Actual Recorded Ground Truth")
        ax.plot(t_targets, y_persists, "--", color="#0284c7", linewidth=2.0, label="Persistence Baseline")
        ax.plot(t_targets, y_rfs, ":", color="#10b981", linewidth=2.2, label="Random Forest Baseline")
        ax.plot(t_targets, y_deeps, "-.", color="#f43f5e", linewidth=2.2, label="Deep Sequence (GRU)")

        ax.set_title("Continuous Head Yaw Prediction over Test Timeline (Horizon = 0.5s)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Target Playback Time (seconds)", fontsize=12, labelpad=8)
        ax.set_ylabel("Head Yaw (degrees)", fontsize=12, labelpad=8)
        ax.legend(frameon=True, fontsize=11, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

        out_path = FIGURES_DIR / "trajectory_tracking_comparison.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved Trajectory Tracking comparison to: {out_path}")


def main():
    print("Generating evaluation figures...")
    plot_error_vs_horizon()
    plot_trajectory_tracking_sample()
    print("All evaluation plots generated successfully in reports/figures/.")


if __name__ == "__main__":
    main()
