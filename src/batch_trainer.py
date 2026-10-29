"""Batch Training CLI Module for Multimodal Viewport Prediction.

Enables automated batch training across multiple participants, videos, and horizons.
Usage:
    python -m src.batch_trainer --persons P01,P02 --videos V01 --horizons 0.1,0.5,1.0
"""

import argparse
from pathlib import Path
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_loader import CombinedDatasetLoader
from src.trainer import ModelTrainer, SAVED_MODELS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Model Trainer for 360° Multimodal Viewport Prediction")
    parser.add_argument(
        "--persons",
        type=str,
        default="P01",
        help="Comma-separated person IDs (e.g., 'P01,P02' or 'all')",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default="V01",
        help="Comma-separated video IDs (e.g., 'V01,V02' or 'all')",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="0.5",
        help="Comma-separated prediction horizons in seconds (e.g., '0.1,0.5,1.0')",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of PyTorch training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--rf-estimators", type=int, default=30, help="Random Forest estimators")
    parser.add_argument("--lookback", type=int, default=30, help="Historical lookback window size")
    return parser.parse_args()


def main():
    args = parse_args()
    loader = CombinedDatasetLoader()

    # Resolve persons
    if args.persons.lower() == "all":
        persons = loader.list_available_persons()
    else:
        persons = [loader.normalize_person_id(p.strip()) for p in args.persons.split(",") if p.strip()]

    # Resolve videos
    if args.videos.lower() == "all":
        videos = [v[0] for v in loader.list_available_videos()]
    else:
        videos = [loader.normalize_video_id(v.strip())[0] for v in args.videos.split(",") if v.strip()]

    # Resolve horizons
    horizons = [float(h.strip()) for h in args.horizons.split(",") if h.strip()]

    total_jobs = len(persons) * len(videos) * len(horizons)
    print(f"=== Starting Batch Training ===")
    print(f"Persons ({len(persons)}): {persons}")
    print(f"Videos ({len(videos)}): {videos}")
    print(f"Horizons ({len(horizons)}): {horizons} seconds")
    print(f"Total Model Combinations: {total_jobs}")
    print("=" * 35)

    job_idx = 1
    for p in persons:
        for v in videos:
            for h in horizons:
                print(f"\n[{job_idx}/{total_jobs}] Training {p} {v} at horizon {h:.1f}s...")
                try:
                    trainer = ModelTrainer(
                        person_id=p,
                        video_id=v,
                        lookback_window=args.lookback,
                        prediction_horizon_sec=h,
                        save_dir=SAVED_MODELS_DIR,
                    )
                    trainer.train_and_evaluate(
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        rf_estimators=args.rf_estimators,
                        verbose=False,
                    )
                    print(f"  [OK] {p} {v} h={h:.1f}s successfully trained & saved.")
                except Exception as ex:
                    print(f"  [FAIL] Failed for {p} {v} h={h:.1f}s: {str(ex)}")
                job_idx += 1

    print("\n=== Batch Training Complete ===")


if __name__ == "__main__":
    main()
