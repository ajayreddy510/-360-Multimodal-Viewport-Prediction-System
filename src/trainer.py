"""Training and Evaluation Module for Multimodal Viewport Prediction.

Trains, evaluates, compares, and saves:
1. PersistenceBaselineModel
2. MachineLearningBaselineModel (Random Forest)
3. DeepSequenceModel (PyTorch GRU)

Saves models, scalers, feature order, target order, and metrics together in saved_models/.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.combined_loader import CombinedDatasetLoader
from src.models import DeepSequenceModel, MachineLearningBaselineModel, PersistenceBaselineModel
from src.preprocessor import MultimodalPreprocessor, DEFAULT_FEATURE_COLUMNS, DEFAULT_TARGET_COLUMNS
from src.sequence_generator import SequenceGenerator, SequenceDataset
from src.utils import compute_metrics

SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"


class ModelTrainer:
    """Orchestrates end-to-end model training, evaluation comparison, and artifact saving."""

    def __init__(
        self,
        person_id: str = "P01",
        video_id: str = "V01",
        lookback_window: int = 30,
        prediction_horizon_sec: float = 0.5,
        save_dir: Path = SAVED_MODELS_DIR,
    ):
        self.person_id = person_id
        self.video_id = video_id
        self.lookback_window = lookback_window
        self.prediction_horizon_sec = prediction_horizon_sec
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.loader = CombinedDatasetLoader()
        self.preprocessor = MultimodalPreprocessor()
        self.seq_gen = SequenceGenerator(
            lookback_window=lookback_window,
            prediction_horizon_sec=prediction_horizon_sec,
            stride=2,
        )

    def train_and_evaluate(
        self,
        epochs: int = 20,
        batch_size: int = 64,
        rf_estimators: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Loads data, prepares chronological splits, trains all models, and evaluates them."""
        if verbose:
            print(f"=== Training Pipeline for Person: {self.person_id}, Video: {self.video_id} ===")
            print(f"Lookback: {self.lookback_window} steps | Horizon: {self.prediction_horizon_sec}s")

        # 1. Load Combined Dataset strictly
        df = self.loader.load_combined_dataset(self.person_id, self.video_id)

        # 2. Chronological Train/Test Split (80% train, 20% test)
        train_data, test_data = self.seq_gen.create_train_test_split(
            df, self.preprocessor, train_ratio=0.8, time_column="PlaybackTime"
        )

        if verbose:
            print(f"Generated {len(train_data)} train sequences, {len(test_data)} test sequences.")
            print(f"Input features: {train_data.X.shape[-1]} | Targets: {len(train_data.target_names)}")

        # Prepare unscaled sequences for persistence baseline directly from test_data
        raw_feat_cols = self.preprocessor.feature_columns
        X_test_unscaled = test_data.X_raw
        y_test_ground_truth = test_data.y
        X_test_features = test_data.X

        # 3. Train & Evaluate Persistence Baseline
        if verbose:
            print("\nEvaluating Persistence Baseline...")
        persistence = PersistenceBaselineModel(
            target_names=train_data.target_names, raw_feature_names=raw_feat_cols
        )
        y_pred_persist = persistence.predict(X_test_unscaled)
        metrics_persist = compute_metrics(
            y_pred_persist, y_test_ground_truth, train_data.target_names
        )

        # Extract last historical values for targets to enable residual delta training
        t_to_feat = {
            t_idx: raw_feat_cols.index(t_name)
            for t_idx, t_name in enumerate(train_data.target_names)
            if t_name in raw_feat_cols
        }

        y_train_last = np.zeros_like(train_data.y)
        y_test_last = np.zeros_like(test_data.y)
        for t_idx, f_idx in t_to_feat.items():
            y_train_last[:, t_idx] = train_data.X_raw[:, -1, f_idx]
            y_test_last[:, t_idx] = test_data.X_raw[:, -1, f_idx]

        from src.utils import wrap_angle_deg
        delta_train = train_data.y - y_train_last
        delta_test = test_data.y - y_test_last

        for t_idx, t_name in enumerate(train_data.target_names):
            if t_name in ["HeadYaw", "GazeYaw", "AbsoluteGazeYaw"]:
                delta_train[:, t_idx] = wrap_angle_deg(delta_train[:, t_idx])
                delta_test[:, t_idx] = wrap_angle_deg(delta_test[:, t_idx])

        # 4. Train & Evaluate Random Forest Baseline (Residual Delta)
        if verbose:
            print("Training Random Forest Tabular Baseline (Residual Delta)...")
        ml_model = MachineLearningBaselineModel(n_estimators=rf_estimators, max_depth=8)
        ml_model.fit(train_data.X, delta_train)

        y_pred_rf = ml_model.predict(X_test_features, y_test_last)
        for t_idx, t_name in enumerate(train_data.target_names):
            if t_name in ["HeadYaw", "GazeYaw", "AbsoluteGazeYaw"]:
                y_pred_rf[:, t_idx] = wrap_angle_deg(y_pred_rf[:, t_idx])
        metrics_rf = compute_metrics(
            y_pred_rf, y_test_ground_truth, train_data.target_names
        )

        # 5. Train & Evaluate Deep Sequence Model (PyTorch GRU with Residual Learning)
        if verbose:
            print(f"Training Deep Sequence Model (PyTorch GRU, {epochs} epochs)...")
        input_dim = train_data.X.shape[-1]
        output_dim = delta_train.shape[-1]

        deep_model = DeepSequenceModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=48,
            num_layers=1,
            rnn_type="GRU",
            lr=0.003,
        )
        deep_model.fit(
            X_train=train_data.X,
            y_train=delta_train,
            X_val=X_test_features,
            y_val=delta_test,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        y_pred_deep = deep_model.predict(X_test_features, y_test_last)
        for t_idx, t_name in enumerate(train_data.target_names):
            if t_name in ["HeadYaw", "GazeYaw", "AbsoluteGazeYaw"]:
                y_pred_deep[:, t_idx] = wrap_angle_deg(y_pred_deep[:, t_idx])
        metrics_deep = compute_metrics(
            y_pred_deep, y_test_ground_truth, train_data.target_names
        )

        # 6. Save Models, Preprocessor & Metadata
        if verbose:
            print("\nSaving trained models, preprocessor, and configuration...")

        prefix = f"{self.person_id}_{self.video_id}_h{int(self.prediction_horizon_sec*1000)}ms"

        # Save preprocessor
        preprocessor_path = self.save_dir / f"{prefix}_preprocessor.joblib"
        self.preprocessor.save(preprocessor_path)

        # Save RF model
        rf_path = self.save_dir / f"{prefix}_rf_model.joblib"
        ml_model.save(rf_path)

        # Save Deep Model
        deep_path = self.save_dir / f"{prefix}_deep_model.pt"
        deep_model.save(deep_path)

        # Save Metadata
        metadata = {
            "person_id": self.person_id,
            "video_id": self.video_id,
            "lookback_window": self.lookback_window,
            "prediction_horizon_sec": self.prediction_horizon_sec,
            "target_names": train_data.target_names,
            "feature_names": self.preprocessor.engineered_feature_names,
            "raw_feature_names": raw_feat_cols,
            "preprocessor_path": str(preprocessor_path),
            "rf_model_path": str(rf_path),
            "deep_model_path": str(deep_path),
            "metrics": {
                "persistence": metrics_persist,
                "random_forest": metrics_rf,
                "deep_gru": metrics_deep,
            },
        }

        meta_path = self.save_dir / f"{prefix}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"Artifacts successfully saved to: {self.save_dir}")
            print(f"Metadata file: {meta_path}")

        return {
            "metadata": metadata,
            "metrics_persistence": metrics_persist,
            "metrics_rf": metrics_rf,
            "metrics_deep": metrics_deep,
            "y_test": y_test_ground_truth,
            "y_pred_deep": y_pred_deep,
            "y_pred_persist": y_pred_persist,
            "y_pred_rf": y_pred_rf,
            "test_endpoints": test_data.history_endpoints,
            "target_timestamps": test_data.target_timestamps,
        }


if __name__ == "__main__":
    trainer = ModelTrainer(person_id="P01", video_id="V01", lookback_window=30, prediction_horizon_sec=0.5)
    results = trainer.train_and_evaluate(epochs=15, batch_size=64, rf_estimators=30, verbose=True)
    print("\nTraining complete! Summary Comparison:")
    vp_pers = results["metrics_persistence"].get("ViewportDirection", {})
    vp_rf = results["metrics_rf"].get("ViewportDirection", {})
    vp_deep = results["metrics_deep"].get("ViewportDirection", {})
    print(f"Viewport Direction Great Circle Error (Mean degrees):")
    print(f" - Persistence Baseline: {vp_pers.get('MeanGreatCircleErrorDeg', 'N/A')}°")
    print(f" - Random Forest:        {vp_rf.get('MeanGreatCircleErrorDeg', 'N/A')}°")
    print(f" - Deep GRU Sequence:    {vp_deep.get('MeanGreatCircleErrorDeg', 'N/A')}°")
