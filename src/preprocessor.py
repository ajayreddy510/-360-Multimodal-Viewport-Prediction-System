"""Preprocessing and Feature Engineering Module for Multimodal Viewport Prediction.

Key Responsibilities:
- Standardizes feature and target definitions.
- Provides sine/cosine encoding for circular angles (e.g. Yaw in [-180, +180]).
- Fits scalers strictly on training data only to prevent data leakage.
- Provides correct inverse-transformation for all target outputs back to physical units.
- Serializes and deserializes preprocessing state (feature order, target order, scalers).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Default Feature Sets grounded in combined.csv
DEFAULT_FEATURE_COLUMNS = [
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "GazeYaw",
    "GazePitch",
    "GazeRelativeH",
    "GazeRelativeV",
    "EyeHeadOffset",
    "AbsoluteGazeYaw",
    "AbsoluteGazePitch",
    "LeftOpenness",
    "RightOpenness",
    "BlinkL",
    "BlinkR",
    "SmileL",
    "SmileR",
    "BrowDownL",
    "BrowDownR",
    "JawOpen",
    "EyeClosedL",
    "EyeClosedR",
    "FaceActivity",
]

# Multimodal target vector covering head movement, gaze, and facial activity
DEFAULT_TARGET_COLUMNS = [
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "GazeYaw",
    "GazePitch",
    "GazeRelativeH",
    "GazeRelativeV",
    "EyeHeadOffset",
    "LeftOpenness",
    "RightOpenness",
    "SmileL",
    "SmileR",
    "BrowDownL",
    "BrowDownR",
    "JawOpen",
    "FaceActivity",
]

CIRCULAR_COLUMNS = ["HeadYaw", "GazeYaw", "AbsoluteGazeYaw"]


class MultimodalPreprocessor:
    """Handles feature extraction, circular angle transformation, scaling, and serialization."""

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        use_circular_encoding: bool = True,
    ):
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
        self.target_columns = target_columns or list(DEFAULT_TARGET_COLUMNS)
        self.use_circular_encoding = use_circular_encoding

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        self.engineered_feature_names: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transforms raw dataframe columns into model-ready features.

        Includes sine/cosine features for circular angles when enabled.
        """
        feats = []
        feat_names = []

        for col in self.feature_columns:
            if col not in df.columns:
                raise KeyError(f"Required feature column '{col}' is missing from DataFrame.")

            val = df[col].to_numpy(dtype=np.float64)

            if self.use_circular_encoding and col in CIRCULAR_COLUMNS:
                # Add sin and cos representations
                rad = np.radians(val)
                sin_col = np.sin(rad)
                cos_col = np.cos(rad)

                feats.append(val)
                feat_names.append(col)

                feats.append(sin_col)
                feat_names.append(f"{col}_sin")

                feats.append(cos_col)
                feat_names.append(f"{col}_cos")
            else:
                feats.append(val)
                feat_names.append(col)

        X = np.column_stack(feats)
        return X, feat_names

    def fit(self, train_df: pd.DataFrame) -> "MultimodalPreprocessor":
        """Fits feature and target scalers strictly on the training dataset."""
        X_train, feat_names = self._engineer_features(train_df)
        self.engineered_feature_names = feat_names

        # Extract target matrix
        missing_targets = [t for t in self.target_columns if t not in train_df.columns]
        if missing_targets:
            raise KeyError(f"Target columns missing from training data: {missing_targets}")

        y_train = train_df[self.target_columns].to_numpy(dtype=np.float64)

        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)
        self.is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw observation features using the fitted feature scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")

        X, _ = self._engineer_features(df)
        return self.feature_scaler.transform(X)

    def transform_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms target values using the fitted target scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transform.")

        y = df[self.target_columns].to_numpy(dtype=np.float64)
        return self.target_scaler.transform(y)

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-scales predicted model outputs back to physical coordinate units."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform.")

        y_orig = self.target_scaler.inverse_transform(y_scaled)

        # Wrap circular columns back to [-180, 180]
        y_clamped = np.copy(y_orig)
        for idx, col in enumerate(self.target_columns):
            if col in CIRCULAR_COLUMNS:
                y_clamped[:, idx] = (y_clamped[:, idx] + 180.0) % 360.0 - 180.0
            elif col in ["LeftOpenness", "RightOpenness", "SmileL", "SmileR", "BrowDownL", "BrowDownR", "JawOpen", "EyeClosedL", "EyeClosedR"]:
                # Clamp normalized blendshapes to valid non-negative ranges
                y_clamped[:, idx] = np.clip(y_clamped[:, idx], 0.0, 1.0)
            elif col == "FaceActivity":
                y_clamped[:, idx] = np.clip(y_clamped[:, idx], 0.0, None)
            elif col in ["HeadPitch", "GazePitch", "AbsoluteGazePitch"]:
                y_clamped[:, idx] = np.clip(y_clamped[:, idx], -90.0, 90.0)

        return y_clamped

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves preprocessing parameters and state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "use_circular_encoding": self.use_circular_encoding,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "is_fitted": self.is_fitted,
            "engineered_feature_names": self.engineered_feature_names,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MultimodalPreprocessor":
        """Loads fitted preprocessing state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {path}")

        state = joblib.load(path)
        instance = cls(
            feature_columns=state["feature_columns"],
            target_columns=state["target_columns"],
            use_circular_encoding=state["use_circular_encoding"],
        )
        instance.feature_scaler = state["feature_scaler"]
        instance.target_scaler = state["target_scaler"]
        instance.is_fitted = state["is_fitted"]
        instance.engineered_feature_names = state["engineered_feature_names"]
        return instance
