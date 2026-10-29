"""Sequence Generation Module for Multimodal Viewport Prediction.

Constructs historical sequence windows (lookback) and aligns future targets
using exact timestamps with strict tolerance checks and discontinuity filtering.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessor import MultimodalPreprocessor


class SequenceDataset:
    """Container holding generated sequences, targets, and metadata."""

    def __init__(
        self,
        X_seq: np.ndarray,
        y_targets: np.ndarray,
        history_endpoints: np.ndarray,
        target_timestamps: np.ndarray,
        target_names: List[str],
        feature_names: List[str],
        X_raw_seq: Optional[np.ndarray] = None,
    ):
        self.X = X_seq  # Shape: (N_samples, lookback_window, num_features)
        self.y = y_targets  # Shape: (N_samples, num_targets)
        self.history_endpoints = history_endpoints  # Shape: (N_samples,)
        self.target_timestamps = target_timestamps  # Shape: (N_samples,)
        self.target_names = target_names
        self.feature_names = feature_names
        self.X_raw = X_raw_seq  # Shape: (N_samples, lookback_window, num_raw_features)

    def __len__(self) -> int:
        return len(self.X)


class SequenceGenerator:
    """Builds sliding lookback sequences aligned with future targets based on real timestamps."""

    def __init__(
        self,
        lookback_window: int = 30,
        prediction_horizon_sec: float = 0.5,
        max_target_tolerance_sec: float = 0.05,
        max_internal_gap_sec: float = 0.15,
        stride: int = 2,
    ):
        self.lookback_window = lookback_window
        self.prediction_horizon_sec = prediction_horizon_sec
        self.max_target_tolerance_sec = max_target_tolerance_sec
        self.max_internal_gap_sec = max_internal_gap_sec
        self.stride = stride

    def generate_sequences_from_dataframe(
        self,
        df: pd.DataFrame,
        preprocessor: MultimodalPreprocessor,
        time_column: str = "PlaybackTime",
    ) -> SequenceDataset:
        """Generates sequences from a single continuous recording DataFrame.

        STRICT: Never called across person/video boundaries.
        """
        # Ensure chronological ordering
        if not df["Timestamp"].is_monotonic_increasing:
            df = df.sort_values("Timestamp").reset_index(drop=True)

        n_rows = len(df)
        if n_rows < self.lookback_window + 1:
            raise ValueError(
                f"DataFrame has only {n_rows} rows, which is less than required lookback ({self.lookback_window} + 1)."
            )

        # Transform features and extract targets
        X_features = preprocessor.transform_features(df)
        X_raw_features = df[preprocessor.feature_columns].to_numpy(dtype=np.float32)
        y_targets_raw = df[preprocessor.target_columns].to_numpy(dtype=np.float64)

        # Timestamps for alignment
        t_align = df[time_column].to_numpy(dtype=np.float64)
        ts_actual = df["Timestamp"].to_numpy(dtype=np.float64)

        X_seq_list = []
        X_raw_list = []
        y_list = []
        endpoints_list = []
        target_ts_list = []

        W = self.lookback_window
        horizon = self.prediction_horizon_sec
        tol = self.max_target_tolerance_sec
        max_gap = self.max_internal_gap_sec

        # Slide through historical indices
        for i in range(W - 1, n_rows, self.stride):
            # Check internal historical continuity
            hist_dts = ts_actual[i - W + 2 : i + 1] - ts_actual[i - W + 1 : i]
            if np.any(hist_dts > max_gap):
                # Discontinuity or sensor freeze in history; discard sequence
                continue

            t_endpoint = t_align[i]
            t_target_desired = t_endpoint + horizon

            # Binary search for closest timestamp at or after desired time
            idx_candidate = int(np.searchsorted(t_align, t_target_desired))

            # Inspect candidate and candidate - 1 to find the closest sample
            best_idx = None
            best_diff = float("inf")

            for cand in [idx_candidate - 1, idx_candidate, idx_candidate + 1]:
                if 0 <= cand < n_rows:
                    diff = abs(t_align[cand] - t_target_desired)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = cand

            if best_idx is not None and best_diff <= tol and best_idx > i:
                # Valid sequence found!
                seq_window = X_features[i - W + 1 : i + 1]
                raw_window = X_raw_features[i - W + 1 : i + 1]
                target_val = y_targets_raw[best_idx]

                X_seq_list.append(seq_window)
                X_raw_list.append(raw_window)
                y_list.append(target_val)
                endpoints_list.append(t_endpoint)
                target_ts_list.append(t_align[best_idx])

        if not X_seq_list:
            raise ValueError(
                f"No valid sequences generated for horizon={horizon}s with tolerance={tol}s."
            )

        return SequenceDataset(
            X_seq=np.array(X_seq_list, dtype=np.float32),
            y_targets=np.array(y_list, dtype=np.float32),
            history_endpoints=np.array(endpoints_list, dtype=np.float64),
            target_timestamps=np.array(target_ts_list, dtype=np.float64),
            target_names=preprocessor.target_columns,
            feature_names=preprocessor.engineered_feature_names,
            X_raw_seq=np.array(X_raw_list, dtype=np.float32),
        )

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        preprocessor: MultimodalPreprocessor,
        train_ratio: float = 0.8,
        time_column: str = "PlaybackTime",
    ) -> Tuple[SequenceDataset, SequenceDataset]:
        """Performs a strict chronological train/test split on a recording.

        First fits the preprocessor on train_df ONLY, then generates sequences for both.
        """
        n_rows = len(df)
        split_idx = int(n_rows * train_ratio)

        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        # Fit preprocessor on training slice ONLY
        preprocessor.fit(train_df)

        train_dataset = self.generate_sequences_from_dataframe(
            train_df, preprocessor, time_column=time_column
        )
        test_dataset = self.generate_sequences_from_dataframe(
            test_df, preprocessor, time_column=time_column
        )

        return train_dataset, test_dataset
