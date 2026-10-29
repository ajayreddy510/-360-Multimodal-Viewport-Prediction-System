"""Model Architectures for Multimodal Future Movement Prediction.

Implements three models for rigorous comparison:
1. PersistenceBaselineModel: Predicts that measurements remain at their last observed historical values.
2. MachineLearningBaselineModel: Multi-output tabular regressor (Random Forest / Ridge) over historical dynamics.
3. DeepSequenceModel: PyTorch GRU / LSTM recurrent neural network trained on multi-modal sequence trajectories.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import wrap_angle_deg


class PersistenceBaselineModel:
    """Persistence baseline: predicts that future values equal the last observed values."""

    def __init__(self, target_names: List[str], raw_feature_names: List[str]):
        self.target_names = target_names
        self.raw_feature_names = raw_feature_names

        # Map each target to its column index in the raw feature window
        self.target_to_feat_idx = {}
        for t_idx, t_name in enumerate(self.target_names):
            if t_name in self.raw_feature_names:
                self.target_to_feat_idx[t_idx] = self.raw_feature_names.index(t_name)

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "PersistenceBaselineModel":
        """No parameters to learn for persistence."""
        return self

    def predict(self, X_seq_unscaled: np.ndarray) -> np.ndarray:
        """Predicts the last observed value from the historical sequence.

        X_seq_unscaled shape: (N_samples, lookback, num_raw_features)
        Returns: (N_samples, num_targets)
        """
        n_samples = len(X_seq_unscaled)
        n_targets = len(self.target_names)
        preds = np.zeros((n_samples, n_targets), dtype=np.float32)

        last_step = X_seq_unscaled[:, -1, :]  # Shape: (N_samples, num_raw_features)

        for t_idx in range(n_targets):
            if t_idx in self.target_to_feat_idx:
                feat_col = self.target_to_feat_idx[t_idx]
                preds[:, t_idx] = last_step[:, feat_col]
            else:
                preds[:, t_idx] = 0.0

        return preds


class MachineLearningBaselineModel:
    """Tabular ML baseline using summary and endpoint statistics across the lookback window."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.regressor = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        self.is_fitted = False

    def _extract_tabular_features(self, X_seq: np.ndarray) -> np.ndarray:
        """Extracts informative summary features from sequence: last, mean, std, velocity (last - first).

        X_seq shape: (N, W, D) -> returns shape: (N, D * 4)
        """
        last_val = X_seq[:, -1, :]
        first_val = X_seq[:, 0, :]
        mean_val = np.mean(X_seq, axis=1)
        velocity = last_val - first_val
        std_val = np.std(X_seq, axis=1)
        return np.column_stack([last_val, velocity, mean_val, std_val])

    def fit(self, X_seq: np.ndarray, delta_y: np.ndarray) -> "MachineLearningBaselineModel":
        """Fits multi-output regressor on residual movement deltas."""
        X_tab = self._extract_tabular_features(X_seq)
        self.regressor.fit(X_tab, delta_y)
        self.is_fitted = True
        return self

    def predict(self, X_seq: np.ndarray, y_last: np.ndarray) -> np.ndarray:
        """Predicts future targets by adding learned delta to the last observed values."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X_tab = self._extract_tabular_features(X_seq)
        pred_delta = self.regressor.predict(X_tab)
        preds = y_last + pred_delta
        return preds

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.regressor, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MachineLearningBaselineModel":
        instance = cls()
        instance.regressor = joblib.load(filepath)
        instance.is_fitted = True
        return instance


class PyTorchSequenceNet(nn.Module):
    """PyTorch Deep Sequence Neural Network using GRU / LSTM cells with multi-target projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 16,
        dropout: float = 0.2,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        if self.rnn_type == "LSTM":
            out, (hn, cn) = self.rnn(x)
        else:
            out, hn = self.rnn(x)

        # Use last timestep hidden state
        last_hidden = out[:, -1, :]
        preds = self.head(last_hidden)
        return preds


class DeepSequenceModel:
    """Wrapper managing training, evaluation, and inference for PyTorchSequenceNet."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "GRU",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = PyTorchSequenceNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=0.2,
            rnn_type=rnn_type,
        ).to(self.device)

        # Initialize output projection layer with small weights for smooth residual adaptation
        nn.init.normal_(self.net.head[-1].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.net.head[-1].bias)

        self.criterion = nn.SmoothL1Loss()  # Huber loss (robust to outliers)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 25,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Trains the sequence neural network using AdamW and Huber loss."""
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.net.train()
            total_train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                preds = self.net(bx)
                loss = self.criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_train_loss += loss.item() * len(bx)

            avg_train_loss = total_train_loss / len(X_train)
            history["train_loss"].append(avg_train_loss)

            val_str = ""
            if X_val is not None and y_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vx = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    vy = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    v_preds = self.net(vx)
                    v_loss = self.criterion(v_preds, vy).item()
                    history["val_loss"].append(v_loss)
                    val_str = f" | Val Loss: {v_loss:.5f}"

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.5f}{val_str}")

        return history

    def predict(self, X_seq: np.ndarray, y_last: Optional[np.ndarray] = None) -> np.ndarray:
        """Inference forward pass. When y_last is provided, reconstructs future targets via residual addition."""
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            preds = self.net(x_tensor).cpu().numpy()
            if y_last is not None:
                return y_last + preds
            return preds

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "rnn_type": self.rnn_type,
                "model_state_dict": self.net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path], device: Optional[str] = None) -> "DeepSequenceModel":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PyTorch model not found at {path}")

        checkpoint = torch.load(path, map_location=device or "cpu")
        instance = cls(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            rnn_type=checkpoint["rnn_type"],
            device=device,
        )
        instance.net.load_state_dict(checkpoint["model_state_dict"])
        instance.net.eval()
        return instance
