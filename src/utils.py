"""Utility module for angular calculations, coordinate math, and evaluation metrics.

Ground-truth handling:
- Proper wrapped angular difference for circular degrees [-180, +180].
- 3D viewing direction vector conversions from spherical yaw/pitch.
- Great-circle angular distance (orthodromic error) between viewports.
- Standard regression metrics (MAE, RMSE, Relative Error).
"""

from typing import Dict, Tuple, Union
import numpy as np


def wrap_angle_deg(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wraps an angle in degrees into the standard interval [-180.0, +180.0)."""
    return (angle + 180.0) % 360.0 - 180.0


def angular_difference_deg(
    pred_deg: Union[float, np.ndarray], true_deg: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Computes the shortest wrapped absolute angular error in degrees between pred and true.

    Example: +179° vs -179° yields 2.0°, NOT 358°.
    """
    diff = np.radians(pred_deg) - np.radians(true_deg)
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.degrees(np.abs(wrapped))


def euler_to_direction_vector(
    yaw_deg: Union[float, np.ndarray], pitch_deg: Union[float, np.ndarray]
) -> np.ndarray:
    """Converts spherical yaw and pitch (in degrees) to 3D Cartesian unit direction vectors [x, y, z].

    x = cos(pitch) * sin(yaw)
    y = sin(pitch)
    z = cos(pitch) * cos(yaw)
    """
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)

    x = np.cos(pitch_rad) * np.sin(yaw_rad)
    y = np.sin(pitch_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)

    if np.isscalar(yaw_deg):
        vec = np.array([x, y, z], dtype=np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    vecs = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def orthodromic_angular_distance_deg(
    pred_yaw_deg: Union[float, np.ndarray],
    pred_pitch_deg: Union[float, np.ndarray],
    true_yaw_deg: Union[float, np.ndarray],
    true_pitch_deg: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculates the 3D great-circle angular distance (in degrees) between predicted

    and actual viewing directions (orthodromic viewing angle error).
    """
    v_pred = euler_to_direction_vector(pred_yaw_deg, pred_pitch_deg)
    v_true = euler_to_direction_vector(true_yaw_deg, true_pitch_deg)

    # Dot product clamped to [-1.0, 1.0] to prevent numerical issues
    if v_pred.ndim == 1:
        dot = np.clip(np.dot(v_pred, v_true), -1.0, 1.0)
    else:
        dot = np.clip(np.sum(v_pred * v_true, axis=-1), -1.0, 1.0)

    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def compute_metrics(
    y_pred: np.ndarray, y_true: np.ndarray, target_names: list
) -> Dict[str, Dict[str, float]]:
    """Computes comprehensive regression metrics for multi-target predictions.

    Distinguishes circular angular targets (using wrapped error) from standard numeric targets.
    """
    metrics = {}
    circular_targets = {"HeadYaw", "GazeYaw", "AbsoluteGazeYaw"}

    for idx, name in enumerate(target_names):
        pred_col = y_pred[:, idx]
        true_col = y_true[:, idx]

        if name in circular_targets:
            err = angular_difference_deg(pred_col, true_col)
            mae = float(np.mean(err))
            rmse = float(np.sqrt(np.mean(err**2)))
            max_err = float(np.max(err))
        else:
            diff = pred_col - true_col
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            max_err = float(np.max(np.abs(diff)))

        metrics[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MaxError": round(max_err, 4),
            "is_circular": name in circular_targets,
        }

    # If both HeadYaw and HeadPitch exist, compute Great Circle Viewport Error
    if "HeadYaw" in target_names and "HeadPitch" in target_names:
        y_idx = target_names.index("HeadYaw")
        p_idx = target_names.index("HeadPitch")
        ortho = orthodromic_angular_distance_deg(
            y_pred[:, y_idx], y_pred[:, p_idx], y_true[:, y_idx], y_true[:, p_idx]
        )
        metrics["ViewportDirection"] = {
            "MeanGreatCircleErrorDeg": round(float(np.mean(ortho)), 4),
            "MedianGreatCircleErrorDeg": round(float(np.median(ortho)), 4),
            "RMSEGreatCircleDeg": round(float(np.sqrt(np.mean(ortho**2))), 4),
            "MaxGreatCircleErrorDeg": round(float(np.max(ortho)), 4),
        }

    # If both GazeYaw and GazePitch exist, compute Gaze Direction Error
    if "GazeYaw" in target_names and "GazePitch" in target_names:
        gy_idx = target_names.index("GazeYaw")
        gp_idx = target_names.index("GazePitch")
        gaze_ortho = orthodromic_angular_distance_deg(
            y_pred[:, gy_idx], y_pred[:, gp_idx], y_true[:, gy_idx], y_true[:, gp_idx]
        )
        metrics["GazeDirection"] = {
            "MeanGreatCircleErrorDeg": round(float(np.mean(gaze_ortho)), 4),
            "MedianGreatCircleErrorDeg": round(float(np.median(gaze_ortho)), 4),
            "RMSEGreatCircleDeg": round(float(np.sqrt(np.mean(gaze_ortho**2))), 4),
            "MaxGreatCircleErrorDeg": round(float(np.max(gaze_ortho)), 4),
        }

    return metrics


if __name__ == "__main__":
    # Test wrapped angular difference
    d1 = angular_difference_deg(179.0, -179.0)
    print("Wrapped diff (+179 vs -179):", d1, "degrees (Expected: 2.0)")
    assert abs(d1 - 2.0) < 1e-4

    d2 = angular_difference_deg(0.0, 360.0)
    print("Wrapped diff (0 vs 360):", d2, "degrees (Expected: 0.0)")
    assert abs(d2 - 0.0) < 1e-4

    orth = orthodromic_angular_distance_deg(0.0, 0.0, 90.0, 0.0)
    print("Orthodromic (0,0) vs (90,0):", orth, "degrees (Expected: 90.0)")
    assert abs(orth - 90.0) < 1e-4
    print("All math tests passed successfully.")
