"""
Utilidades para tratar la dirección como variable circular (sin/cos + métricas angulares).
"""

from __future__ import annotations

import numpy as np


def error_angular(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Error angular mínimo en radianes, en el rango (-π, π]."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return np.arctan2(np.sin(diff), np.cos(diff))


def direction_to_sincos(y_direction: np.ndarray) -> np.ndarray:
    """
    Convierte ángulos (rad) en targets sin/cos intercalados por horizonte.

    Entrada:  (n_samples, horizon) o (n_samples,)
    Salida:   (n_samples, 2 * horizon) con columnas [sin_h1, cos_h1, sin_h2, cos_h2, ...]
    """
    y = np.asarray(y_direction)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_sin = np.sin(y)
    y_cos = np.cos(y)
    n_samples, horizon = y.shape
    y_sincos = np.empty((n_samples, 2 * horizon), dtype=float)
    y_sincos[:, 0::2] = y_sin
    y_sincos[:, 1::2] = y_cos
    return y_sincos


def sincos_to_direction(y_sincos: np.ndarray) -> np.ndarray:
    """
    Reconstruye ángulos en (-π, π] a partir de sin/cos predichos.

    Entrada:  (n_samples, 2 * horizon) intercalado [sin, cos, sin, cos, ...]
    Salida:   (n_samples, horizon)
    """
    y = np.asarray(y_sincos)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    sin_pred = y[:, 0::2]
    cos_pred = y[:, 1::2]
    return np.arctan2(sin_pred, cos_pred)


def metricas_circulares(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE y RMSE circulares (en radianes) usando error_angular."""
    err = error_angular(y_true, y_pred).ravel()
    mae_circular = float(np.mean(np.abs(err)))
    rmse_circular = float(np.sqrt(np.mean(err**2)))
    return {
        "MAE_circular_rad": mae_circular,
        "RMSE_circular_rad": rmse_circular,
        "MAE_circular_deg": float(np.degrees(mae_circular)),
        "RMSE_circular_deg": float(np.degrees(rmse_circular)),
    }
