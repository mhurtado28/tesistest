#!/usr/bin/env python
"""
Comparación Random Forest: dirección directa (rad estandarizada en X) vs sin/cos + error circular.

Ejecutar desde el directorio del proyecto:
    python rf_direction_circular_test.py

Si existen los artefactos del pipeline principal (*.pkl y datos procesados), los reutiliza.
Si no, genera datos sintéticos con la misma estructura (secuencias, horizon=12).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dataset_loader import load_direction_train_test
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from circular_direction import (
    direction_to_sincos,
    error_angular,
    metricas_circulares,
    sincos_to_direction,
)

_SCRIPT_DIR = Path(__file__).resolve().parent


def metricas_regresion_robustas(y_true, y_pred, eps=1e-6, umbral_mape=1.0):
    """Mismas métricas lineales que en tesis_fisico_secuential.py."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) >= umbral_mape
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
        if mask.any()
        else np.nan
    )
    smape = (
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))
        * 100
    )
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_%_filtrado": mape,
        "sMAPE_%": smape,
    }


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def create_sequences_synthetic(
    n_storms: int = 50,
    min_len: int = 30,
    max_len: int = 50,
    timesteps: int = 12,
    horizon: int = 12,
    n_features: int = 12,
    random_state: int = 42,
):
    """Genera secuencias sintéticas con ángulos circulares y features correlacionadas."""
    rng = np.random.default_rng(random_state)
    X_dir_list, y_dir_list, grupos = [], [], []

    for storm_idx in range(n_storms):
        storm_id = f"AL{storm_idx:04d}"
        seq_len = rng.integers(min_len, max_len + 1)

        # Trayectoria suave con componente circular
        t = np.arange(seq_len, dtype=float)
        base_angle = 0.3 * t + 0.5 * np.sin(t / 5)
        noise = rng.normal(0, 0.15, seq_len)
        angles = np.arctan2(np.sin(base_angle + noise), np.cos(base_angle + noise))

        features = rng.normal(size=(seq_len, n_features))
        # Correlacionar algunas features con el ángulo
        features[:, 0] = np.sin(angles) + rng.normal(0, 0.1, seq_len)
        features[:, 1] = np.cos(angles) + rng.normal(0, 0.1, seq_len)
        features[:, 2] = rng.normal(0, 1, seq_len) + 0.3 * np.sin(angles)

        if seq_len < timesteps + horizon:
            continue

        for i in range(seq_len - timesteps - horizon + 1):
            X_dir_list.append(features[i : i + timesteps])
            y_dir_list.append(angles[i + timesteps : i + timesteps + horizon])
            grupos.append(storm_id)

    X_dir = np.array(X_dir_list)
    y_dir = np.array(y_dir_list)
    grupos = np.array(grupos)
    return X_dir, y_dir, grupos


def split_by_storm(X, y, grupos, test_frac=0.10):
    ids = np.unique(grupos)
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_ids = set(ids[-n_test:])
    mask_test = np.isin(grupos, list(test_ids))
    return (
        X[~mask_test],
        y[~mask_test],
        grupos[~mask_test],
        X[mask_test],
        y[mask_test],
        grupos[mask_test],
    )


def build_rf_model(estimar_target: bool = False):
    """RF con StandardScaler en X; opcionalmente también estandariza y (vec_direction)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rfr",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    if estimar_target:
        pipe = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler(),
        )
    return MultiOutputRegressor(pipe)


def entrenar_rf_direccion_directa(X_train, y_train, grupos_train):
    """Enfoque ANTES: vec_direction estandarizado con StandardScaler y predicho directamente."""
    model = build_rf_model(estimar_target=True)
    model.fit(X_train, y_train)
    return model


def entrenar_rf_direccion_sincos(X_train, y_train, grupos_train):
    """Enfoque DESPUÉS: predice sin/cos con MultiOutputRegressor (2 outputs por horizonte)."""
    y_sincos = direction_to_sincos(y_train)
    model = build_rf_model(estimar_target=False)
    model.fit(X_train, y_sincos)
    return model


def evaluar_enfoque_directo(y_test, y_pred):
    """Métricas lineales + circulares sobre predicción directa en radianes."""
    m_lineal = metricas_regresion_robustas(y_test, y_pred, umbral_mape=0.5)
    m_circular = metricas_circulares(y_test, y_pred)
    return {**m_lineal, **m_circular}


def evaluar_enfoque_sincos(y_test, y_pred_sincos):
    """Reconstruye ángulo y calcula métricas circulares (+ lineales de referencia)."""
    y_pred = sincos_to_direction(y_pred_sincos)
    m_lineal = metricas_regresion_robustas(y_test, y_pred, umbral_mape=0.5)
    m_circular = metricas_circulares(y_test, y_pred)
    return {**m_lineal, **m_circular}, y_pred


def cargar_datos_pipeline():
    """
    Intenta cargar datos reales del pipeline principal.
    Retorna (X_train, y_train, grupos_train, X_test, y_test) o None.
    """
    pkl_dir = _SCRIPT_DIR
    model_path = pkl_dir / "rfr_multioutput_secuential_dir.pkl"
    if not model_path.exists():
        return None

    # El modelo guardado implica que ya se corrió el pipeline; sin arrays en disco,
    # no podemos reconstruir X/y. Retornamos None para usar sintéticos.
    return None


def tabla_comparativa(metricas_antes: dict, metricas_despues: dict) -> pd.DataFrame:
    filas = []
    columnas = [
        "RMSE",
        "MAE",
        "R2",
        "MAPE_%_filtrado",
        "sMAPE_%",
        "RMSE_circular_rad",
        "MAE_circular_rad",
        "RMSE_circular_deg",
        "MAE_circular_deg",
    ]
    for nombre, m in [
        ("Antes (vec_direction estandarizado)", metricas_antes),
        ("Después (sin/cos + arctan2)", metricas_despues),
    ]:
        fila = {"Enfoque": nombre}
        for col in columnas:
            fila[col] = m.get(col, np.nan)
        filas.append(fila)
    return pd.DataFrame(filas)


def demostrar_error_angular():
    """Muestra que error_angular corrige la discontinuidad en ±π."""
    y_true = np.array([np.pi - 0.1])
    y_pred = np.array([-np.pi + 0.1])
    err_lineal = y_true - y_pred
    err_circ = error_angular(y_true, y_pred)
    print("\n--- Demostración discontinuidad ±π ---")
    print(f"  y_true = {y_true[0]:+.4f} rad, y_pred = {y_pred[0]:+.4f} rad")
    print(f"  Error lineal (abs): {np.abs(err_lineal[0]):.4f} rad  (~{np.degrees(np.abs(err_lineal[0])):.1f}°)")
    print(f"  Error circular (abs): {np.abs(err_circ[0]):.4f} rad  (~{np.degrees(np.abs(err_circ[0])):.1f}°)")


def main():
    print("=" * 70)
    print("Random Forest — Comparación dirección directa vs sin/cos (circular)")
    print("Datos: HURDAT2 + SHIPS + NAO")
    print("=" * 70)

    demostrar_error_angular()

    print("\nCargando y preparando datos reales...")
    X_train, y_train, g_train, X_test, y_test, g_test, info = load_direction_train_test()
    print(f"  Filas EDA (con SHIPS): {info['n_rows_eda']}")
    print(f"  Tormentas train/test: {info['n_storms_train']} / {info['n_storms_test']}")
    print(f"\nTrain: {X_train.shape[0]} secuencias | Test: {X_test.shape[0]} secuencias")
    print(f"Horizonte: {info['horizon']} pasos | Features aplanadas: {X_train.shape[1]}")

    print("\n[1/2] Entrenando RF — ANTES (vec_direction estandarizado, predicción directa)...")
    model_directo = entrenar_rf_direccion_directa(X_train, y_train, g_train)
    y_pred_directo = model_directo.predict(X_test)
    metricas_antes = evaluar_enfoque_directo(y_test, y_pred_directo)

    print("[2/2] Entrenando RF — enfoque DESPUÉS (sin/cos + arctan2)...")
    model_sincos = entrenar_rf_direccion_sincos(X_train, y_train, g_train)
    y_pred_sincos = model_sincos.predict(X_test)
    metricas_despues, y_pred_circular = evaluar_enfoque_sincos(y_test, y_pred_sincos)

    df_comp = tabla_comparativa(metricas_antes, metricas_despues)

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA — Random Forest | Target: Dirección")
    print("=" * 70)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df_comp.to_string(index=False))

    print("\n--- Interpretación ---")
    mejora_rmse = (
        (metricas_antes["RMSE_circular_rad"] - metricas_despues["RMSE_circular_rad"])
        / metricas_antes["RMSE_circular_rad"]
        * 100
    )
    mejora_mae = (
        (metricas_antes["MAE_circular_rad"] - metricas_despues["MAE_circular_rad"])
        / metricas_antes["MAE_circular_rad"]
        * 100
    )
    print(
        f"  RMSE circular: {metricas_antes['RMSE_circular_deg']:.2f}° → "
        f"{metricas_despues['RMSE_circular_deg']:.2f}°  ({mejora_rmse:+.1f}%)"
    )
    print(
        f"  MAE  circular: {metricas_antes['MAE_circular_deg']:.2f}° → "
        f"{metricas_despues['MAE_circular_deg']:.2f}°  ({mejora_mae:+.1f}%)"
    )
    print(
        f"  R² lineal (referencia): {metricas_antes['R2']:.4f} → {metricas_despues['R2']:.4f}"
    )
    print(
        f"  sMAPE lineal: {metricas_antes['sMAPE_%']:.1f}% → {metricas_despues['sMAPE_%']:.1f}%"
    )

    out_csv = _SCRIPT_DIR / "comparacion_rf_direccion_circular.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"\nTabla guardada en: {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
