#!/usr/bin/env python
"""
Comparación Random Forest con GridSearchCV (sección 5.3 del pipeline):
vec_direction directo vs sin/cos + error circular.

Ejecutar:
    python3 rf_direction_circular_test.py
    python3 rf_direction_visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from circular_direction import (
    direction_to_sincos,
    error_angular,
    metricas_circulares,
    sincos_to_direction,
)
from dataset_loader import load_direction_train_test

_SCRIPT_DIR = Path(__file__).resolve().parent

# Misma grilla que tesis_fisico_secuential.py sección 5.3
PARAM_GRID = {
    "estimator__rfr__n_estimators": [50, 100, 200],
    "estimator__rfr__max_depth": [None, 5, 10],
    "estimator__rfr__min_samples_leaf": [1, 2],
}
CV = GroupKFold(n_splits=7)


def metricas_regresion_robustas(y_true, y_pred, eps=1e-6, umbral_mape=0.5):
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


def _build_gridsearch_multioutput():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rfr", RandomForestRegressor(random_state=42)),
    ])
    multi_pipe = MultiOutputRegressor(pipe)
    return GridSearchCV(
        multi_pipe,
        PARAM_GRID,
        cv=CV,
        scoring="neg_mean_squared_error",
        n_jobs=2,
        verbose=1,
    )


def entrenar_rf_direccion_directa(X_train, y_train, grupos_train):
    """Enfoque ANTES: igual que sección 5.3 — vec_direction en radianes, GridSearchCV."""
    grid = _build_gridsearch_multioutput()
    grid.fit(X_train, y_train, groups=grupos_train)
    print(f"  Mejores params (antes): {grid.best_params_}")
    print(f"  CV score (antes): {grid.best_score_:.4f}")
    return grid.best_estimator_


def entrenar_rf_direccion_sincos(X_train, y_train, grupos_train):
    """Enfoque DESPUÉS: sin/cos con GridSearchCV (sección 5.3.1)."""
    y_sincos = direction_to_sincos(y_train)
    grid = _build_gridsearch_multioutput()
    grid.fit(X_train, y_sincos, groups=grupos_train)
    print(f"  Mejores params (sin/cos): {grid.best_params_}")
    print(f"  CV score (sin/cos): {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluar_enfoque_directo(y_test, y_pred):
    m_lineal = metricas_regresion_robustas(y_test, y_pred)
    m_circular = metricas_circulares(y_test, y_pred)
    return {**m_lineal, **m_circular}


def evaluar_enfoque_sincos(y_test, y_pred_sincos):
    y_pred = sincos_to_direction(y_pred_sincos)
    m_lineal = metricas_regresion_robustas(y_test, y_pred)
    m_circular = metricas_circulares(y_test, y_pred)
    return {**m_lineal, **m_circular}, y_pred


def tabla_comparativa(metricas_antes: dict, metricas_despues: dict) -> pd.DataFrame:
    columnas = [
        "RMSE", "MAE", "R2", "MAPE_%_filtrado", "sMAPE_%",
        "RMSE_circular_rad", "MAE_circular_rad",
        "RMSE_circular_deg", "MAE_circular_deg",
    ]
    filas = []
    for nombre, m in [
        ("Antes (vec_direction directo)", metricas_antes),
        ("Después (sin/cos + arctan2)", metricas_despues),
    ]:
        filas.append({"Enfoque": nombre, **{c: m.get(c, np.nan) for c in columnas}})
    return pd.DataFrame(filas)


def main():
    print("=" * 70)
    print("Random Forest + GridSearchCV — Dirección: directo vs sin/cos")
    print("Datos: HURDAT2 + SHIPS + NAO")
    print("=" * 70)

    print("\nCargando datos reales...")
    X_train, y_train, g_train, X_test, y_test, _, info = load_direction_train_test()
    print(f"  Filas EDA: {info['n_rows_eda']} | Tormentas: {info['n_storms_train']}/{info['n_storms_test']}")
    print(f"  Secuencias train/test: {X_train.shape[0]} / {X_test.shape[0]}")

    print("\n[1/2] GridSearchCV — ANTES (vec_direction directo)...")
    model_antes = entrenar_rf_direccion_directa(X_train, y_train, g_train)
    y_pred_antes = model_antes.predict(X_test)
    metricas_antes = evaluar_enfoque_directo(y_test, y_pred_antes)

    print("\n[2/2] GridSearchCV — DESPUÉS (sin/cos + arctan2)...")
    model_despues = entrenar_rf_direccion_sincos(X_train, y_train, g_train)
    y_pred_sincos = model_despues.predict(X_test)
    metricas_despues, y_pred_despues = evaluar_enfoque_sincos(y_test, y_pred_sincos)

    df_comp = tabla_comparativa(metricas_antes, metricas_despues)
    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA — Random Forest + GridSearchCV | Dirección")
    print("=" * 70)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df_comp.to_string(index=False))

    mejora_rmse = (
        (metricas_antes["RMSE_circular_deg"] - metricas_despues["RMSE_circular_deg"])
        / metricas_antes["RMSE_circular_deg"] * 100
    )
    mejora_mae = (
        (metricas_antes["MAE_circular_deg"] - metricas_despues["MAE_circular_deg"])
        / metricas_antes["MAE_circular_deg"] * 100
    )
    print(f"\n  RMSE circular: {metricas_antes['RMSE_circular_deg']:.2f}° → "
          f"{metricas_despues['RMSE_circular_deg']:.2f}° ({mejora_rmse:+.1f}%)")
    print(f"  MAE  circular: {metricas_antes['MAE_circular_deg']:.2f}° → "
          f"{metricas_despues['MAE_circular_deg']:.2f}° ({mejora_mae:+.1f}%)")

    out_csv = _SCRIPT_DIR / "comparacion_rf_direccion_circular.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"\nTabla guardada: {out_csv}")

    # Guardar predicciones para visualización
    np.savez(
        _SCRIPT_DIR / "predicciones_direccion_test.npz",
        y_test=y_test,
        y_pred_antes=y_pred_antes,
        y_pred_despues=y_pred_despues,
        metricas_antes=metricas_antes,
        metricas_despues=metricas_despues,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
