#!/usr/bin/env python
"""
Visualización de predicciones de vec_direction: real vs modelo estandarizado vs sin/cos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from circular_direction import error_angular, metricas_circulares, sincos_to_direction
from dataset_loader import load_direction_train_test
from rf_direction_circular_test import (
    entrenar_rf_direccion_directa,
    entrenar_rf_direccion_sincos,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _SCRIPT_DIR / "plots_direccion"


def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Desenvuelve ángulos para graficar series temporales sin salto ±π."""
    return np.unwrap(np.asarray(angles, dtype=float))


def plot_secuencias_horizonte(
    y_true: np.ndarray,
    y_pred_antes: np.ndarray,
    y_pred_despues: np.ndarray,
    indices: list[int],
    out_path: Path,
):
    """Serie temporal por horizonte (t+1 … t+12) para varias secuencias de test."""
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    pasos = np.arange(1, y_true.shape[1] + 1)

    for ax, idx in zip(axes, indices):
        yt = _unwrap_angles(y_true[idx])
        ya = _unwrap_angles(y_pred_antes[idx])
        yd = _unwrap_angles(y_pred_despues[idx])

        ax.plot(pasos, np.degrees(yt), "k-o", linewidth=2, markersize=5, label="Real (vec_direction)")
        ax.plot(pasos, np.degrees(ya), "s--", color="#E74C3C", linewidth=1.8, markersize=4, label="Antes (estandarizado)")
        ax.plot(pasos, np.degrees(yd), "^--", color="#2ECC71", linewidth=1.8, markersize=4, label="Después (sin/cos)")

        err_a = np.degrees(np.abs(error_angular(y_true[idx], y_pred_antes[idx])))
        err_d = np.degrees(np.abs(error_angular(y_true[idx], y_pred_despues[idx])))
        ax.set_title(
            f"Secuencia test #{idx}  |  MAE circular: "
            f"antes={err_a.mean():.1f}°  después={err_d.mean():.1f}°",
            fontsize=11,
        )
        ax.set_ylabel("Ángulo (°)")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Horizonte (t+1 … t+h)")
    fig.suptitle("Predicción de vec_direction — comparación por horizonte", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_real_vs_pred(
    y_true: np.ndarray,
    y_pred_antes: np.ndarray,
    y_pred_despues: np.ndarray,
    out_path: Path,
):
    """Dispersión ángulo real vs predicho (todos los puntos de test)."""
    yt = y_true.ravel()
    ya = y_pred_antes.ravel()
    yd = y_pred_despues.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, yp, titulo, color in zip(
        axes,
        [ya, yd],
        ["Antes (vec_direction estandarizado)", "Después (sin/cos + arctan2)"],
        ["#E74C3C", "#2ECC71"],
    ):
        ax.scatter(np.degrees(yt), np.degrees(yp), alpha=0.35, s=18, c=color, edgecolors="none")
        lim = (-190, 190)
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ángulo real (°)")
        ax.set_ylabel("Ángulo predicho (°)")
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

        m = metricas_circulares(yt, yp)
        ax.text(
            0.04, 0.96,
            f"MAE circ. = {m['MAE_circular_deg']:.1f}°\nRMSE circ. = {m['RMSE_circular_deg']:.1f}°",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Real vs predicho — todas las secuencias y horizontes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_polar_secuencia(
    y_true: np.ndarray,
    y_pred_antes: np.ndarray,
    y_pred_despues: np.ndarray,
    idx: int,
    out_path: Path,
):
    """Gráfico polar de una secuencia: real vs ambos modelos."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    t = np.linspace(0, 2 * np.pi, y_true.shape[1], endpoint=False)
    yt, ya, yd = y_true[idx], y_pred_antes[idx], y_pred_despues[idx]

    ax.plot(t, np.ones_like(t), "k.", alpha=0.15, markersize=20)
    ax.plot(t, yt, "k-o", linewidth=2, markersize=7, label="Real")
    ax.plot(t, ya, "s--", color="#E74C3C", linewidth=1.8, markersize=6, label="Antes")
    ax.plot(t, yd, "^--", color="#2ECC71", linewidth=1.8, markersize=6, label="Después")

    ax.set_title(f"Dirección en coordenadas polares — secuencia #{idx}", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_angular_hist(
    y_true: np.ndarray,
    y_pred_antes: np.ndarray,
    y_pred_despues: np.ndarray,
    out_path: Path,
):
    """Histograma del error angular absoluto para ambos modelos."""
    err_a = np.degrees(np.abs(error_angular(y_true, y_pred_antes))).ravel()
    err_d = np.degrees(np.abs(error_angular(y_true, y_pred_despues))).ravel()

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(err_a.max(), err_d.max(), 1), 35)
    ax.hist(err_a, bins=bins, alpha=0.55, color="#E74C3C", label=f"Antes (mediana={np.median(err_a):.1f}°)")
    ax.hist(err_d, bins=bins, alpha=0.55, color="#2ECC71", label=f"Después (mediana={np.median(err_d):.1f}°)")
    ax.set_xlabel("Error angular absoluto (°)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del error angular — Random Forest | Dirección")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_panel_resumen(
    y_true: np.ndarray,
    y_pred_antes: np.ndarray,
    y_pred_despues: np.ndarray,
    idx: int,
    out_path: Path,
):
    """Panel único con serie, polar e histograma de errores de una secuencia."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    pasos = np.arange(1, y_true.shape[1] + 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pasos, np.degrees(_unwrap_angles(y_true[idx])), "k-o", label="Real")
    ax1.plot(pasos, np.degrees(_unwrap_angles(y_pred_antes[idx])), "s--", color="#E74C3C", label="Antes")
    ax1.plot(pasos, np.degrees(_unwrap_angles(y_pred_despues[idx])), "^--", color="#2ECC71", label="Después")
    ax1.set_xlabel("Horizonte")
    ax1.set_ylabel("Ángulo (°)")
    ax1.set_title(f"Serie temporal — secuencia #{idx}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0], projection="polar")
    t = np.linspace(0, 2 * np.pi, y_true.shape[1], endpoint=False)
    ax2.plot(t, y_true[idx], "k-o", label="Real")
    ax2.plot(t, y_pred_antes[idx], "s--", color="#E74C3C", label="Antes")
    ax2.plot(t, y_pred_despues[idx], "^--", color="#2ECC71", label="Después")
    ax2.set_title("Vista polar", pad=12)

    ax3 = fig.add_subplot(gs[1, 1])
    err_a = np.degrees(np.abs(error_angular(y_true[idx], y_pred_antes[idx])))
    err_d = np.degrees(np.abs(error_angular(y_true[idx], y_pred_despues[idx])))
    x = np.arange(len(err_a))
    w = 0.35
    ax3.bar(x - w / 2, err_a, width=w, color="#E74C3C", label="Antes", alpha=0.85)
    ax3.bar(x + w / 2, err_d, width=w, color="#2ECC71", label="Después", alpha=0.85)
    ax3.set_xlabel("Paso del horizonte")
    ax3.set_ylabel("Error angular (°)")
    ax3.set_title("Error por horizonte")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Resumen visual — predicción de vec_direction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Cargando datos reales (HURDAT2 + SHIPS + NAO)...")
    X_train, y_train, g_train, X_test, y_test, g_test, info = load_direction_train_test()
    print(f"  Train: {X_train.shape[0]} secuencias | Test: {X_test.shape[0]} secuencias")

    print("Entrenando modelos...")
    model_antes = entrenar_rf_direccion_directa(X_train, y_train, g_train)
    model_despues = entrenar_rf_direccion_sincos(X_train, y_train, g_train)

    y_pred_antes = model_antes.predict(X_test)
    y_pred_despues = sincos_to_direction(model_despues.predict(X_test))

    # Secuencias con distinto perfil de error para ilustrar
    err_antes = np.mean(np.abs(error_angular(y_test, y_pred_antes)), axis=1)
    err_despues = np.mean(np.abs(error_angular(y_test, y_pred_despues)), axis=1)
    idx_mejor_despues = int(np.argmax(err_antes - err_despues))
    idx_peor_despues = int(np.argmin(err_antes - err_despues))
    idx_medio = len(y_test) // 2

    indices_plot = sorted(set([idx_mejor_despues, idx_medio, idx_peor_despues]))

    archivos = {
        "01_secuencias_horizonte.png": lambda: plot_secuencias_horizonte(
            y_test, y_pred_antes, y_pred_despues, indices_plot,
            OUTPUT_DIR / "01_secuencias_horizonte.png",
        ),
        "02_scatter_real_vs_pred.png": lambda: plot_scatter_real_vs_pred(
            y_test, y_pred_antes, y_pred_despues,
            OUTPUT_DIR / "02_scatter_real_vs_pred.png",
        ),
        "03_polar_secuencia.png": lambda: plot_polar_secuencia(
            y_test, y_pred_antes, y_pred_despues, idx_mejor_despues,
            OUTPUT_DIR / "03_polar_secuencia.png",
        ),
        "04_error_angular_hist.png": lambda: plot_error_angular_hist(
            y_test, y_pred_antes, y_pred_despues,
            OUTPUT_DIR / "04_error_angular_hist.png",
        ),
        "05_panel_resumen.png": lambda: plot_panel_resumen(
            y_test, y_pred_antes, y_pred_despues, idx_mejor_despues,
            OUTPUT_DIR / "05_panel_resumen.png",
        ),
    }

    print("\nGenerando gráficos:")
    for nombre, fn in archivos.items():
        fn()
        print(f"  ✓ {OUTPUT_DIR / nombre}")

    print(f"\nListo. Gráficos en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
