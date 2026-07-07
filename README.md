# tesistest

Pipeline de predicción de trayectoria e intensidad de huracanes (HURDAT2 + SHIPS + NAO).

## Datos (en la raíz del repo)

| Archivo | Descripción |
|---------|-------------|
| `hurdat2-1851-2023-051124.txt` | Histórico HURDAT2 |
| `ships_filter.csv` | Variables ambientales SHIPS |
| `norm_nao_monthly_b5001_current_ascii.txt` | Índice NAO mensual |

## Dirección circular (Random Forest)

```bash
python3 rf_direction_circular_test.py      # comparación antes vs sin/cos
python3 rf_direction_visualization.py      # gráficos comparativos
```

Módulos: `circular_direction.py`, `dataset_loader.py`. Integración en `tesis_fisico_secuential.py` sección **5.3.1**.