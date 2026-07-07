"""
Carga y preparación de datos HURDAT2 + SHIPS + NAO para modelos de dirección.
Rutas relativas al directorio del repositorio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent

HURDAT_FILE = _SCRIPT_DIR / "hurdat2-1851-2023-051124.txt"
SHIPS_FILE = _SCRIPT_DIR / "ships_filter.csv"
NAO_FILE = _SCRIPT_DIR / "norm_nao_monthly_b5001_current_ascii.txt"

TIMESTEPS = 12
HORIZON = 12
FEATURES = [
    "MinPress", "Max_wind", "vec_len", "vec_direction", "Lat_N", "Lon_W",
    "shear_kt", "sst_c", "rh_pct", "heat_content",
    "storm_bearing_deg", "storm_motion_kt", "nao",
]

MEDIANAS_MINPRESS = {
    "Cat 1": 983.0, "Cat 2": 968.0, "Cat 3": 955.0, "Cat 4": 941.0, "Cat 5": 918.1,
    "DB": 1010.0, "EX": 995.0, "LO": 1009.0, "SD": 1008.0, "SS": 998.0,
    "TD": 1008.0, "TS": 1000.0, "WV": 1009.0,
}


def _clasificacion_huracan(df: pd.DataFrame) -> pd.DataFrame:
    condiciones = [
        (df["Max_wind"] >= 64) & (df["Max_wind"] <= 82),
        (df["Max_wind"] >= 83) & (df["Max_wind"] <= 95),
        (df["Max_wind"] >= 96) & (df["Max_wind"] <= 113),
        (df["Max_wind"] >= 114) & (df["Max_wind"] <= 135),
        (df["Max_wind"] > 135),
    ]
    categorias = ["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
    df = df.copy()
    df["categoria_huracan"] = np.select(condiciones, categorias, default="Sin categoria")
    return df


def _calculate_direction(vec_x, vec_y):
    return np.arctan2(vec_x, vec_y)


def _compute_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vec_x"] = np.nan
    df["vec_y"] = np.nan
    vecs = {"vec_x": [], "vec_y": [], "id": []}

    for storm_id in df["Code_storm"].unique():
        storm_data = df[df["Code_storm"] == storm_id].copy()
        storm_data = storm_data.sort_values(["year", "month", "day", "hour"]).reset_index()
        last_x = last_y = None
        for _, row in storm_data.iterrows():
            current_x, current_y = row["Lon_W"], row["Lat_N"]
            if last_x is not None:
                vecs["vec_x"].append(current_x - last_x)
                vecs["vec_y"].append(current_y - last_y)
                vecs["id"].append(row["index"])
            last_x, last_y = current_x, current_y

    df.loc[vecs["id"], "vec_x"] = vecs["vec_x"]
    df.loc[vecs["id"], "vec_y"] = vecs["vec_y"]
    df["vec_len"] = np.sqrt(df["vec_x"] ** 2 + df["vec_y"] ** 2)
    df["vec_direction"] = df.apply(
        lambda r: _calculate_direction(r.vec_x, r.vec_y), axis=1
    )
    return df


def load_hurdat_etl() -> pd.DataFrame:
    """ETL HURDAT2 hasta features vec_direction / vec_len (sin SHIPS/NAO)."""
    df = pd.read_csv(
        HURDAT_FILE,
        sep=",",
        names=[
            "Date", "Hour", "RecIdentifier", "type_storm", "Latitude", "Longitude",
            "Max_wind", "MinPress", "NE34", "SE34", "SW34", "NW34",
            "NE50", "SE50", "SW50", "NW50", "NE64", "SE64", "SW64", "NW64",
            "Rad_Max_Wind",
        ],
    )

    df["Code_storm"] = pd.Series(dtype=object)
    df["Name_storm"] = pd.Series(dtype=object)
    df["Trackets"] = pd.Series(dtype=object)
    current_code = current_name = current_rec = None
    for index, row in df.iterrows():
        if str(row["Date"]).startswith("AL"):
            current_code, current_name, current_rec = row["Date"], row["Hour"], row["RecIdentifier"]
        df.at[index, "Code_storm"] = current_code
        df.at[index, "Name_storm"] = current_name
        df.at[index, "Trackets"] = current_rec

    df = df[~df["Date"].astype(str).str.startswith("AL")].reset_index(drop=True)
    column_order = ["Code_storm", "Name_storm", "Trackets"] + [
        c for c in df.columns if c not in ["Code_storm", "Name_storm", "Trackets"]
    ]
    df = df[column_order]

    df["Lat_N"] = pd.to_numeric(df["Latitude"].str.extract(r"(\d+\.\d+)", expand=False))
    df["Lon_W"] = pd.to_numeric(df["Longitude"].str.extract(r"(\d+\.\d+)", expand=False))
    df.drop(["Latitude", "Longitude"], axis=1, inplace=True)

    df["type_storm"] = df["type_storm"].astype(str).str.strip().str.upper()
    df["Code_storm"] = df["Code_storm"].astype(str).str.strip().str.upper()

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df.drop("Date", axis=1, inplace=True)

    df["Hour"] = df["Hour"].astype(int)
    df["hour"] = df["Hour"] // 100
    df.drop("Hour", axis=1, inplace=True)

    df.replace(-999, np.nan, inplace=True)
    df.replace(-99, np.nan, inplace=True)
    df.drop(["Rad_Max_Wind", "RecIdentifier", "Trackets"], axis=1, inplace=True)

    df = df.drop(
        columns=[
            "NW64", "NE64", "SE64", "SW64", "NW50", "NE50", "SE50", "SW50",
            "NW34", "NE34", "SE34", "SW34",
        ]
    )

    df = _clasificacion_huracan(df)
    df.loc[df["type_storm"] == "HU", "type_storm"] = df.loc[
        df["type_storm"] == "HU", "categoria_huracan"
    ]
    df.drop("categoria_huracan", axis=1, inplace=True)

    if 25198 in df.index and 25199 in df.index:
        df = df.drop([25198, 25199])

    mapeo_target = {
        "TD": 1, "TS": 2, "Cat 1": 3, "Cat 2": 4, "Cat 3": 5,
        "Cat 4": 6, "Cat 5": 7, "DB": 13, "WV": 12, "LO": 11,
        "SD": 10, "SS": 9, "EX": 8,
    }
    df["type_storm_n"] = df["type_storm"].map(mapeo_target)
    df["MinPress"] = df["MinPress"].fillna(df["type_storm"].map(MEDIANAS_MINPRESS))

    df["Lon_W"] = -df["Lon_W"]
    df = _compute_vectors(df)
    df = df[df["vec_len"] < 354]

    for k in range(1, 4):
        df[f"prev_len_{k}"] = df.groupby("Code_storm")["vec_len"].shift(k)
        df[f"prev_direction_{k}"] = df.groupby("Code_storm")["vec_direction"].shift(k)

    df["next_len"] = df.groupby("Code_storm")["vec_len"].shift(-1)
    df["next_direction"] = df.groupby("Code_storm")["vec_direction"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def merge_ships(df: pd.DataFrame) -> pd.DataFrame:
    df_ships = pd.read_csv(SHIPS_FILE, sep=";")
    ships_filter = df_ships[
        [
            "storm_code", "shear_kt", "sst_c", "rh_pct", "heat_content",
            "storm_bearing_deg", "storm_motion_kt", "lat_hurdat2", "lon_hurdat2",
            "lat_s", "lon_s",
        ]
    ].rename(columns={
        "storm_code": "Code_storm",
        "lat_hurdat2": "Lat_N",
        "lon_hurdat2": "Lon_W",
    })

    columnas_llave = ["Code_storm", "Lat_N", "Lon_W"]
    columnas_ships = [
        "shear_kt", "sst_c", "rh_pct", "heat_content",
        "storm_bearing_deg", "storm_motion_kt", "lat_s", "lon_s",
    ]

    df_out = df.merge(
        ships_filter[columnas_llave + columnas_ships],
        on=columnas_llave,
        how="left",
    )
    return df_out[df_out["sst_c"].notna()].copy()


def merge_nao(df: pd.DataFrame) -> pd.DataFrame:
    df_nao = pd.read_csv(
        NAO_FILE,
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=[
            "year", "m1", "m2", "m3", "m4", "m5", "m6",
            "m7", "m8", "m9", "m10", "m11", "m12",
        ],
    )
    df_nao_long = df_nao.melt(
        id_vars=["year"],
        value_vars=[f"m{k}" for k in range(1, 13)],
        var_name="_mk",
        value_name="nao",
    )
    df_nao_long["month"] = df_nao_long["_mk"].str.replace("m", "", regex=False).astype(int)
    df_nao_long = df_nao_long.drop(columns=["_mk"])
    df_nao_long["nao"] = pd.to_numeric(df_nao_long["nao"], errors="coerce")
    df_nao_long.loc[df_nao_long["nao"] <= -900, "nao"] = pd.NA

    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df.merge(df_nao_long[["year", "month", "nao"]], on=["year", "month"], how="left")


def load_eda_dataframe() -> pd.DataFrame:
    """DataFrame final listo para crear secuencias (HURDAT2 + SHIPS + NAO)."""
    df = load_hurdat_etl()
    df = merge_ships(df)
    df = merge_nao(df)
    return df


def create_sequences_separated(
    df: pd.DataFrame,
    timesteps: int = TIMESTEPS,
    horizon: int = HORIZON,
    features: list[str] | None = None,
):
    if features is None:
        features = FEATURES

    features_dir = [f for f in features if f not in ("vec_direction", "Max_wind")]
    X_sequences_dir, y_sequences_dir, grupos_sequences = [], [], []

    df_sorted = df.sort_values(["Code_storm"]).reset_index(drop=True)
    for storm_id in df_sorted["Code_storm"].unique():
        storm_data = df_sorted[df_sorted["Code_storm"] == storm_id].copy().reset_index(drop=True)
        if len(storm_data) < timesteps + horizon:
            continue
        for i in range(len(storm_data) - timesteps - horizon + 1):
            X_sequences_dir.append(storm_data[features_dir].iloc[i : i + timesteps].values)
            y_sequences_dir.append(
                storm_data["vec_direction"].iloc[i + timesteps : i + timesteps + horizon].values
            )
            grupos_sequences.append(storm_id)

    X_dir = np.array(X_sequences_dir)
    y_dir = np.array(y_sequences_dir)
    grupos = np.array(grupos_sequences)
    return X_dir, y_dir, grupos, features_dir


def split_train_test_by_storm(df: pd.DataFrame, test_frac: float = 0.10):
    ids = df["Code_storm"].dropna().unique()
    n_test = max(1, min(int(round(len(ids) * test_frac)), len(ids)))
    test_ids = set(ids[-n_test:])
    train = df[df["Code_storm"].isin(set(ids) - test_ids)]
    test = df[df["Code_storm"].isin(test_ids)]
    return train, test


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def load_direction_train_test(test_frac: float = 0.10):
    """
    Carga datos reales y devuelve arrays aplanados para modelos de dirección.

    Returns
    -------
    X_train, y_train, grupos_train, X_test, y_test, grupos_test, info
    """
    df = load_eda_dataframe()
    train, test = split_train_test_by_storm(df, test_frac=test_frac)

    X_train, y_train, grupos_train, features_dir = create_sequences_separated(train)
    X_test, y_test, grupos_test, _ = create_sequences_separated(test)

    info = {
        "n_storms_train": train["Code_storm"].nunique(),
        "n_storms_test": test["Code_storm"].nunique(),
        "n_features_dir": len(features_dir),
        "features_dir": features_dir,
        "timesteps": TIMESTEPS,
        "horizon": HORIZON,
        "n_rows_eda": len(df),
    }
    return (
        flatten_sequences(X_train),
        y_train,
        grupos_train,
        flatten_sequences(X_test),
        y_test,
        grupos_test,
        info,
    )
