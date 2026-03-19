import pandas as pd
import numpy as np

import config


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek.astype("int8")
    df["dayofmonth"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["is_payday"] = ((df["dayofmonth"] == 15) | (df["dayofmonth"] == df["date"].dt.days_in_month)).astype("int8")
    return df


def add_lag_features(df: pd.DataFrame, lags: list = config.LAG_DAYS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_nbr", "item_nbr"])["unit_sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = config.ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        grp = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]
        df[f"rolling_mean_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std()).fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def get_feature_columns() -> list:
    cols = [
        "onpromotion", "dcoilwtico", "is_holiday",
        "dayofweek", "dayofmonth", "month", "year", "is_weekend", "is_payday",
    ]
    for lag in config.LAG_DAYS:
        cols.append(f"lag_{lag}")
    for w in config.ROLLING_WINDOWS:
        cols.extend([f"rolling_mean_{w}", f"rolling_std_{w}"])
    cols.extend(["store_nbr", "item_nbr", "cluster"])
    return cols
