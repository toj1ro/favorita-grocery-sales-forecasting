import pandas as pd
import numpy as np
from typing import Tuple, Dict

import config


def load_train(nrows: int | None = None) -> pd.DataFrame:
    dtypes = {
        "id": "int32",
        "store_nbr": "int8",
        "item_nbr": "int32",
        "unit_sales": "float32",
        "onpromotion": "object",
    }
    df = pd.read_csv(
        config.TRAIN_FILE,
        dtype=dtypes,
        parse_dates=["date"],
        nrows=nrows,
    )
    df["onpromotion"] = df["onpromotion"].map({"True": 1, "False": 0}).fillna(0).astype("int8")
    df["unit_sales"] = df["unit_sales"].clip(lower=0)
    return df


def load_metadata() -> Dict[str, pd.DataFrame]:
    stores = pd.read_csv(config.STORES_FILE)
    items = pd.read_csv(config.ITEMS_FILE)

    oil = pd.read_csv(config.OIL_FILE, parse_dates=["date"])
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").ffill().bfill()

    holidays = pd.read_csv(config.HOLIDAYS_FILE, parse_dates=["date"])
    holidays = holidays[holidays["transferred"] == False].copy()
    holidays["is_holiday"] = 1

    transactions = pd.read_csv(config.TRANSACTIONS_FILE, parse_dates=["date"])

    return {
        "stores": stores,
        "items": items,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }


def sample_series(df: pd.DataFrame, meta: Dict[str, pd.DataFrame],
                  n: int = config.N_SERIES, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    items = meta["items"]
    stores = meta["stores"]

    combos = df.groupby(["store_nbr", "item_nbr"]).agg(
        n_days=("date", "nunique"),
        total_sales=("unit_sales", "sum"),
    ).reset_index()

    combos = combos[combos["n_days"] >= 365]

    combos = combos.merge(stores[["store_nbr", "type"]], on="store_nbr")
    combos = combos.merge(items[["item_nbr", "family"]], on="item_nbr")

    n_groups = combos.groupby(["type", "family"]).ngroups
    sampled = combos.groupby(["type", "family"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, n // n_groups)), random_state=rng)
        if len(g) > 0 else g
    )

    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=rng)
    elif len(sampled) < n:
        remaining = combos[~combos.index.isin(sampled.index)]
        extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=rng)
        sampled = pd.concat([sampled, extra])

    keys = sampled[["store_nbr", "item_nbr"]].drop_duplicates()
    df_sampled = df.merge(keys, on=["store_nbr", "item_nbr"])

    return df_sampled


def enrich_data(df: pd.DataFrame, meta: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.merge(meta["stores"], on="store_nbr", how="left")
    df = df.merge(meta["items"], on="item_nbr", how="left")
    df = df.merge(meta["oil"][["date", "dcoilwtico"]], on="date", how="left")

    holidays_national = meta["holidays"][meta["holidays"]["locale"] == "National"][["date", "is_holiday"]]
    holidays_national = holidays_national.drop_duplicates(subset="date")
    df = df.merge(holidays_national, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    return df


def prepare_data(nrows: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading train data...")
    train_raw = load_train(nrows=nrows)

    print("Loading metadata...")
    meta = load_metadata()

    print(f"Sampling {config.N_SERIES} time series...")
    df = sample_series(train_raw, meta)

    print("Enriching with metadata...")
    df = enrich_data(df, meta)
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

    train = df[df["date"] <= config.TRAIN_END].copy()
    val = df[(df["date"] > config.TRAIN_END) & (df["date"] <= config.VAL_END)].copy()
    test = df[df["date"] > config.VAL_END].copy()

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def to_statsforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    sf_df = df[["store_nbr", "item_nbr", "date", "unit_sales"]].copy()
    sf_df["unique_id"] = sf_df["store_nbr"].astype(str) + "_" + sf_df["item_nbr"].astype(str)
    sf_df = sf_df.rename(columns={"date": "ds", "unit_sales": "y"})
    sf_df = sf_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return sf_df
