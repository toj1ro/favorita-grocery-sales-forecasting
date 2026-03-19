import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Tuple

import config
from src.features import build_features, get_feature_columns


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame) -> Tuple[lgb.Booster, list]:
    feature_cols = get_feature_columns()

    combined = pd.concat([train, val], ignore_index=True)
    combined = combined.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    combined_feat = build_features(combined)

    train_feat = combined_feat[combined_feat["date"].isin(train["date"].values)].copy()
    val_feat = combined_feat[combined_feat["date"].isin(val["date"].values)].copy()

    train_feat = train_feat.dropna(subset=feature_cols)
    val_feat = val_feat.dropna(subset=feature_cols)

    X_train = train_feat[feature_cols]
    y_train = train_feat["unit_sales"]
    X_val = val_feat[feature_cols]
    y_val = val_feat["unit_sales"]

    cat_cols = ["store_nbr", "item_nbr", "cluster"]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "gpu",
        "verbosity": -1,
        "seed": config.RANDOM_SEED,
    }

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    return model, feature_cols


def predict_lightgbm(model: lgb.Booster, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    df_feat = build_features(df)
    X = df_feat[feature_cols].fillna(0)
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def run_lightgbm(train: pd.DataFrame, val: pd.DataFrame,
                 test: pd.DataFrame) -> pd.DataFrame:
    model, feature_cols = train_lightgbm(train, val)

    test = test.copy()
    test["LGBMRegressor"] = predict_lightgbm(model, test, feature_cols)

    result = test[["store_nbr", "item_nbr", "date", "unit_sales", "LGBMRegressor"]].copy()
    result["unique_id"] = result["store_nbr"].astype(str) + "_" + result["item_nbr"].astype(str)
    result = result.rename(columns={"date": "ds", "unit_sales": "y"})

    print(f"LightGBM done. Predictions: {len(result)}")
    return result
