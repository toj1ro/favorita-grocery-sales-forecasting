import os
import json
import pandas as pd
import numpy as np

import config
from src.data_loader import prepare_data, to_statsforecast_format
from src.evaluation import compute_all_metrics, summary_table
from src.visualization import plot_metrics_comparison, plot_sample_forecasts
from src.models.baselines import run_baselines
from src.models.classical import run_lightgbm
from src.models.neural import run_neural_models


def get_perishable_weights(test: pd.DataFrame) -> np.ndarray:
    if "perishable" in test.columns:
        return np.where(test["perishable"] == 1, 1.25, 1.0)
    return np.ones(len(test))


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train, val, test = prepare_data()

    train_sf = to_statsforecast_format(train)
    trainval_sf = to_statsforecast_format(pd.concat([train, val], ignore_index=True))
    test_sf = to_statsforecast_format(test)

    all_results = {}

    print("\nBASELINES")
    baseline_preds = run_baselines(trainval_sf)
    baseline_merged = test_sf.merge(baseline_preds, on=["unique_id", "ds"], how="inner")

    weight_map = None
    if "perishable" in test.columns:
        weight_map = test.drop_duplicates(["store_nbr", "item_nbr"])[["store_nbr", "item_nbr", "perishable"]]
        weight_map["unique_id"] = weight_map["store_nbr"].astype(str) + "_" + weight_map["item_nbr"].astype(str)
        weight_map = weight_map[["unique_id", "perishable"]]

    def get_weights(merged_df):
        if weight_map is None:
            return None
        m = merged_df.merge(weight_map, on="unique_id", how="left")
        return get_perishable_weights(m)

    for model_name in ["Naive", "SeasonalNaive", "AutoTheta", "AutoETS"]:
        if model_name in baseline_merged.columns:
            y_true = baseline_merged["y"].values
            y_pred = baseline_merged[model_name].values
            all_results[model_name] = compute_all_metrics(y_true, y_pred, get_weights(baseline_merged))
            print(f"  {model_name}: {all_results[model_name]}")

    print("\nLIGHTGBM")
    lgbm_preds = run_lightgbm(train, val, test)
    lgbm_merged = test_sf.merge(
        lgbm_preds[["unique_id", "ds", "LGBMRegressor"]],
        on=["unique_id", "ds"], how="inner"
    )
    y_true = lgbm_merged["y"].values
    y_pred = lgbm_merged["LGBMRegressor"].values
    all_results["LightGBM"] = compute_all_metrics(y_true, y_pred, get_weights(lgbm_merged))
    print(f"  LightGBM: {all_results['LightGBM']}")

    print("\nNEURAL MODELS")
    neural_preds = run_neural_models(trainval_sf)
    neural_merged = test_sf.merge(neural_preds, on=["unique_id", "ds"], how="inner")

    for model_name in ["NHITS", "TFT"]:
        if model_name in neural_merged.columns:
            y_true = neural_merged["y"].values
            y_pred = neural_merged[model_name].values
            all_results[model_name] = compute_all_metrics(y_true, y_pred, get_weights(neural_merged))
            print(f"  {model_name}: {all_results[model_name]}")

    print("\nRESULTS")
    table = summary_table(all_results)
    print(table)

    table.to_csv(os.path.join(config.RESULTS_DIR, "metrics.csv"))
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nGenerating plots...")
    plot_metrics_comparison(table, save_path=os.path.join(config.RESULTS_DIR, "metrics_comparison.png"))

    all_preds = test_sf.copy()
    for df_pred, cols in [(baseline_merged, ["Naive", "SeasonalNaive", "AutoTheta", "AutoETS"]),
                          (lgbm_merged, ["LGBMRegressor"]),
                          (neural_merged, ["NHITS", "TFT"])]:
        for col in cols:
            if col in df_pred.columns:
                all_preds = all_preds.merge(df_pred[["unique_id", "ds", col]], on=["unique_id", "ds"], how="left")

    pred_cols = [c for c in all_preds.columns if c not in ("unique_id", "ds", "y")]
    plot_sample_forecasts(all_preds, pred_cols, n_samples=6,
                          save_dir=os.path.join(config.RESULTS_DIR, "forecasts"))

    print(f"\nResults saved to {config.RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
