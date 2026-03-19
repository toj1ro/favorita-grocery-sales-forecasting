import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS

import config


def run_baselines(train_sf: pd.DataFrame, horizon: int = config.HORIZON,
                  season_length: int = config.SEASON_LENGTH) -> pd.DataFrame:
    models = [
        Naive(),
        SeasonalNaive(season_length=season_length),
        AutoTheta(season_length=season_length),
        AutoETS(season_length=season_length),
    ]

    sf = StatsForecast(
        models=models,
        freq="D",
        n_jobs=-1,
    )

    print("Fitting baseline models...")
    forecasts = sf.forecast(df=train_sf, h=horizon)
    forecasts = forecasts.reset_index()

    for col in forecasts.columns:
        if col not in ("unique_id", "ds"):
            forecasts[col] = forecasts[col].clip(lower=0)

    print(f"Baselines done. Shape: {forecasts.shape}")
    return forecasts
