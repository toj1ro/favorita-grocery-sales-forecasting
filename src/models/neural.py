import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT
from neuralforecast.losses.pytorch import MAE

import config


def run_neural_models(train_sf: pd.DataFrame, horizon: int = config.HORIZON,
                      season_length: int = config.SEASON_LENGTH,
                      exog_df: pd.DataFrame | None = None) -> pd.DataFrame:
    models = [
        NHITS(
            h=horizon,
            input_size=horizon * 3,
            max_steps=config.MAX_EPOCHS * 100,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            scaler_type="standard",
            random_seed=config.RANDOM_SEED,
            loss=MAE(),
            accelerator="gpu",
        ),
        TFT(
            h=horizon,
            input_size=horizon * 3,
            max_steps=config.MAX_EPOCHS * 100,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            scaler_type="standard",
            random_seed=config.RANDOM_SEED,
            loss=MAE(),
            accelerator="gpu",
        ),
    ]

    nf = NeuralForecast(models=models, freq="D")

    print("Training neural models (N-HiTS, TFT)...")
    nf.fit(df=train_sf)

    print("Generating neural forecasts...")
    forecasts = nf.predict(futr_df=exog_df)
    forecasts = forecasts.reset_index()

    for col in forecasts.columns:
        if col not in ("unique_id", "ds"):
            forecasts[col] = forecasts[col].clip(lower=0)

    print(f"Neural models done. Shape: {forecasts.shape}")
    return forecasts
