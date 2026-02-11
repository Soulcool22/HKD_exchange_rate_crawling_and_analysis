from __future__ import annotations

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from .features import make_supervised_multistep


def build_model(n_features: int, lookback: int, horizon: int):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, activation="tanh", input_shape=(lookback, n_features)),
            LSTM(32, activation="tanh"),
            Dense(horizon),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_recent_window(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    lookback: int,
    horizon: int,
    value_min: float | None = None,
    value_max: float | None = None,
):
    eval_days = min(horizon, len(target_scaled) - lookback - 5)
    if eval_days <= 1:
        raise ValueError("可用样本不足，无法评估。")

    hist_x = features_scaled[:-eval_days]
    hist_y = target_scaled[:-eval_days]
    x_train, y_train = make_supervised_multistep(hist_x, hist_y, lookback=lookback, horizon=horizon)

    n_features = hist_x.shape[1]
    model = build_model(n_features=n_features, lookback=lookback, horizon=horizon)
    es = EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=35, batch_size=64, verbose=0, callbacks=[es])

    x_input = hist_x[-lookback:, :].reshape((1, lookback, n_features))
    y_hat = model.predict(x_input, verbose=0)[0]
    preds = y_hat[:eval_days]
    actuals = target_scaled[-eval_days:]

    if value_min is not None and value_max is not None:
        preds_eval = preds * (value_max - value_min) + value_min
        actuals_eval = actuals * (value_max - value_min) + value_min
    else:
        preds_eval = preds
        actuals_eval = actuals

    rmse = float(np.sqrt(np.mean((preds_eval - actuals_eval) ** 2)))
    mae = float(np.mean(np.abs(preds_eval - actuals_eval)))
    mape = float(np.mean(np.abs((preds_eval - actuals_eval) / (actuals_eval + 1e-8))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape, "window_days": int(eval_days)}


def forecast_future(features_scaled: np.ndarray, target_scaled: np.ndarray, lookback: int, steps: int):
    x_train, y_train = make_supervised_multistep(features_scaled, target_scaled, lookback=lookback, horizon=steps)
    n_features = features_scaled.shape[1]

    model = build_model(n_features=n_features, lookback=lookback, horizon=steps)
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=60, batch_size=64, verbose=0, callbacks=[es])

    x_input = features_scaled[-lookback:, :].reshape((1, lookback, n_features))
    y_hat = model.predict(x_input, verbose=0)[0]
    return y_hat.astype(np.float32)
