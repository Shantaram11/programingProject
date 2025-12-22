from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LogCb = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class CleaningConfig:
    # Missing values: none/drop_rows/ffill/bfill/interpolate_linear
    missing_method: str = "ffill"
    # Outliers: none/clip_quantile
    outlier_method: str = "none"
    clip_q_low: float = 0.01
    clip_q_high: float = 0.99
    # Scaling: none/standard/minmax
    scaler: str = "standard"
    # Target transform: none/log1p
    target_transform: str = "none"


@dataclass(frozen=True)
class FeatureConfig:
    time_col: str
    targets: List[str]
    features: List[str]
    freq: str = "auto"
    sort_time: bool = True
    train_ratio: float = 0.8
    horizon: int = 24
    lag_window: int = 48


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class PipelineConfig:
    cleaning: CleaningConfig
    features: FeatureConfig
    models: Dict[str, ModelConfig]


@dataclass
class TrainingResult:
    run_id: str
    config: Dict[str, Any]
    targets: List[str]
    models: List[str]
    # per_target[target] = {"t": np.ndarray, "y_true": np.ndarray, "preds": {model: np.ndarray}}
    per_target: Dict[str, Dict[str, Any]]
    # metrics_by_model[model] = aggregated across targets
    metrics_by_model: Dict[str, Dict[str, float]]


def load_dataset(path: str) -> pd.DataFrame:
    p = str(path)
    lower = p.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(p)
    if lower.endswith(".parquet"):
        return pd.read_parquet(p)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {path} (supported: csv/xlsx/xls/parquet)")


class AvailableModels:
    def __init__(self, ok: Dict[str, Tuple[bool, str]]) -> None:
        self._ok = ok

    @staticmethod
    def detect() -> "AvailableModels":
        ok: Dict[str, Tuple[bool, str]] = {}
        ok["ma"] = (True, "")
        ok["wma"] = (True, "")

        ok["arima"] = _try_import("statsmodels.tsa.arima.model", "ARIMA")
        ok["prophet"] = _try_import("prophet", "Prophet")
        ok["xgboost"] = _try_import("xgboost", "XGBRegressor")
        ok["deepar"] = _try_import("torch", "nn")
        return AvailableModels(ok)

    def status_for(self, model_key: str) -> Tuple[bool, str]:
        return self._ok.get(model_key, (False, "unknown model"))


def _try_import(module: str, attr: str) -> Tuple[bool, str]:
    try:
        m = __import__(module, fromlist=[attr])
        getattr(m, attr)
        return True, ""
    except Exception as e:
        return False, f"{module}.{attr}"


def run_training(df: pd.DataFrame, cfg: PipelineConfig, log_cb: LogCb = None) -> TrainingResult:
    def log(msg: str) -> None:
        if log_cb is not None:
            log_cb(msg)

    feat = cfg.features
    clean = cfg.cleaning

    if feat.time_col not in df.columns:
        raise ValueError(f"time column '{feat.time_col}' not found in dataframe")
    for t in feat.targets:
        if t not in df.columns:
            raise ValueError(f"target column '{t}' not found in dataframe")

    for c in feat.features:
        if c not in df.columns:
            raise ValueError(f"feature column '{c}' not found in dataframe")

    work = df.copy()
    work[feat.time_col] = pd.to_datetime(work[feat.time_col], errors="coerce")
    work = work.dropna(subset=[feat.time_col])

    if feat.sort_time:
        work = work.sort_values(feat.time_col, ascending=True)

    # keep only needed columns
    keep_cols = [feat.time_col] + list(dict.fromkeys(feat.targets + feat.features))
    work = work[keep_cols]

    # missing values
    work = _handle_missing(work, clean.missing_method)

    # outliers (numeric only)
    if clean.outlier_method == "clip_quantile":
        work = _clip_outliers(work, q_low=clean.clip_q_low, q_high=clean.clip_q_high)

    # Ensure no NaNs remain in used columns (prevents NA/NaN predictions, esp. DeepAR)
    work = work.dropna(axis=0)

    # Ensure we still have enough data
    n = len(work)
    if n < max(20, feat.lag_window + feat.horizon + 5):
        raise ValueError(
            f"Not enough rows after cleaning: {n}. "
            f"Need at least ~{feat.lag_window + feat.horizon + 5} rows for chosen lag/horizon."
        )

    # Determine test window length based on train_ratio, but ensure >= horizon
    split_idx = int(n * feat.train_ratio)
    split_idx = min(split_idx, n - feat.horizon)
    split_idx = max(split_idx, feat.lag_window + 5)
    test_len = n - split_idx
    if test_len < feat.horizon:
        split_idx = n - feat.horizon
        test_len = feat.horizon

    log(f"Rows after cleaning: {n}. Train rows: {split_idx}. Test rows: {test_len}.")

    per_target: Dict[str, Dict[str, Any]] = {}
    models_enabled = list(cfg.models.keys())
    metrics_by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in models_enabled}

    for target in feat.targets:
        log(f"--- Target: {target} ---")
        t_all = work[feat.time_col].to_numpy()

        y_raw = work[target].astype(float).to_numpy()
        X_raw = work[feat.features].astype(float).to_numpy() if feat.features else None
        feat_names = feat.features

        # Transform target (e.g. log1p)
        y_t, inv_transform = _target_transform(y_raw, clean.target_transform)

        # Scale for ML/Deep models (stats models typically don't need scaling)
        scaler_bundle = _make_scalers(clean.scaler, y_t, X_raw)
        y_scaled = scaler_bundle["y_scaled"]  # transformed + scaled
        X_scaled = scaler_bundle["X_scaled"]
        inv_scale_y = scaler_bundle["inv_scale_y"]

        y_train_t = y_t[:split_idx]  # transformed only
        y_train_scaled = y_scaled[:split_idx]  # transformed + scaled
        y_test_true = y_raw[split_idx:]  # original scale for metrics/plot
        t_test = t_all[split_idx:]

        X_train = X_scaled[:split_idx] if X_scaled is not None else None
        X_test = X_scaled[split_idx:] if X_scaled is not None else None
        X_train_raw = X_raw[:split_idx] if X_raw is not None else None
        X_test_raw = X_raw[split_idx:] if X_raw is not None else None

        preds_by_model: Dict[str, np.ndarray] = {}

        for model_name, mc in cfg.models.items():
            log(f"Training model: {model_name}")
            use_scaling = model_name in {"xgboost", "deepar"}
            y_train_model = y_train_scaled if use_scaling else y_train_t
            y_pred_model = _fit_predict_model(
                model_name=model_name,
                mc=mc,
                y_train=y_train_model,
                horizon=test_len,
                lag_window=feat.lag_window,
                X_train=X_train,
                X_future=X_test,
                X_train_raw=X_train_raw,
                X_future_raw=X_test_raw,
                feat_names=feat_names,
                t_train=work[feat.time_col].to_numpy()[:split_idx],
                t_future=t_test,
                log=log,
            )

            # Inverse scale (only for ML/Deep), then inverse transform to get back to original scale
            y_pred_t = inv_scale_y(y_pred_model) if use_scaling else np.asarray(y_pred_model, dtype=float)
            y_pred = inv_transform(y_pred_t)
            y_pred = np.asarray(y_pred, dtype=float)

            # Align lengths
            y_pred = y_pred[:test_len]
            y_true = y_test_true[:test_len]
            t_plot = t_test[:test_len]

            preds_by_model[model_name] = y_pred
            m = _metrics(y_true, y_pred)
            metrics_by_model[model_name].append(m)
            log(f"  {model_name} metrics: MAE={m['mae']:.5g} RMSE={m['rmse']:.5g} MAPE%={m['mape_pct']:.5g}")

        per_target[target] = {"t": t_plot, "y_true": y_test_true[:test_len], "preds": preds_by_model}

    # Aggregate metrics across targets per model
    agg: Dict[str, Dict[str, float]] = {}
    for model_name, ms in metrics_by_model.items():
        agg[model_name] = {
            "mae": float(np.mean([m["mae"] for m in ms])) if ms else float("nan"),
            "rmse": float(np.mean([m["rmse"] for m in ms])) if ms else float("nan"),
            "mape_pct": float(np.mean([m["mape_pct"] for m in ms])) if ms else float("nan"),
        }

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return TrainingResult(
        run_id=run_id,
        config=asdict(cfg),
        targets=list(feat.targets),
        models=models_enabled,
        per_target=per_target,
        metrics_by_model=agg,
    )


def _handle_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "drop_rows":
        return df.dropna(axis=0)
    if method == "ffill":
        # ffill alone can leave leading NaNs; always fill both directions
        return df.ffill().bfill()
    if method == "bfill":
        return df.bfill().ffill()
    if method == "interpolate_linear":
        return df.interpolate(method="linear").ffill().bfill()
    raise ValueError(f"Unknown missing_method: {method}")


def _clip_outliers(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        lo = out[c].quantile(q_low)
        hi = out[c].quantile(q_high)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def _target_transform(y: np.ndarray, method: str) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    y = np.asarray(y, dtype=float)
    if method == "none":
        return y, lambda z: np.asarray(z, dtype=float)
    if method == "log1p":
        y2 = np.log1p(np.maximum(y, -0.999999))  # keep valid

        def inv(z: np.ndarray) -> np.ndarray:
            return np.expm1(z)

        return y2, inv
    raise ValueError(f"Unknown target_transform: {method}")


def _make_scalers(
    method: str, y: np.ndarray, X: Optional[np.ndarray]
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if method == "none":
        return {
            "y_scaled": y.reshape(-1),
            "X_scaled": X,
            "inv_scale_y": lambda z: np.asarray(z, dtype=float),
        }

    if method not in {"standard", "minmax"}:
        raise ValueError(f"Unknown scaler: {method}")

    try:
        if method == "standard":
            from sklearn.preprocessing import StandardScaler

            sy = StandardScaler()
            sx = StandardScaler() if X is not None else None
        else:
            from sklearn.preprocessing import MinMaxScaler

            sy = MinMaxScaler()
            sx = MinMaxScaler() if X is not None else None
    except Exception as e:
        raise RuntimeError("scikit-learn is required for scaling. Install it from requirements.txt") from e

    y_scaled = sy.fit_transform(y).reshape(-1)
    X_scaled = sx.fit_transform(X) if (X is not None and sx is not None) else X

    def inv_scale_y(z: np.ndarray) -> np.ndarray:
        z2 = np.asarray(z, dtype=float).reshape(-1, 1)
        return sy.inverse_transform(z2).reshape(-1)

    return {"y_scaled": y_scaled, "X_scaled": X_scaled, "inv_scale_y": inv_scale_y}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}


def _fit_predict_model(
    model_name: str,
    mc: ModelConfig,
    y_train: np.ndarray,
    horizon: int,
    lag_window: int,
    X_train: Optional[np.ndarray],
    X_future: Optional[np.ndarray],
    X_train_raw: Optional[np.ndarray],
    X_future_raw: Optional[np.ndarray],
    feat_names: List[str],
    t_train: np.ndarray,
    t_future: np.ndarray,
    log: Callable[[str], None],
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=float)

    if model_name == "ma":
        window = int(mc.params.get("window", 24))
        return _predict_ma(y_train, horizon=horizon, window=window)
    if model_name == "wma":
        window = int(mc.params.get("window", 24))
        weights = str(mc.params.get("weights", "linear_recent_heavier"))
        return _predict_wma(y_train, horizon=horizon, window=window, weights=weights)
    if model_name == "arima":
        ok, msg = AvailableModels.detect().status_for("arima")
        if not ok:
            raise RuntimeError(f"ARIMA requires missing dependency: {msg}")
        return _predict_arima(
            y_train_t=y_train,
            horizon=horizon,
            p=int(mc.params.get("p", 2)),
            d=int(mc.params.get("d", 1)),
            q=int(mc.params.get("q", 2)),
            X_train_raw=X_train_raw if bool(mc.params.get("use_exog", False)) else None,
            X_future_raw=X_future_raw if bool(mc.params.get("use_exog", False)) else None,
        )
    if model_name == "prophet":
        ok, msg = AvailableModels.detect().status_for("prophet")
        if not ok:
            raise RuntimeError(f"Prophet requires missing dependency: {msg}")
        return _predict_prophet(
            t_train=t_train,
            y_train_t=y_train,
            t_future=t_future,
            horizon=horizon,
            X_train_raw=X_train_raw if bool(mc.params.get("use_regressors", False)) else None,
            X_future_raw=X_future_raw if bool(mc.params.get("use_regressors", False)) else None,
            feat_names=feat_names,
            params=mc.params,
        )
    if model_name == "xgboost":
        ok, msg = AvailableModels.detect().status_for("xgboost")
        if not ok:
            raise RuntimeError(f"XGBoost requires missing dependency: {msg}")
        return _predict_xgboost(
            y_train=y_train,
            X_train=X_train,
            X_future=X_future,
            horizon=horizon,
            lag_window=lag_window,
            params=mc.params,
        )
    if model_name == "deepar":
        ok, msg = AvailableModels.detect().status_for("deepar")
        if not ok:
            raise RuntimeError(f"DeepAR requires missing dependency: {msg}")
        return _predict_deepar_torch(
            y_train=y_train,
            X_train=X_train,
            X_future=X_future,
            horizon=horizon,
            lag_window=lag_window,
            params=mc.params,
            log=log,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _predict_ma(y_train: np.ndarray, horizon: int, window: int) -> np.ndarray:
    window = max(1, int(window))
    history = list(np.asarray(y_train, dtype=float))
    preds: List[float] = []
    for _ in range(int(horizon)):
        w = history[-window:] if len(history) >= window else history
        preds.append(float(np.mean(w)))
        history.append(preds[-1])
    return np.asarray(preds, dtype=float)


def _predict_wma(y_train: np.ndarray, horizon: int, window: int, weights: str) -> np.ndarray:
    window = max(1, int(window))
    history = list(np.asarray(y_train, dtype=float))
    preds: List[float] = []
    for _ in range(int(horizon)):
        w = history[-window:] if len(history) >= window else history
        n = len(w)
        if n == 1:
            preds.append(float(w[0]))
            history.append(preds[-1])
            continue

        if weights == "linear_recent_heavier":
            ws = np.arange(1, n + 1, dtype=float)
        elif weights == "linear_older_heavier":
            ws = np.arange(n, 0, -1, dtype=float)
        else:
            ws = np.ones(n, dtype=float)
        ws = ws / ws.sum()
        preds.append(float(np.dot(ws, np.asarray(w, dtype=float))))
        history.append(preds[-1])
    return np.asarray(preds, dtype=float)


def _predict_arima(
    y_train_t: np.ndarray,
    horizon: int,
    p: int,
    d: int,
    q: int,
    X_train_raw: Optional[np.ndarray],
    X_future_raw: Optional[np.ndarray],
) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(y_train_t, dtype=float)
    exog = np.asarray(X_train_raw, dtype=float) if X_train_raw is not None else None
    model = ARIMA(y, order=(p, d, q), exog=exog)
    fit = model.fit()
    exog_future = np.asarray(X_future_raw, dtype=float) if X_future_raw is not None else None
    fc = fit.forecast(steps=int(horizon), exog=exog_future)
    return np.asarray(fc, dtype=float)


def _predict_prophet(
    t_train: np.ndarray,
    y_train_t: np.ndarray,
    t_future: np.ndarray,
    horizon: int,
    X_train_raw: Optional[np.ndarray],
    X_future_raw: Optional[np.ndarray],
    feat_names: List[str],
    params: Dict[str, Any],
) -> np.ndarray:
    from prophet import Prophet

    train = pd.DataFrame({"ds": pd.to_datetime(t_train), "y": np.asarray(y_train_t, dtype=float)})
    m = Prophet(
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", 0.05)),
        seasonality_prior_scale=float(params.get("seasonality_prior_scale", 10.0)),
        seasonality_mode=str(params.get("seasonality_mode", "additive")),
        n_changepoints=int(params.get("n_changepoints", 25)),
    )
    if X_train_raw is not None and feat_names:
        for i, name in enumerate(feat_names):
            m.add_regressor(name)
            train[name] = np.asarray(X_train_raw[:, i], dtype=float)

    m.fit(train)

    future = pd.DataFrame({"ds": pd.to_datetime(t_future[: int(horizon)])})
    if X_future_raw is not None and feat_names:
        for i, name in enumerate(feat_names):
            future[name] = np.asarray(X_future_raw[: int(horizon), i], dtype=float)
    forecast = m.predict(future)
    return np.asarray(forecast["yhat"].to_numpy(), dtype=float)


def _build_supervised(
    y: np.ndarray, X: Optional[np.ndarray], lag_window: int
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= lag_window:
        raise ValueError("Not enough rows to build supervised dataset.")
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    for i in range(lag_window, n):
        lags = y[i - lag_window : i][::-1]  # recent first
        if X is not None:
            feat = np.concatenate([lags, np.asarray(X[i], dtype=float)], axis=0)
        else:
            feat = lags
        Xs.append(feat)
        ys.append(float(y[i]))
    return np.vstack(Xs), np.asarray(ys, dtype=float)


def _predict_xgboost(
    y_train: np.ndarray,
    X_train: Optional[np.ndarray],
    X_future: Optional[np.ndarray],
    horizon: int,
    lag_window: int,
    params: Dict[str, Any],
) -> np.ndarray:
    from xgboost import XGBRegressor

    Xs, ys = _build_supervised(y_train, X_train, lag_window=lag_window)
    model = XGBRegressor(
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 600)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        objective="reg:squarederror",
        n_jobs=0,
        random_state=42,
    )
    model.fit(Xs, ys)

    history = list(np.asarray(y_train, dtype=float))
    preds: List[float] = []
    for i in range(int(horizon)):
        lags = np.asarray(history[-lag_window:][::-1], dtype=float)
        if X_future is not None:
            ex = np.asarray(X_future[i], dtype=float)
            feat = np.concatenate([lags, ex], axis=0)
        else:
            feat = lags
        pred = float(model.predict(feat.reshape(1, -1))[0])
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def _predict_deepar_torch(
    y_train: np.ndarray,
    X_train: Optional[np.ndarray],
    X_future: Optional[np.ndarray],
    horizon: int,
    lag_window: int,
    params: Dict[str, Any],
    log: Callable[[str], None],
) -> np.ndarray:
    import torch
    import torch.nn as nn

    device_pref = str(params.get("device", "auto"))
    if device_pref == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_size = int(params.get("hidden_size", 64))
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("dropout", 0.1))
    epochs = int(params.get("epochs", 20))
    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 64))

    y = np.asarray(y_train, dtype=float)
    X = np.asarray(X_train, dtype=float) if X_train is not None else None
    if not np.isfinite(y).all():
        raise ValueError("DeepAR received NaN/Inf in y_train after cleaning. Try a different missing-value method.")
    if X is not None and (not np.isfinite(X).all()):
        raise ValueError("DeepAR received NaN/Inf in feature matrix after cleaning. Try a different missing-value method.")
    n_exog = int(X.shape[1]) if X is not None else 0
    input_size = 1 + n_exog

    class DeepARLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, input_size)
            h, _ = self.rnn(x)
            out = self.head(h[:, -1, :])
            return out.squeeze(-1)

    model = DeepARLite().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Build training windows: context -> next value
    X_windows: List[np.ndarray] = []
    y_next: List[float] = []
    for i in range(lag_window, len(y)):
        y_ctx = y[i - lag_window : i]
        if X is not None:
            x_ctx = X[i - lag_window : i]
            seq = np.concatenate([y_ctx.reshape(-1, 1), x_ctx], axis=1)
        else:
            seq = y_ctx.reshape(-1, 1)
        X_windows.append(seq)
        y_next.append(float(y[i]))

    Xw = torch.tensor(np.stack(X_windows, axis=0), dtype=torch.float32)
    yn = torch.tensor(np.asarray(y_next, dtype=float), dtype=torch.float32)

    # Shuffle once per epoch
    idx = np.arange(len(X_windows))
    Xw = Xw.to(device)
    yn = yn.to(device)

    model.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        total = 0.0
        count = 0
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            xb = Xw[b]
            yb = yn[b]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("DeepAR training became NaN/Inf. Try lower learning_rate or enable scaling.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item()) * len(b)
            count += len(b)
        if ep == 1 or ep == epochs or ep % 5 == 0:
            log(f"  DeepAR epoch {ep}/{epochs} - train MSE: {total / max(count, 1):.6g} (device={device.type})")

    # Forecast autoregressively
    model.eval()
    history_y = list(np.asarray(y, dtype=float))
    preds: List[float] = []
    Xf = np.asarray(X_future, dtype=float) if X_future is not None else None
    with torch.no_grad():
        for i in range(int(horizon)):
            y_ctx = np.asarray(history_y[-lag_window:], dtype=float)
            if Xf is not None:
                x_ctx = np.asarray(Xf[max(0, i - lag_window) : i], dtype=float)
                # For context, we only have future exog aligned with future steps; use last known exog if needed.
                # Simpler: if not enough future exog for context, pad with first row.
                if x_ctx.shape[0] < lag_window:
                    pad = np.repeat(Xf[[0]], repeats=(lag_window - x_ctx.shape[0]), axis=0)
                    x_ctx = np.vstack([pad, x_ctx]) if x_ctx.size else pad
                seq = np.concatenate([y_ctx.reshape(-1, 1), x_ctx[-lag_window:]], axis=1)
            else:
                seq = y_ctx.reshape(-1, 1)
            xb = torch.tensor(seq.reshape(1, lag_window, -1), dtype=torch.float32, device=device)
            pred = float(model(xb).cpu().numpy().reshape(-1)[0])
            preds.append(pred)
            history_y.append(pred)
    return np.asarray(preds, dtype=float)

