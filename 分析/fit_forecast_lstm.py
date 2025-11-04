import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def _read_csv_safely(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def _ensure_chinese_font():
    for font_name in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


def load_series(csv_path: str, rate_col: str) -> pd.Series:
    df = _read_csv_safely(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_col_candidates = ["日期", "date", "Date"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV中未找到日期列（期望: 日期/date/Date）")
    if rate_col not in df.columns:
        raise ValueError(f"CSV中未找到目标列: {rate_col}")

    ser_date = df[date_col].astype(str).str.strip()
    if ser_date.iloc[0].find("年") != -1 and ser_date.iloc[0].find("月") != -1:
        try:
            df[date_col] = pd.to_datetime(ser_date, format="%Y年%m月%d日", errors="coerce")
        except Exception:
            tmp = (
                ser_date.str.replace("年", "-", regex=False)
                        .str.replace("月", "-", regex=False)
                        .str.replace("日", "", regex=False)
            )
            df[date_col] = pd.to_datetime(tmp, format="%Y-%m-%d", errors="coerce")
    else:
        df[date_col] = pd.to_datetime(ser_date, errors="coerce")

    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col)
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=[rate_col])
    s = df.set_index(date_col)[rate_col].copy()
    s = s[~s.index.duplicated(keep="last")]
    return s


def as_daily_continuous(s: pd.Series) -> pd.Series:
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s_daily = s.reindex(full_index).ffill()
    return s_daily


def build_features(s: pd.Series) -> pd.DataFrame:
    """
    构造多特征：周期(周内/月底) sin/cos，统计(1日差分、7日均值偏差、7日标准差)，周末指示。
    返回与 s 对齐的 DataFrame，列：['y','dow_sin','dow_cos','dom_sin','dom_cos','delta1','ma7_diff','std7','is_weekend']
    """
    idx = s.index
    dow = pd.Series(idx.dayofweek, index=idx)
    dom = pd.Series(idx.day, index=idx)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    dom_sin = np.sin(2 * np.pi * dom / 31.0)
    dom_cos = np.cos(2 * np.pi * dom / 31.0)

    delta1 = s.diff().fillna(0.0)
    ma7 = s.rolling(7, min_periods=1).mean()
    std7 = s.rolling(7, min_periods=1).std().fillna(0.0)
    ma7_diff = (s - ma7)
    is_weekend = (dow >= 5).astype(np.float32)

    df = pd.DataFrame({
        'y': s.astype(np.float32).values,
        'dow_sin': dow_sin.astype(np.float32).values,
        'dow_cos': dow_cos.astype(np.float32).values,
        'dom_sin': dom_sin.astype(np.float32).values,
        'dom_cos': dom_cos.astype(np.float32).values,
        'delta1': delta1.astype(np.float32).values,
        'ma7_diff': ma7_diff.astype(np.float32).values,
        'std7': std7.astype(np.float32).values,
        'is_weekend': is_weekend.astype(np.float32).values,
    }, index=idx)
    return df


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """对数值特征做列级标准化（z-score）；周期 sin/cos 与周末指示无需缩放。"""
    X = df.copy()
    for col in ['delta1', 'ma7_diff', 'std7']:
        if col in X.columns:
            m = float(np.nanmean(X[col].values))
            s = float(np.nanstd(X[col].values))
            s = s if s > 1e-8 else 1.0
            X[col] = ((X[col] - m) / s).astype(np.float32)
    return X.astype(np.float32).values


def scale_series(s: pd.Series):
    v = s.values.astype(np.float32)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax - vmin < 1e-12:
        scaled = np.zeros_like(v)
    else:
        scaled = (v - vmin) / (vmax - vmin)
    return scaled, vmin, vmax


def inverse_scale(arr, vmin, vmax):
    return arr * (vmax - vmin) + vmin


def make_supervised_multistep(features: np.ndarray, target_scaled: np.ndarray, lookback: int = 30, horizon: int = 30):
    """生成多步监督样本：X (n_samples, lookback, n_features)，y (n_samples, horizon)"""
    X_list, y_list = [], []
    n = len(target_scaled)
    n_features = features.shape[1]
    for i in range(n - lookback - horizon + 1):
        X_list.append(features[i:i+lookback, :])
        y_list.append(target_scaled[i+lookback:i+lookback+horizon])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    X = X.reshape((X.shape[0], lookback, n_features))
    y = y.reshape((y.shape[0], horizon))
    return X, y


def build_model(n_features: int, lookback: int = 30, horizon: int = 30):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=(lookback, n_features)),
        LSTM(32, activation='tanh'),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def _series_stats(s: pd.Series):
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    net = float(end - start)
    pct = float((net / (start + 1e-8)) * 100.0)
    vmin = float(np.min(s.values))
    vmax = float(np.max(s.values))
    dmin = s.index[int(np.argmin(s.values))]
    dmax = s.index[int(np.argmax(s.values))]
    diffs = s.diff().dropna()
    mean_diff = float(np.mean(diffs.values)) if len(diffs) else 0.0
    std_diff = float(np.std(diffs.values)) if len(diffs) else 0.0
    pos = int(np.sum(diffs.values > 0))
    neg = int(np.sum(diffs.values < 0))
    # 连续上涨/下跌最长长度
    longest_up, longest_down = 0, 0
    cur_up, cur_down = 0, 0
    for d in diffs.values:
        if d > 0:
            cur_up += 1
            longest_up = max(longest_up, cur_up)
            cur_down = 0
        elif d < 0:
            cur_down += 1
            longest_down = max(longest_down, cur_down)
            cur_up = 0
        else:
            cur_up = 0
            cur_down = 0
    return {
        "start": start, "end": end, "net": net, "pct": pct,
        "vmin": vmin, "vmax": vmax, "dmin": dmin, "dmax": dmax,
        "mean_diff": mean_diff, "std_diff": std_diff,
        "pos": pos, "neg": neg, "longest_up": longest_up, "longest_down": longest_down
    }


def _build_trend_text(name: str, s: pd.Series) -> str:
    st = _series_stats(s)
    return (
        f"{name}从 {s.index.min():%Y-%m-%d} 至 {s.index.max():%Y-%m-%d}，"
        f"起点 {st['start']:.4f}，终点 {st['end']:.4f}，净变动 {st['net']:.4f}（{st['pct']:.2f}%）。\n"
        f"最高值 {st['vmax']:.4f} 出现在 {st['dmax']:%Y-%m-%d}；最低值 {st['vmin']:.4f} 出现在 {st['dmin']:%Y-%m-%d}。\n"
        f"上涨天数 {st['pos']}，下跌天数 {st['neg']}，平均日变动 {st['mean_diff']:.4f}，波动（日差标准差） {st['std_diff']:.4f}。\n"
        f"最长连续上涨 {st['longest_up']} 天，最长连续下跌 {st['longest_down']} 天。"
    )


def evaluate_walk_forward(model_builder, features_scaled: np.ndarray, target_scaled: np.ndarray, lookback: int = 30, eval_days: int = 30, horizon: int = 30):
    """一次训练后对最近窗口进行直接多步预测评估。"""
    eval_days = min(eval_days, horizon, len(target_scaled) - lookback - 5)
    hist_X = features_scaled[:-eval_days]
    hist_y = target_scaled[:-eval_days]
    X_train, y_train = make_supervised_multistep(hist_X, hist_y, lookback, horizon)
    n_features = hist_X.shape[1]
    model = model_builder(n_features=n_features, lookback=lookback, horizon=horizon)
    es = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=35, batch_size=64, verbose=0, callbacks=[es])

    x_input = hist_X[-lookback:, :].reshape((1, lookback, n_features))
    yhat = model.predict(x_input, verbose=0)[0]  # shape (horizon,)
    preds = yhat[:eval_days]
    actuals = target_scaled[-eval_days:]

    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    mape = float(np.mean(np.abs((preds - actuals) / (actuals + 1e-8))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape, "window_days": int(eval_days)}


def forecast_future(model_builder, features_scaled: np.ndarray, target_scaled: np.ndarray, lookback: int = 30, steps: int = 30):
    """直接多步：一次训练，单次前向输出 steps 个未来值（在缩放空间）。"""
    X_train, y_train = make_supervised_multistep(features_scaled, target_scaled, lookback, steps)
    n_features = features_scaled.shape[1]
    model = model_builder(n_features=n_features, lookback=lookback, horizon=steps)
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=60, batch_size=64, verbose=0, callbacks=[es])
    x_input = features_scaled[-lookback:, :].reshape((1, lookback, n_features))
    yhat = model.predict(x_input, verbose=0)[0]
    return yhat.astype(np.float32)


def plot_png(last15: pd.Series, forecast: pd.Series, out_png: str, title: str):
    _ensure_chinese_font()
    plt.figure(figsize=(12, 3.6))
    plt.plot(last15.index, last15.values, color="#1f77b4", label="实际(近15天)")
    if len(forecast) > 0:
        join_x = [last15.index[-1], forecast.index[0]]
        join_y = [last15.values[-1], forecast.iloc[0]]
        plt.plot(join_x, join_y, color="#ff7f0e", linestyle="--", alpha=0.7)
    plt.plot(forecast.index, forecast.values, color="#ff7f0e", linestyle="--", label="预测(30天)")
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("汇率")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    ymin = min(np.nanmin(last15.values), np.nanmin(forecast.values))
    ymax = max(np.nanmax(last15.values), np.nanmax(forecast.values))
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def write_html_chart(last15: pd.Series, forecast: pd.Series, out_html: str, title: str):
    dates_actual = [d.strftime("%Y-%m-%d") for d in last15.index]
    vals_actual = [float(v) for v in last15.values]
    dates_fc = [d.strftime("%Y-%m-%d") for d in forecast.index]
    vals_fc = [float(v) for v in forecast.values]
    html = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; }} #container {{ max-width: 1200px; margin: 20px auto; }}</style>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.0\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1\"></script>
</head>
<body>
  <div id=\"container\">
    <h3>{title}</h3>
    <canvas id=\"chart\" height=\"120\"></canvas>
    <p>图例：<span style=\"color:#1f77b4\">实际(近15天)</span>，<span style=\"color:#ff7f0e\">预测(虚线)</span></p>
  </div>
  <script>
    const actualLabels = {dates_actual};
    const actualData = {vals_actual};
    const forecastLabels = {dates_fc};
    const forecastData = {vals_fc};
    const ctx = document.getElementById('chart').getContext('2d');
    const allLabels = actualLabels.concat(forecastLabels);
    const minVal = Math.min(...actualData, ...forecastData);
    const maxVal = Math.max(...actualData, ...forecastData);
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: allLabels,
        datasets: [
          {{ label: '实际(近15天)', data: actualData, borderColor: '#1f77b4', backgroundColor: 'rgba(31,119,180,0.1)', tension: 0.2, fill: false }},
          {{ label: '预测(30天)', data: Array(actualData.length).fill(null).concat(forecastData), borderColor: '#ff7f0e', backgroundColor: 'rgba(255,127,14,0.1)', borderDash: [6,4], tension: 0.2, fill: false }}
        ]
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          zoom: {{ pan: {{ enabled: true, mode: 'xy' }}, zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'xy' }} }},
          legend: {{ position: 'top' }},
          tooltip: {{
            enabled: true,
            callbacks: {{
              title: (items) => '日期：' + (items && items.length ? items[0].label : ''),
              label: (ctx) => {{
                const val = ctx.parsed.y;
                const ds = ctx.dataset && ctx.dataset.label ? ctx.dataset.label : '值';
                return ds + '：' + Number(val).toFixed(4);
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ type: 'time', time: {{ unit: 'day' }} }},
          y: {{ suggestedMin: minVal - (maxVal - minVal) * 0.05, suggestedMax: maxVal + (maxVal - minVal) * 0.05 }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def write_report(path: str, info: dict):
    lines = [
        "LSTM模型性能评估报告",
        "------------------",
        f"当前日期: {info.get('current_date')}",
        f"目标列: {info.get('rate_col')}",
        f"数据起止: {info.get('start_date')} 至 {info.get('end_date')}",
        f"lookback窗口: {info.get('lookback')}",
        "",
        f"评估窗口(日): {info.get('window_days')}",
        f"RMSE: {info.get('rmse'):.4f}",
        f"MAE: {info.get('mae'):.4f}",
        f"MAPE(%): {info.get('mape'):.4f}",
        "",
        f"预测区间: {info.get('fc_start')} 至 {info.get('fc_end')}",
        f"预测最小值: {info.get('fc_min'):.4f} | 最大值: {info.get('fc_max'):.4f}",
        f"区间净变动: {info.get('fc_change'):.4f}",
        "",
        "说明: 仅保留LSTM模型；HTML图支持悬浮显示具体汇率，趋势图保留便于观察。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_report_v2(path: str, info: dict):
    lines = []
    lines += [
        "LSTM(多特征·直接多步)预测与方法说明",
        "================================",
        f"当前日期: {info.get('current_date')}",
        f"目标列: {info.get('rate_col')}",
        f"数据起止: {info.get('start_date')} 至 {info.get('end_date')}",
        f"lookback窗口: {info.get('lookback')} 天 | 预测步长: {info.get('horizon')} 天",
        "",
        "一、方法概述",
        "- 多特征输入：周内/月底周期特征(dow_sin, dow_cos, dom_sin, dom_cos)，统计特征(1日差分delta1、7日均值偏差ma7_diff、7日标准差std7)，周末指示(is_weekend)。",
        "- 直接多步(Seq2Seq)训练：一次性输出未来30天，不使用递归，从而减少误差累积并学习跨步相关性。",
        "- 与普通LSTM的区别：普通LSTM通常使用单特征、单步训练+递归预测，易回归均值且跨步相关性弱；本方法在特征和训练目标上都更信息丰富。",
        "",
        "二、参数设置",
        "- 模型结构：LSTM(64, return_sequences=True) -> LSTM(32) -> Dense(30)。",
        "- 训练参数：评估阶段epochs=35, patience=4；预测阶段epochs=60, patience=5；batch_size=64；随机种子tf=42/np=42。",
        "- 特征缩放：delta1/ma7_diff/std7 采用列级z-score；目标序列采用min-max缩放并在输出处反缩放。",
        "- 数据处理：按日补齐并前向填充，保证等间隔时间步。",
        "",
        "三、评估指标(最近30天)",
        f"- RMSE: {info.get('rmse'):.4f}",
        f"- MAE: {info.get('mae'):.4f}",
        f"- MAPE(%): {info.get('mape'):.4f}",
        f"- 评估窗口(日): {info.get('window_days')}",
        "",
        "四、预测区间统计",
        f"- 区间: {info.get('fc_start')} 至 {info.get('fc_end')}",
        f"- 最小/最大: {info.get('fc_min'):.4f} / {info.get('fc_max'):.4f}",
        f"- 净变动: {info.get('fc_change'):.4f}",
        "",
        "五、趋势文字分析",
        "(1) 近15天实际：",
        info.get("actual_text", "(无)"),
        "",
        "(2) 未来30天预测：",
        info.get("forecast_text", "(无)"),
        "",
        "六、文件索引",
        "- 交互图: 分析/未来30天预测.html",
        "- 静态图: 分析/未来30天预测.png",
        "- 预测CSV: 分析/未来30天预测.csv",
        "- 报告TXT: 分析/预测报告.txt",
        "",
        "七、复现命令",
        f"python 分析/fit_forecast_lstm.py --rate_col \"{info.get('rate_col')}\" --lookback {info.get('lookback')}"
    ]
    # Windows记事本友好编码
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="HKD 30天趋势预测 (LSTM)")
    parser.add_argument("--input", default="hkd_rates_30pages.csv")
    parser.add_argument("--outdir", default="分析")
    parser.add_argument("--rate_col", default="汇卖价")
    parser.add_argument("--current_date", default=None, help="当前日期，如 2025-11-03")
    parser.add_argument("--lookback", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    s_raw = load_series(args.input, args.rate_col)
    current_date = pd.to_datetime(args.current_date) if args.current_date else s_raw.index.max()
    s_raw = s_raw.loc[:current_date]
    s = as_daily_continuous(s_raw)

    # 构造多特征并缩放
    feat_df = build_features(s)
    y = feat_df['y']
    X = scale_features(feat_df.drop(columns=['y']))
    scaled, vmin, vmax = scale_series(y)

    # 直接多步评估与预测
    eval_metrics = evaluate_walk_forward(build_model, X, scaled, lookback=args.lookback, eval_days=30, horizon=30)
    fc_scaled = forecast_future(build_model, X, scaled, lookback=args.lookback, steps=30)
    fc_values = inverse_scale(fc_scaled, vmin, vmax)
    fc_index = pd.date_range(current_date + timedelta(days=1), periods=30, freq="D")
    forecast = pd.Series(fc_values, index=fc_index)

    # Last 15 days
    last15_index_start = current_date - timedelta(days=14)
    last15 = s.loc[last15_index_start: current_date]

    chart_title = f"{current_date.strftime('%Y-%m-%d')} 至 {(current_date + timedelta(days=30)).strftime('%Y-%m-%d')} 趋势预测（{args.rate_col}，LSTM·多特征·直接多步）"
    out_png = os.path.join(args.outdir, "未来30天预测.png")
    out_html = os.path.join(args.outdir, "未来30天预测.html")
    plot_png(last15, forecast, out_png, chart_title)
    write_html_chart(last15, forecast, out_html, chart_title)

    report_path = os.path.join(args.outdir, "预测报告.txt")
    # 趋势文字分析
    actual_text = _build_trend_text("近15天实际", last15)
    forecast_text = _build_trend_text("未来30天预测", forecast)
    report_info = {
        "current_date": current_date.strftime("%Y-%m-%d"),
        "rate_col": args.rate_col,
        "start_date": s.index.min().strftime("%Y-%m-%d"),
        "end_date": s.index.max().strftime("%Y-%m-%d"),
        "lookback": args.lookback,
        "horizon": 30,
        **eval_metrics,
        "fc_start": fc_index.min().strftime("%Y-%m-%d"),
        "fc_end": fc_index.max().strftime("%Y-%m-%d"),
        "fc_min": float(np.min(forecast.values)),
        "fc_max": float(np.max(forecast.values)),
        "fc_change": float(forecast.values[-1] - forecast.values[0]),
        "actual_text": actual_text,
        "forecast_text": forecast_text,
    }
    write_report_v2(report_path, report_info)

    out_csv = os.path.join(args.outdir, "未来30天预测.csv")
    fc_df = pd.DataFrame({"日期": forecast.index.strftime("%Y-%m-%d"), "预测值": forecast.values})
    fc_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"未来30天预测PNG: {out_png}")
    print(f"未来30天预测HTML: {out_html}")
    print(f"预测报告: {report_path}")
    print(f"未来30天预测CSV: {out_csv}")


if __name__ == "__main__":
    main()