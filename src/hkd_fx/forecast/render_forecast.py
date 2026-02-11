from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_chinese_font() -> None:
    for font_name in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


def plot_forecast_png(last_history: pd.Series, forecast: pd.Series, output_path: Path, title: str) -> None:
    _ensure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3.6))
    plt.plot(last_history.index, last_history.values, color="#1f77b4", label="实际(近15天)")
    if len(forecast) > 0:
        join_x = [last_history.index[-1], forecast.index[0]]
        join_y = [last_history.values[-1], forecast.iloc[0]]
        plt.plot(join_x, join_y, color="#ff7f0e", linestyle="--", alpha=0.7)

    plt.plot(forecast.index, forecast.values, color="#ff7f0e", linestyle="--", label="预测(30天)")
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("汇率")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    min_v = min(np.nanmin(last_history.values), np.nanmin(forecast.values))
    max_v = max(np.nanmax(last_history.values), np.nanmax(forecast.values))
    pad = (max_v - min_v) * 0.05 if max_v > min_v else 1.0
    plt.ylim(min_v - pad, max_v + pad)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_forecast_html(last_history: pd.Series, forecast: pd.Series, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates_actual = [d.strftime("%Y-%m-%d") for d in last_history.index]
    vals_actual = [float(v) for v in last_history.values]
    dates_forecast = [d.strftime("%Y-%m-%d") for d in forecast.index]
    vals_forecast = [float(v) for v in forecast.values]

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
    const forecastLabels = {dates_forecast};
    const forecastData = {vals_forecast};
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
    output_path.write_text(html, encoding="utf-8")

