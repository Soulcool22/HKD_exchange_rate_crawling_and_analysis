from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "STSong",
        "STHeiti",
        "Arial Unicode MS",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_comparison_png(merged_df: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    _ensure_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    dates = merged_df["日期"]
    actual = merged_df["实际值"]
    forecast = merged_df["预测值"]

    ax1 = axes[0]
    ax1.plot(dates, actual, "b-o", label="实际值", markersize=4, linewidth=1.5)
    ax1.plot(dates, forecast, "r--s", label="预测值", markersize=4, linewidth=1.5)
    ax1.fill_between(dates, actual, forecast, alpha=0.3, color="gray", label="误差区间")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("汇率")
    ax1.set_title("预测值 vs 实际值对比", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    errors = forecast - actual
    colors = ["green" if val >= 0 else "red" for val in errors]
    ax2.bar(dates, errors, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=np.mean(errors), color="blue", linestyle="--", linewidth=1, label=f"平均误差: {np.mean(errors):.4f}")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("预测误差 (预测 - 实际)")
    ax2.set_title("每日预测误差分布", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    ax3 = axes[2]
    cumulative_error = np.cumsum(np.abs(errors))
    ax3.plot(dates, cumulative_error, "purple", linewidth=2, marker="o", markersize=3)
    ax3.fill_between(dates, 0, cumulative_error, alpha=0.3, color="purple")
    ax3.set_xlabel("日期")
    ax3.set_ylabel("累计绝对误差")
    ax3.set_title("累计绝对误差趋势", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    direction = metrics.get("方向准确率(%)")
    direction_text = "N/A" if direction is None else f"{direction:.1f}%"
    metrics_text = (
        f"评估指标:\n"
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAE: {metrics['MAE']:.4f}\n"
        f"MAPE: {metrics['MAPE(%)']:.2f}%\n"
        f"方向准确率: {direction_text}"
    )
    fig.text(
        0.02,
        0.98,
        metrics_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_comparison_html(merged_df: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = merged_df["日期"].dt.strftime("%Y-%m-%d").tolist()
    actual = merged_df["实际值"].tolist()
    forecast = merged_df["预测值"].tolist()
    errors = (merged_df["预测值"] - merged_df["实际值"]).tolist()
    direction = metrics.get("方向准确率(%)")
    direction_text = "N/A" if direction is None else f"{direction:.1f}%"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>预测值与实际值对比分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #333; }}
        .metrics-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; margin: 20px 0;
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;
        }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        canvas {{ max-height: 400px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 预测值 vs 实际值对比分析</h1>
        <div class="metrics-box">
            <div class="metric"><div class="metric-value">{metrics['RMSE']:.4f}</div><div class="metric-label">RMSE</div></div>
            <div class="metric"><div class="metric-value">{metrics['MAE']:.4f}</div><div class="metric-label">MAE</div></div>
            <div class="metric"><div class="metric-value">{metrics['MAPE(%)']:.2f}%</div><div class="metric-label">MAPE</div></div>
            <div class="metric"><div class="metric-value">{direction_text}</div><div class="metric-label">方向准确率</div></div>
        </div>
        <div class="chart-container"><canvas id="comparisonChart"></canvas></div>
        <div class="chart-container"><canvas id="errorChart"></canvas></div>
    </div>
    <script>
        const dates = {dates};
        const actual = {actual};
        const forecast = {forecast};
        const errors = {errors};

        new Chart(document.getElementById('comparisonChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{ label: '实际值', data: actual, borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.1)', fill: false, tension: 0.1 }},
                    {{ label: '预测值', data: forecast, borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.1)', fill: false, tension: 0.1, borderDash: [5, 5] }}
                ]
            }},
            options: {{ responsive: true, plugins: {{ title: {{ display: true, text: '预测值 vs 实际值趋势对比', font: {{ size: 16 }} }} }} }}
        }});

        new Chart(document.getElementById('errorChart'), {{
            type: 'bar',
            data: {{
                labels: dates,
                datasets: [{{
                    label: '预测误差 (预测 - 实际)',
                    data: errors,
                    backgroundColor: errors.map(e => e >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'),
                    borderColor: errors.map(e => e >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'),
                    borderWidth: 1
                }}]
            }},
            options: {{ responsive: true, plugins: {{ title: {{ display: true, text: '每日预测误差分布', font: {{ size: 16 }} }} }} }}
        }});
    </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def write_comparison_report(merged_df: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    direction = metrics.get("方向准确率(%)")
    direction_text = "N/A" if direction is None else f"{direction:.1f}%"

    report = [
        "预测值与实际值对比分析报告",
        "=" * 50,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"对比区间: {merged_df['日期'].min().strftime('%Y-%m-%d')} 至 {merged_df['日期'].max().strftime('%Y-%m-%d')}",
        f"对比天数: {len(merged_df)} 天",
        "",
        "一、评估指标",
        "─" * 50,
        f"RMSE: {metrics['RMSE']:.4f}",
        f"MAE: {metrics['MAE']:.4f}",
        f"MAPE: {metrics['MAPE(%)']:.2f}%",
        f"方向准确率: {direction_text}",
        f"最大误差: {metrics['最大误差']:.4f}",
        f"最小误差: {metrics['最小误差']:.4f}",
        f"平均误差: {metrics['平均误差']:.4f}",
        "",
        "二、每日对比明细",
        "─" * 50,
        f"{'日期':<12} {'实际值':<10} {'预测值':<12} {'误差':<10} {'误差率':<10}",
    ]

    for _, row in merged_df.iterrows():
        date_str = row["日期"].strftime("%Y-%m-%d")
        actual = row["实际值"]
        forecast = row["预测值"]
        error = forecast - actual
        err_pct = abs(error) / (actual + 1e-8) * 100
        report.append(f"{date_str:<12} {actual:<10.4f} {forecast:<12.4f} {error:<+10.4f} {err_pct:<10.2f}%")

    output_path.write_text("\n".join(report), encoding="utf-8-sig")


def write_comparison_csv(merged_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = merged_df.copy()
    df["误差"] = df["预测值"] - df["实际值"]
    df["误差率(%)"] = abs(df["误差"]) / (df["实际值"] + 1e-8) * 100
    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

