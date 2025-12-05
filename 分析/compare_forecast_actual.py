# -*- coding: utf-8 -*-
"""
预测值与实际值对比分析脚本
对比 LSTM 预测的未来30天汇率与实际汇率数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
from pathlib import Path
from datetime import datetime

# ============ 中文字体配置 ============
def ensure_chinese_font():
    """配置中文字体和负号显示"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "FangSong", "STSong", "STHeiti", "Arial Unicode MS"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False

ensure_chinese_font()

# ============ 数据读取 ============
def read_csv_safely(path, encodings=["utf-8-sig", "utf-8", "gbk", "gb18030"]):
    """多编码安全读取 CSV"""
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")

def parse_date(date_str):
    """解析日期字符串，支持多种格式"""
    date_str = str(date_str).strip()
    # 格式: 2025年11月04日
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # 格式: 2025-11-04
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def load_actual_data(csv_path, rate_col="汇卖价"):
    """加载实际汇率数据"""
    df = read_csv_safely(csv_path)
    df["日期"] = df["日期"].apply(parse_date)
    df = df.dropna(subset=["日期"])
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=[rate_col])
    df = df.sort_values("日期").reset_index(drop=True)
    return df[["日期", rate_col]].rename(columns={rate_col: "实际值"})

def load_forecast_data(csv_path):
    """加载预测数据"""
    df = read_csv_safely(csv_path)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"预测值": "预测值"})
    return df[["日期", "预测值"]]

# ============ 对比分析 ============
def compute_metrics(actual, forecast):
    """计算评估指标"""
    diff = forecast - actual
    abs_diff = np.abs(diff)
    
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(abs_diff)
    mape = np.mean(abs_diff / actual) * 100
    
    # 方向准确率（涨跌方向）
    actual_dir = np.sign(np.diff(actual))
    forecast_dir = np.sign(np.diff(forecast))
    direction_acc = np.mean(actual_dir == forecast_dir) * 100
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "方向准确率(%)": direction_acc,
        "最大误差": np.max(abs_diff),
        "最小误差": np.min(abs_diff),
        "平均误差": np.mean(diff),
    }

def analyze_comparison(merged_df):
    """分析对比结果"""
    actual = merged_df["实际值"].values
    forecast = merged_df["预测值"].values
    
    metrics = compute_metrics(actual, forecast)
    
    # 趋势分析
    actual_trend = actual[-1] - actual[0]
    forecast_trend = forecast[-1] - forecast[0]
    
    return metrics, actual_trend, forecast_trend

# ============ 可视化 ============
def plot_comparison(merged_df, metrics, output_path):
    """绘制对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    dates = merged_df["日期"]
    actual = merged_df["实际值"]
    forecast = merged_df["预测值"]
    
    # 图1: 实际值 vs 预测值
    ax1 = axes[0]
    ax1.plot(dates, actual, "b-o", label="实际值", markersize=4, linewidth=1.5)
    ax1.plot(dates, forecast, "r--s", label="预测值", markersize=4, linewidth=1.5)
    ax1.fill_between(dates, actual, forecast, alpha=0.3, color="gray", label="误差区间")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("汇卖价")
    ax1.set_title("港币汇卖价：预测值 vs 实际值对比", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 图2: 误差分布
    ax2 = axes[1]
    errors = forecast - actual
    colors = ["green" if e >= 0 else "red" for e in errors]
    ax2.bar(dates, errors, color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=np.mean(errors), color="blue", linestyle="--", linewidth=1, label=f"平均误差: {np.mean(errors):.4f}")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("预测误差 (预测 - 实际)")
    ax2.set_title("每日预测误差分布", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 图3: 累计误差
    ax3 = axes[2]
    cumulative_error = np.cumsum(np.abs(errors))
    ax3.plot(dates, cumulative_error, "purple", linewidth=2, marker="o", markersize=3)
    ax3.fill_between(dates, 0, cumulative_error, alpha=0.3, color="purple")
    ax3.set_xlabel("日期")
    ax3.set_ylabel("累计绝对误差")
    ax3.set_title("累计绝对误差趋势", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加指标文本框
    metrics_text = (
        f"评估指标:\n"
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAE: {metrics['MAE']:.4f}\n"
        f"MAPE: {metrics['MAPE(%)']:.2f}%\n"
        f"方向准确率: {metrics['方向准确率(%)']:.1f}%"
    )
    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存: {output_path}")

def write_html_chart(merged_df, metrics, output_path):
    """生成交互式 HTML 图表"""
    dates = merged_df["日期"].dt.strftime("%Y-%m-%d").tolist()
    actual = merged_df["实际值"].tolist()
    forecast = merged_df["预测值"].tolist()
    errors = (merged_df["预测值"] - merged_df["实际值"]).tolist()
    
    html_content = f"""<!DOCTYPE html>
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
        <h1>🔮 港币汇卖价：预测值 vs 实际值对比分析</h1>
        
        <div class="metrics-box">
            <div class="metric">
                <div class="metric-value">{metrics['RMSE']:.4f}</div>
                <div class="metric-label">RMSE (均方根误差)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['MAE']:.4f}</div>
                <div class="metric-label">MAE (平均绝对误差)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['MAPE(%)']:.2f}%</div>
                <div class="metric-label">MAPE (平均百分比误差)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['方向准确率(%)']:.1f}%</div>
                <div class="metric-label">方向准确率</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="comparisonChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="errorChart"></canvas>
        </div>
    </div>
    
    <script>
        const dates = {dates};
        const actual = {actual};
        const forecast = {forecast};
        const errors = {errors};
        
        // 对比图
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: '实际值',
                        data: actual,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: false,
                        tension: 0.1
                    }},
                    {{
                        label: '预测值',
                        data: forecast,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: false,
                        tension: 0.1,
                        borderDash: [5, 5]
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: '预测值 vs 实际值趋势对比', font: {{ size: 16 }} }}
                }},
                scales: {{
                    y: {{ title: {{ display: true, text: '汇卖价' }} }}
                }}
            }}
        }});
        
        // 误差图
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
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: '每日预测误差分布', font: {{ size: 16 }} }}
                }},
                scales: {{
                    y: {{ title: {{ display: true, text: '误差值' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"交互图已保存: {output_path}")

def write_report(merged_df, metrics, actual_trend, forecast_trend, output_path):
    """生成对比报告"""
    report = f"""预测值与实际值对比分析报告
{'='*50}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
对比区间: {merged_df['日期'].min().strftime('%Y-%m-%d')} 至 {merged_df['日期'].max().strftime('%Y-%m-%d')}
对比天数: {len(merged_df)} 天

一、评估指标
{'─'*50}
RMSE (均方根误差): {metrics['RMSE']:.4f}
MAE (平均绝对误差): {metrics['MAE']:.4f}
MAPE (平均百分比误差): {metrics['MAPE(%)']:.2f}%
方向准确率: {metrics['方向准确率(%)']:.1f}%
最大误差: {metrics['最大误差']:.4f}
最小误差: {metrics['最小误差']:.4f}
平均误差: {metrics['平均误差']:.4f}

二、趋势对比
{'─'*50}
实际趋势变动: {actual_trend:.4f} ({'上涨' if actual_trend > 0 else '下跌'})
预测趋势变动: {forecast_trend:.4f} ({'上涨' if forecast_trend > 0 else '下跌'})
趋势方向: {'一致 ✓' if (actual_trend > 0) == (forecast_trend > 0) else '不一致 ✗'}

三、数据统计
{'─'*50}
实际值范围: {merged_df['实际值'].min():.2f} ~ {merged_df['实际值'].max():.2f}
预测值范围: {merged_df['预测值'].min():.4f} ~ {merged_df['预测值'].max():.4f}
实际值均值: {merged_df['实际值'].mean():.4f}
预测值均值: {merged_df['预测值'].mean():.4f}

四、每日对比明细
{'─'*50}
{'日期':<12} {'实际值':<10} {'预测值':<12} {'误差':<10} {'误差率':<10}
"""
    
    for _, row in merged_df.iterrows():
        date_str = row['日期'].strftime('%Y-%m-%d')
        actual = row['实际值']
        forecast = row['预测值']
        error = forecast - actual
        error_pct = abs(error) / actual * 100
        report += f"{date_str:<12} {actual:<10.2f} {forecast:<12.4f} {error:<+10.4f} {error_pct:<10.2f}%\n"
    
    report += f"""
五、结论
{'─'*50}
"""
    if metrics['MAPE(%)'] < 1:
        report += "预测精度: 优秀 (MAPE < 1%)\n"
    elif metrics['MAPE(%)'] < 5:
        report += "预测精度: 良好 (MAPE < 5%)\n"
    else:
        report += "预测精度: 一般 (MAPE >= 5%)\n"
    
    if metrics['方向准确率(%)'] >= 60:
        report += f"方向预测: 较好 (准确率 {metrics['方向准确率(%)']:.1f}%)\n"
    else:
        report += f"方向预测: 需改进 (准确率 {metrics['方向准确率(%)']:.1f}%)\n"
    
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(report)
    print(f"对比报告已保存: {output_path}")

def save_comparison_csv(merged_df, output_path):
    """保存对比数据 CSV"""
    df = merged_df.copy()
    df["误差"] = df["预测值"] - df["实际值"]
    df["误差率(%)"] = abs(df["误差"]) / df["实际值"] * 100
    df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"对比数据已保存: {output_path}")

# ============ 主函数 ============
def main():
    import argparse
    parser = argparse.ArgumentParser(description="预测值与实际值对比分析")
    parser.add_argument("--actual", default="actual_rates_recent.csv", help="实际汇率数据文件")
    parser.add_argument("--forecast", default="分析/未来30天预测.csv", help="预测数据文件")
    parser.add_argument("--rate_col", default="汇卖价", help="汇率列名")
    parser.add_argument("--outdir", default="分析", help="输出目录")
    args = parser.parse_args()
    
    # 确定路径
    base_dir = Path(__file__).parent.parent
    actual_path = base_dir / args.actual
    forecast_path = base_dir / args.forecast
    outdir = base_dir / args.outdir
    outdir.mkdir(exist_ok=True)
    
    print(f"加载实际数据: {actual_path}")
    actual_df = load_actual_data(actual_path, args.rate_col)
    print(f"  - 数据范围: {actual_df['日期'].min()} ~ {actual_df['日期'].max()}")
    print(f"  - 数据条数: {len(actual_df)}")
    
    print(f"\n加载预测数据: {forecast_path}")
    forecast_df = load_forecast_data(forecast_path)
    print(f"  - 数据范围: {forecast_df['日期'].min()} ~ {forecast_df['日期'].max()}")
    print(f"  - 数据条数: {len(forecast_df)}")
    
    # 合并数据
    merged_df = pd.merge(actual_df, forecast_df, on="日期", how="inner")
    merged_df = merged_df.sort_values("日期").reset_index(drop=True)
    print(f"\n合并后数据条数: {len(merged_df)}")
    print(f"对比区间: {merged_df['日期'].min()} ~ {merged_df['日期'].max()}")
    
    if len(merged_df) == 0:
        print("错误: 没有匹配的日期数据!")
        return
    
    # 分析
    metrics, actual_trend, forecast_trend = analyze_comparison(merged_df)
    
    print(f"\n{'='*50}")
    print("评估指标:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE(%)']:.2f}%")
    print(f"  方向准确率: {metrics['方向准确率(%)']:.1f}%")
    print(f"{'='*50}")
    
    # 输出文件
    plot_comparison(merged_df, metrics, outdir / "预测对比图.png")
    write_html_chart(merged_df, metrics, outdir / "预测对比图.html")
    write_report(merged_df, metrics, actual_trend, forecast_trend, outdir / "预测对比报告.txt")
    save_comparison_csv(merged_df, outdir / "预测对比数据.csv")
    
    print(f"\n完成! 输出文件位于: {outdir}")

if __name__ == "__main__":
    main()
