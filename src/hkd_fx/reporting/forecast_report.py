from __future__ import annotations

from pathlib import Path


def write_forecast_report(path: Path, info: dict) -> None:
    lines = [
        "LSTM(多特征·直接多步)预测与方法说明",
        "================================",
        f"当前日期: {info.get('current_date')}",
        f"目标列: {info.get('rate_col')}",
        f"数据起止: {info.get('start_date')} 至 {info.get('end_date')}",
        f"lookback窗口: {info.get('lookback')} 天 | 预测步长: {info.get('horizon')} 天",
        "",
        "一、方法概述",
        "- 多特征输入：周内/月底周期特征、统计特征与周末指示。",
        "- 直接多步训练：一次输出未来序列，降低递归误差累积。",
        "",
        "二、评估指标(最近窗口)",
        f"- RMSE: {info.get('rmse'):.4f}",
        f"- MAE: {info.get('mae'):.4f}",
        f"- MAPE(%): {info.get('mape'):.4f}",
        f"- 评估窗口(日): {info.get('window_days')}",
        "",
        "三、预测区间统计",
        f"- 区间: {info.get('fc_start')} 至 {info.get('fc_end')}",
        f"- 最小/最大: {info.get('fc_min'):.4f} / {info.get('fc_max'):.4f}",
        f"- 净变动: {info.get('fc_change'):.4f}",
        "",
        "四、趋势文字分析",
        "(1) 近15天实际：",
        info.get("actual_text", "(无)"),
        "",
        "(2) 未来30天预测：",
        info.get("forecast_text", "(无)"),
        "",
        "五、文件索引",
        f"- 交互图: {info.get('html_path')}",
        f"- 静态图: {info.get('png_path')}",
        f"- 预测CSV: {info.get('csv_path')}",
        f"- 报告TXT: {info.get('report_path')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")

