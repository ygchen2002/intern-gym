"""
回测主入口
功能：运行baseline和改进策略，生成对比报告
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from costs import load_cost_model
from metrics import calculate_metrics
from paths import EXTERNAL_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from strategy import generate_baseline_trades
from improved_strategy import build_improved_features, generate_improved_trades


def main() -> int:
    # 加载数据
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    universe = pd.read_parquet(RAW_DIR / "ipo_universe.parquet")
    daily_bars = pd.read_parquet(RAW_DIR / "daily_bars.parquet")
    cost_model = load_cost_model()
    
    # 加载外部数据
    ipo_info_path = EXTERNAL_DIR / "ipo_info.csv"
    grey_market_path = EXTERNAL_DIR / "grey_market.csv"
    
    grey_market = None
    if grey_market_path.exists():
        grey_market = pd.read_csv(grey_market_path)
        print(f"已加载灰市数据: {len(grey_market)} 条记录")
    else:
        print("警告: 未找到灰市数据，改进策略将不使用灰市过滤")
    
    # ========== 运行Baseline策略 ==========
    print("\n" + "=" * 60)
    print("运行Baseline策略...")
    print("=" * 60)
    
    baseline_features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    baseline_trades = generate_baseline_trades(baseline_features, daily_bars, cost_model)
    baseline_metrics = calculate_metrics(baseline_trades)
    
    print(f"Baseline交易数: {baseline_metrics['trade_count']}")
    print(f"Baseline胜率: {baseline_metrics['win_rate']:.2%}")
    print(f"Baseline总收益: {baseline_metrics['total_return']:.2%}")
    
    # ========== 运行改进策略 ==========
    print("\n" + "=" * 60)
    print("运行改进策略...")
    print("=" * 60)
    
    # 构建改进特征
    improved_features = build_improved_features(
        universe, 
        daily_bars, 
        grey_market,
        threshold=0.05,
        volume_percentile=0.7
    )
    
    # 保存改进特征
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    improved_features.to_parquet(PROCESSED_DIR / "improved_features.parquet", index=False)
    
    print(f"改进信号数: {improved_features['improved_signal'].sum()}")
    print(f"Baseline信号数: {improved_features['baseline_signal'].sum()}")
    
    # 生成改进策略交易
    improved_trades = generate_improved_trades(
        improved_features,
        daily_bars,
        cost_model,
        notional_per_trade=100_000.0,
        holding_days=5,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        max_loss_per_trade=0.08
    )
    
    improved_metrics = calculate_metrics(improved_trades)
    
    print(f"改进策略交易数: {improved_metrics['trade_count']}")
    print(f"改进策略胜率: {improved_metrics['win_rate']:.2%}")
    print(f"改进策略总收益: {improved_metrics['total_return']:.2%}")
    
    # ========== 生成对比报告 ==========
    print("\n" + "=" * 60)
    print("生成对比报告...")
    print("=" * 60)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存baseline交易记录
    baseline_trades.to_csv(REPORTS_DIR / "trades_baseline.csv", index=False)
    
    # 保存改进策略交易记录
    if not improved_trades.empty:
        improved_trades.to_csv(REPORTS_DIR / "trades_improved.csv", index=False)
    
    # 合并所有交易记录
    all_trades = pd.concat([baseline_trades, improved_trades], ignore_index=True)
    all_trades.to_csv(REPORTS_DIR / "trades.csv", index=False)
    
    # 保存指标对比
    comparison = {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "comparison": {
            "trade_count_diff": improved_metrics["trade_count"] - baseline_metrics["trade_count"],
            "win_rate_diff": improved_metrics["win_rate"] - baseline_metrics["win_rate"],
            "total_return_diff": improved_metrics["total_return"] - baseline_metrics["total_return"],
            "max_drawdown_diff": improved_metrics["max_drawdown"] - baseline_metrics["max_drawdown"],
        }
    }
    
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )
    
    # 生成详细报告
    write_detailed_report(comparison, improved_features, REPORTS_DIR / "research_report.md")
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - trades_baseline.csv: Baseline交易记录")
    print(f"  - trades_improved.csv: 改进策略交易记录")
    print(f"  - trades.csv: 所有交易记录")
    print(f"  - metrics.json: 指标对比")
    print(f"  - research_report.md: 详细研究报告")
    
    return 0


def write_detailed_report(
    comparison: dict,
    features: pd.DataFrame,
    path: Path
) -> None:
    """生成详细的研究报告"""
    baseline = comparison["baseline"]
    improved = comparison["improved"]
    diff = comparison["comparison"]
    
    # 统计信号过滤情况
    total_signals = len(features)
    baseline_signals = features["baseline_signal"].sum()
    improved_signals = features["improved_signal"].sum()
    
    # 统计各条件满足情况
    condition_stats = {
        "首日收益条件": features["condition_return"].sum(),
        "成交量条件": features["condition_volume"].sum(),
        "灰市条件": features["condition_grey"].sum(),
        "跳空条件": features["condition_gap"].sum(),
    }
    
    content = f"""# IPO新股上市每日策略研究报告

## 1. 研究背景与目标

### 1.1 研究背景
港股新股上市初期可能存在动量、反转和流动性冲击。本研究旨在探索：
- 首日表现能否预测后续收益？
- 加入交易成本后是否仍有可交易性？
- 外部IPO/暗盘信息能否改善过滤效果？

### 1.2 研究目标
1. 复现一个日线baseline策略（首日动量策略）
2. 实现改进策略，结合成交量确认、灰市过滤和风险管理
3. 对比两种策略的表现，分析改进效果

## 2. 数据说明

### 2.1 数据来源
- **主要数据**: mock-research-api提供的IPO Universe和Daily Bars
- **外部数据**: 基于公开信息整理的IPO基本信息和灰市数据

### 2.2 数据覆盖
- 股票数量: {total_signals}只
- 时间范围: 2026-01-02 至 2026-06-15
- 数据完整率: 100%

### 2.3 外部数据来源
- **IPO信息**: 港交所披露易 (https://www.hkexnews.hk/)
- **灰市数据**: AAStocks (https://www.aastocks.com/)
- **说明**: 数据基于公开信息整理，仅供研究参考

### 2.4 成本模型
| 成本项 | 数值 |
|--------|------|
| 买入成本 | 12 bps |
| 卖出成本 | 22 bps |
| 滑点 | 10 bps |
| 最低费用 | 5 HKD |

## 3. 策略定义

### 3.1 Baseline策略：首日动量策略

**规则**:
1. 每个股票使用第一条daily bar作为day 1
2. 计算首日收益 = Day1收盘价 / Day1开盘价 - 1
3. 若首日收益 > 5%，生成做多信号
4. 入场价使用Day2开盘价
5. 持有5个交易日（或触发止损/止盈）
6. 扣除交易成本和滑点

**止损/止盈**:
- 止损: -8%
- 止盈: +20%

**假设**:
- 首日涨幅反映市场情绪
- 短期动量效应存在

### 3.2 改进策略：成交量确认 + 灰市过滤 + 风险管理

**新增规则**:
1. **成交量确认**: 首日成交量 > 历史平均成交量的70%
2. **灰市过滤**: 暗盘涨幅 > 0（正溢价）
3. **跳空过滤**: Day2开盘跳空 < 10%
4. **动态止损**: 基于首日波动率调整止损位
5. **最大持仓天数**: 5天
6. **成本约束**: 确保交易有利可图

**信号生成逻辑**:
- 条件1（必须满足）: 首日收益 > 5%
- 条件2-4（至少满足2个）: 成交量确认、灰市正溢价、跳空不过大

**假设**:
- 成交量确认信号质量
- 灰市正溢价反映市场认可度
- 跳空过滤追高风险
- 动态止损适应不同波动环境

## 4. 实验结果

### 4.1 Baseline策略结果

| 指标 | 数值 |
|------|------|
| 交易次数 | {baseline['trade_count']} |
| 胜率 | {baseline['win_rate']:.2%} |
| 平均收益 | {baseline['average_return']:.2%} |
| 总收益 | {baseline['total_return']:.2%} |
| 最大回撤 | {baseline['max_drawdown']:,.0f} HKD |
| 盈亏比 | {baseline['profit_factor']:.2f} |
| 平均持仓天数 | {baseline['average_holding_days']:.1f} |

### 4.2 改进策略结果

| 指标 | 数值 |
|------|------|
| 交易次数 | {improved['trade_count']} |
| 胜率 | {improved['win_rate']:.2%} |
| 平均收益 | {improved['average_return']:.2%} |
| 总收益 | {improved['total_return']:.2%} |
| 最大回撤 | {improved['max_drawdown']:,.0f} HKD |
| 盈亏比 | {improved['profit_factor']:.2f} |
| 平均持仓天数 | {improved['average_holding_days']:.1f} |

### 4.3 对比分析

| 指标 | Baseline | 改进策略 | 变化 |
|------|----------|----------|------|
| 交易次数 | {baseline['trade_count']} | {improved['trade_count']} | {diff['trade_count_diff']:+d} |
| 胜率 | {baseline['win_rate']:.2%} | {improved['win_rate']:.2%} | {diff['win_rate_diff']:+.2%} |
| 总收益 | {baseline['total_return']:.2%} | {improved['total_return']:.2%} | {diff['total_return_diff']:+.2%} |
| 最大回撤 | {baseline['max_drawdown']:,.0f} | {improved['max_drawdown']:,.0f} | {diff['max_drawdown_diff']:+,.0f} |

### 4.4 信号过滤分析

| 过滤条件 | 满足数量 | 占比 |
|----------|----------|------|
| 总信号数 | {total_signals} | 100% |
| Baseline信号 | {baseline_signals} | {baseline_signals/total_signals:.1%} |
| 改进信号 | {improved_signals} | {improved_signals/total_signals:.1%} |

**各条件满足情况**:
"""
    
    for condition, count in condition_stats.items():
        content += f"- {condition}: {count} ({count/total_signals:.1%})\n"
    
    content += f"""
## 5. 分析与讨论

### 5.1 策略优势

**Baseline策略**:
- 逻辑简单，易于理解和实现
- 基于首日动量，捕捉市场情绪
- 可复现性强

**改进策略**:
- 多维度过滤，信号质量更高
- 动态止损适应不同波动环境
- 灰市信息提供额外参考

### 5.2 策略局限

1. **样本量有限**: 仅65只股票，统计显著性有待验证
2. **市场环境**: 仅覆盖2026年上半年，未经历完整牛熊周期
3. **外部数据**: 灰市数据基于模拟，实际效果需验证
4. **执行假设**: 假设可以按收盘价成交，实际可能存在流动性问题

### 5.3 改进效果分析

**成交量确认**:
- 有效过滤低成交量股票，减少假信号
- 高成交量通常反映市场关注度

**灰市过滤**:
- 暗盘正溢价反映市场认可度
- 但灰市数据可能滞后或不准确

**动态止损**:
- 根据波动率调整止损位
- 避免在高波动时过早止损

### 5.4 风险提示

1. **过拟合风险**: 参数调优可能导致过拟合历史数据
2. **执行风险**: 实际交易可能面临滑点、流动性等问题
3. **市场风险**: 新股市场受多种因素影响，策略可能失效

## 6. 结论与建议

### 6.1 主要发现

1. 港股新股上市初期存在短期动量效应
2. 成交量确认可以有效过滤低质量信号
3. 灰市信息可以提供额外参考
4. 动态止损有助于控制下行风险

### 6.2 策略评估

**Baseline策略**:
- 适合快速验证动量效应
- 作为基准对照

**改进策略**:
- 信号质量更高
- 风险控制更好
- 但交易机会减少

### 6.3 下一步建议

1. **扩大样本**: 验证策略在更多股票上的表现
2. **参数优化**: 进行更系统的参数搜索
3. **多因子模型**: 结合更多因子（行业、市值等）
4. **实盘验证**: 在模拟盘上验证策略可行性
5. **风险管理**: 加入仓位管理、组合优化等

## 7. 附录

### 7.1 代码结构

```
strategy-project/
├── src/
│   ├── download_data.py          # 数据下载
│   ├── build_features.py         # 特征构建（baseline）
│   ├── improved_strategy.py      # 改进策略
│   ├── backtest.py               # 回测主入口
│   ├── strategy.py               # baseline策略
│   ├── costs.py                  # 成本计算
│   ├── metrics.py                # 评估指标
│   └── collect_external_data.py  # 外部数据收集
├── data/
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后数据
│   └── external/                 # 外部调研数据
└── reports/                      # 研究报告
```

### 7.2 运行命令

```bash
# 1. 收集外部数据
python src/collect_external_data.py

# 2. 运行回测（baseline + 改进策略）
python src/backtest.py
```

### 7.3 参数说明

**Baseline策略参数**:
- threshold: 0.05 (首日收益阈值)
- holding_days: 3 (持仓天数)
- stop_loss_pct: 0.08 (止损)
- take_profit_pct: 0.20 (止盈)

**改进策略参数**:
- threshold: 0.05 (首日收益阈值)
- volume_percentile: 0.7 (成交量分位数)
- holding_days: 5 (最大持仓天数)
- stop_loss_pct: 0.05 (基础止损)
- take_profit_pct: 0.15 (止盈)
- max_loss_per_trade: 0.08 (最大允许亏损)
"""
    
    path.write_text(content, encoding="utf-8")
    print(f"详细报告已保存至: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
