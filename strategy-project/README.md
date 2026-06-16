# Strategy Project: IPO / New Listing Daily Research

港股新股上市每日策略研究项目

## 项目概述

本项目旨在探索港股新股上市初期的日线动量效应，通过对比baseline策略和改进策略，验证多维度过滤和风险管理对策略表现的提升效果。

## 主要功能

1. **数据下载**: 从mock-research-api获取IPO Universe和Daily Bars数据
2. **外部数据收集**: 收集IPO基本信息和灰市数据
3. **特征构建**: 计算首日收益、成交量、跳空等特征
4. **策略回测**: 运行baseline和改进策略，生成对比报告
5. **单元测试**: 验证代码正确性

## 快速开始

### 环境要求

- Python 3.9+
- 依赖包：见 requirements.txt

### 安装

```bash
cd intern-gym
pip install -r requirements.txt
```

### 运行步骤

#### 1. 下载数据

```bash
cd strategy-project

# 使用本地数据（推荐）
python src/download_data.py --source-root ../research-data

# 或使用API数据（需要先启动API）
# make serve-research
# python src/download_data.py --base-url http://127.0.0.1:9041 --start 2026-01-01
```

#### 2. 收集外部数据

```bash
python src/collect_external_data.py
```

#### 3. 运行回测

```bash
python src/backtest.py
```

#### 4. 运行测试

```bash
cd ..
python -m pytest strategy-project/tests/test_improved_strategy.py -v
```

## 输出文件

### 数据文件

```
data/raw/
├── ipo_universe.parquet      # IPO股票池
├── daily_bars.parquet        # 日线数据
├── cost_model.json           # 成本模型
└── coverage_summary.json     # 数据覆盖摘要

data/processed/
├── features.parquet          # Baseline特征
└── improved_features.parquet # 改进策略特征

data/external/
├── ipo_info.csv              # IPO基本信息
└── grey_market.csv           # 灰市数据
```

### 报告文件

```
reports/
├── trades.csv                # 所有交易记录
├── trades_baseline.csv       # Baseline交易记录
├── trades_improved.csv       # 改进策略交易记录
├── metrics.json              # 指标对比
└── research_report.md        # 详细研究报告
```

## 策略说明

### Baseline策略：首日动量策略

**规则**:
1. 计算首日收益 = Day1收盘价 / Day1开盘价 - 1
2. 若首日收益 > 5%，在Day2开盘买入
3. 持有5个交易日（或触发止损/止盈）
4. 扣除交易成本和滑点

**参数**:
- threshold: 0.05 (首日收益阈值)
- holding_days: 5 (持仓天数)
- stop_loss_pct: 0.08 (止损)
- take_profit_pct: 0.20 (止盈)

### 改进策略：成交量确认 + 灰市过滤 + 风险管理

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

## 实验结果

### Baseline策略

| 指标 | 数值 |
|------|------|
| 交易次数 | 14 |
| 胜率 | 42.86% |
| 总收益 | 9.55% |
| 最大回撤 | -63,183 HKD |
| 盈亏比 | 1.09 |

### 改进策略

| 指标 | 数值 |
|------|------|
| 交易次数 | 14 |
| 胜率 | 28.57% |
| 总收益 | 105.24% |
| 最大回撤 | -76,107 HKD |
| 盈亏比 | 1.90 |

### 改进效果

- 总收益提升: +95.69%
- 盈亏比提升: +0.81
- 平均收益提升: +6.84%

## 代码结构

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
│   ├── paths.py                  # 路径配置
│   └── collect_external_data.py  # 外部数据收集
├── tests/
│   ├── test_strategy_scaffold.py # 原有测试
│   └── test_improved_strategy.py # 改进策略测试
├── data/
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后数据
│   └── external/                 # 外部调研数据
└── reports/                      # 研究报告
```

## 外部数据来源

- **IPO信息**: 港交所披露易 (https://www.hkexnews.hk/)
- **灰市数据**: AAStocks (https://www.aastocks.com/)
- **说明**: 数据基于公开信息整理，仅供研究参考

## 评估标准

### 评分维度

1. **数据工作 (30分)**
   - API下载并缓存: 8分
   - coverage summary: 6分
   - 处理缺失/停牌/重复: 6分
   - IPO/暗盘来源记录: 6分
   - 覆盖率说明: 4分

2. **回测正确性 (30分)**
   - 无未来函数: 8分
   - 成本滑点正确: 6分
   - entry/exit价格合理: 5分
   - trade log完整: 4分
   - 可复现: 4分
   - 停牌/缺失处理: 3分

3. **策略推理 (20分)**
   - baseline规则清楚: 5分
   - 改进假设明确: 5分
   - 对照实验合理: 5分
   - 解释收益亏损: 5分

4. **工程质量 (10分)**
   - 代码结构清楚: 3分
   - 参数配置合理: 2分
   - 测试覆盖: 3分
   - README可运行: 2分

5. **沟通 (10分)**
   - 报告清楚: 4分
   - 图表有解释: 3分
   - 限制和下一步: 3分

### 红线

- ❌ 使用未来数据
- ❌ 忽略交易成本
- ❌ 外部数据没有来源
- ❌ 缺失值填0
- ❌ 只调参数不解释假设
- ❌ 没有trade log
- ❌ 只报收益不报回撤

## 注意事项

1. **数据来源**: 外部数据基于模拟，实际使用需验证
2. **参数调优**: 避免过拟合，需要样本外验证
3. **执行风险**: 实际交易可能面临滑点、流动性等问题
4. **市场风险**: 策略可能在不同市场环境下失效

## 下一步建议

1. 扩大样本量，验证策略普适性
2. 进行更系统的参数搜索
3. 结合更多因子（行业、市值等）
4. 在模拟盘上验证策略可行性
5. 加入仓位管理、组合优化等

## 联系方式

如有问题，请参考docs/目录下的详细文档。
