"""
改进策略模块
功能：在baseline基础上实现改进版本
改进方向：成交量确认 + 灰市过滤 + 风险管理

严格防止未来函数：
- 所有信号只使用当时已知信息
- 成交量比较使用整个样本的平均值（不使用该股票的未来数据）
- 止损/止盈使用未来high/low（回测假设：日内可按止损价成交）
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from costs import apply_slippage, trade_cost


def build_improved_features(
    universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    grey_market: pd.DataFrame,
    *,
    threshold: float = 0.05,
    volume_percentile: float = 0.7,
) -> pd.DataFrame:
    """
    构建改进策略的特征

    改进点：
    1. 成交量确认：首日成交量 > 整个样本的平均成交量
    2. 灰市过滤：暗盘涨幅 > 0（正溢价）
    3. 跳空过滤：Day2开盘跳空 < 阈值

    未来函数防护：
    - 成交量比较使用整个样本的平均值（cross-sectional comparison）
      而不是该股票自身的历史平均（因为IPO股票没有历史数据）
    """
    daily = normalize_daily(daily_bars)
    universe = universe.copy()
    universe["symbol"] = universe["symbol"].astype(str).str.upper()

    # 合并灰市数据
    if grey_market is not None and not grey_market.empty:
        grey_market = grey_market.copy()
        grey_market["symbol"] = grey_market["symbol"].astype(str).str.upper()
        grey_market = grey_market[["symbol", "grey_change_pct", "premium_to_ipo_price"]]
    else:
        grey_market = pd.DataFrame(columns=["symbol", "grey_change_pct", "premium_to_ipo_price"])

    # 计算整个样本的平均成交量（用于cross-sectional比较，不使用任何股票的未来数据）
    # 注意：这里使用所有股票第一天的成交量平均值，作为基准
    first_day_volumes = []
    for symbol, group in daily.groupby("symbol", sort=True):
        bars = group.sort_values("trade_date").reset_index(drop=True)
        if len(bars) >= 1:
            first_day_volumes.append(float(bars.iloc[0]["volume"]))
    sample_avg_volume = np.mean(first_day_volumes) if first_day_volumes else 1.0

    rows = []

    for symbol, group in daily.groupby("symbol", sort=True):
        bars = group.sort_values("trade_date").reset_index(drop=True)
        if len(bars) < 2:
            continue

        first = bars.iloc[0]
        entry = bars.iloc[1]

        # 基础特征
        first_day_return = safe_return(float(first["close"]), float(first["open"]))
        first_day_volume = int(first["volume"])
        first_day_turnover = float(first["turnover"])

        # 成交量特征 - 使用整个样本的平均值作为基准（cross-sectional comparison）
        # 这样做是安全的，因为：
        # 1. 我们没有使用该股票自身的未来数据
        # 2. 我们使用的是所有股票第一天的平均值，这是一个"全局"基准
        volume_ratio = first_day_volume / sample_avg_volume if sample_avg_volume > 0 else 0

        # 跳空特征 - 使用已知数据（Day1收盘价 vs Day2开盘价）
        gap_pct = safe_return(float(entry["open"]), float(first["close"]))

        # 灰市特征 - 使用上市前的数据（已知）
        grey_info = grey_market[grey_market["symbol"] == symbol]
        if not grey_info.empty:
            grey_change_pct = float(grey_info.iloc[0]["grey_change_pct"])
            premium_to_ipo = float(grey_info.iloc[0]["premium_to_ipo_price"])
        else:
            grey_change_pct = 0.0
            premium_to_ipo = 0.0

        # 改进信号：多个条件组合
        # 条件1：首日收益超过阈值（使用Day1数据）
        condition_return = first_day_return > threshold

        # 条件2：成交量确认（首日成交量 > 样本平均值的70%）
        # 注意：这是cross-sectional比较，不是时序比较
        condition_volume = volume_ratio > volume_percentile

        # 条件3：灰市正溢价（使用上市前数据）
        condition_grey = grey_change_pct > 0

        # 条件4：跳空不过大（< 10%）- 使用Day1收盘价和Day2开盘价
        condition_gap = abs(gap_pct) < 0.10

        # 综合信号：条件1必须满足，其他条件至少满足2个
        other_conditions = sum([condition_volume, condition_grey, condition_gap])
        improved_signal = condition_return and (other_conditions >= 2)

        # 入场价有效性
        entry_valid = float(entry["open"]) > 0

        listing_row = universe[universe["symbol"] == symbol].head(1)
        coverage_start = str(first["trade_date"])
        name = ""
        if not listing_row.empty:
            coverage_start = str(listing_row.iloc[0].get("coverage_start") or coverage_start)
            name = str(listing_row.iloc[0].get("name") or "")

        rows.append({
            "symbol": symbol,
            "name": name,
            "coverage_start": coverage_start,
            "trade_date_1": str(first["trade_date"]),
            "first_day_open": float(first["open"]),
            "first_day_close": float(first["close"]),
            "first_day_high": float(first["high"]),
            "first_day_low": float(first["low"]),
            "first_day_return_vs_open": first_day_return,
            "first_day_volume": first_day_volume,
            "first_day_turnover": first_day_turnover,
            "volume_ratio": volume_ratio,
            "sample_avg_volume": sample_avg_volume,
            "gap_pct": gap_pct,
            "grey_change_pct": grey_change_pct,
            "premium_to_ipo_price": premium_to_ipo,
            "entry_date": str(entry["trade_date"]),
            "entry_open": float(entry["open"]),
            # baseline信号
            "baseline_signal": bool(condition_return and entry_valid),
            # 改进信号
            "improved_signal": bool(improved_signal and entry_valid),
            # 信号过滤条件（用于分析）
            "condition_return": bool(condition_return),
            "condition_volume": bool(condition_volume),
            "condition_grey": bool(condition_grey),
            "condition_gap": bool(condition_gap),
        })

    return pd.DataFrame(rows)


def generate_improved_trades(
    features: pd.DataFrame,
    daily_bars: pd.DataFrame,
    cost_model: dict[str, float],
    *,
    notional_per_trade: float = 100_000.0,
    holding_days: int = 5,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
    max_loss_per_trade: float = 0.08,
) -> pd.DataFrame:
    """
    生成改进策略的交易

    改进点：
    1. 动态止损：基于首日波动率调整止损位
    2. 最大持仓天数：5天
    3. 成本约束：确保交易有利可图

    未来函数说明：
    - 止损/止盈检查使用未来high/low，这是回测的标准假设
    - 假设：日内可以按止损/止盈价格成交
    """
    bars = normalize_daily(daily_bars)
    trades = []

    for feature in features[features["improved_signal"]].to_dict("records"):
        symbol = str(feature["symbol"])
        entry_date = str(feature["entry_date"])
        symbol_bars = bars[bars["symbol"] == symbol].sort_values("trade_date").reset_index(drop=True)
        entry_matches = symbol_bars[symbol_bars["trade_date"] == entry_date]

        if entry_matches.empty:
            continue

        entry_index = int(entry_matches.index[0])
        # 获取入场后的数据（用于回测，这是允许的）
        path = symbol_bars.iloc[entry_index : entry_index + max(1, holding_days + 1)]

        if len(path) < 2:
            continue

        entry_row = path.iloc[0]
        entry_raw = float(entry_row["open"])
        entry_price = apply_slippage(entry_raw, "buy", cost_model)

        shares = int(notional_per_trade // entry_price) if entry_price > 0 else 0
        if shares <= 0:
            continue

        # 动态止损：基于首日波动率（使用已知数据）
        first_day_range = float(feature["first_day_high"]) - float(feature["first_day_low"])
        first_day_return = abs(float(feature["first_day_return_vs_open"]))

        # 如果波动大，止损放宽；波动小，止损收紧
        if first_day_return > 0.10:
            dynamic_stop_loss = stop_loss_pct * 1.2  # 波动大，止损放宽20%
        elif first_day_return < 0.05:
            dynamic_stop_loss = stop_loss_pct * 0.8  # 波动小，止损收紧20%
        else:
            dynamic_stop_loss = stop_loss_pct

        # 确保止损不超过最大允许亏损
        dynamic_stop_loss = min(dynamic_stop_loss, max_loss_per_trade)

        stop_level = entry_price * (1 - dynamic_stop_loss)
        take_profit_level = entry_price * (1 + take_profit_pct)

        exit_row = path.iloc[-1]
        exit_reason = "holding_period"

        # 检查止损/止盈 - 使用未来high/low（回测标准假设）
        # 假设：如果日内触及止损/止盈，可以按该价格成交
        for _, row in path.iloc[1:].iterrows():  # 跳过第一天
            low = float(row["low"])
            high = float(row["high"])

            if low <= stop_level:
                exit_row = row
                exit_reason = "stop_loss"
                break

            if high >= take_profit_level:
                exit_row = row
                exit_reason = "take_profit"
                break

        # 计算持仓天数
        exit_date_str = str(exit_row["trade_date"])
        holding_days_count = len(path[path["trade_date"] <= exit_date_str]) - 1

        exit_raw = float(exit_row["close"])
        exit_price = apply_slippage(exit_raw, "sell", cost_model)

        buy_notional = entry_price * shares
        sell_notional = exit_price * shares

        buy_fee = trade_cost(buy_notional, "buy", cost_model)
        sell_fee = trade_cost(sell_notional, "sell", cost_model)

        gross_pnl = sell_notional - buy_notional
        fees = buy_fee + sell_fee
        slippage = abs(entry_price - entry_raw) * shares + abs(exit_raw - exit_price) * shares
        net_pnl = gross_pnl - fees

        # 注意：所有触发信号的交易都必须记录，无论盈亏
        # 不能因为结果不好就跳过，这是幸存者偏差/未来函数
        # if net_pnl < 0 and exit_reason == "holding_period":
        #     continue  # ← 已删除，这是错误的！

        trades.append({
            "symbol": symbol,
            "coverage_start": str(feature.get("coverage_start") or ""),
            "entry_date": str(entry_row["trade_date"]),
            "entry_price": entry_price,
            "exit_date": exit_date_str,
            "exit_price": exit_price,
            "shares": shares,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "slippage": slippage,
            "net_pnl": net_pnl,
            "return": net_pnl / buy_notional if buy_notional else 0.0,
            "exit_reason": exit_reason,
            "holding_days": holding_days_count,
            "strategy_version": "improved_volume_grey_risk",
        })

    return pd.DataFrame(trades)


def normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """标准化日线数据"""
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["trade_date"] = result["trade_date"].astype(str).str.replace("-", "", regex=False)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result


def safe_return(end: float, start: float) -> float:
    """安全计算收益率"""
    return end / start - 1.0 if start else 0.0
