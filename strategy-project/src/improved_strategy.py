"""
改进策略模块
功能：在baseline基础上实现改进版本
改进方向：成交量确认 + 灰市过滤 + 风险管理
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
    1. 成交量确认：首日成交量 > 历史分位数
    2. 灰市过滤：暗盘涨幅 > 0（正溢价）
    3. 跳空过滤：Day2开盘跳空 < 阈值
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
        
        # 成交量特征
        avg_volume = bars["volume"].mean()
        volume_ratio = first_day_volume / avg_volume if avg_volume > 0 else 0
        
        # 跳空特征
        gap_pct = safe_return(float(entry["open"]), float(first["close"]))
        
        # 灰市特征
        grey_info = grey_market[grey_market["symbol"] == symbol]
        if not grey_info.empty:
            grey_change_pct = float(grey_info.iloc[0]["grey_change_pct"])
            premium_to_ipo = float(grey_info.iloc[0]["premium_to_ipo_price"])
        else:
            grey_change_pct = 0.0
            premium_to_ipo = 0.0
        
        # 改进信号：多个条件组合
        # 条件1：首日收益超过阈值
        condition_return = first_day_return > threshold
        
        # 条件2：成交量确认（首日成交量 > 70%分位数）
        condition_volume = volume_ratio > volume_percentile
        
        # 条件3：灰市正溢价（暗盘涨幅 > 0）
        condition_grey = grey_change_pct > 0
        
        # 条件4：跳空不过大（< 10%）
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
    1. 动态止损：基于波动率调整止损位
    2. 最大持仓天数：5天
    3. 成本约束：确保交易有利可图
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
        # 获取未来数据（最多holding_days天）
        path = symbol_bars.iloc[entry_index : entry_index + max(1, holding_days + 1)]
        
        if len(path) < 2:
            continue
        
        entry_row = path.iloc[0]
        entry_raw = float(entry_row["open"])
        entry_price = apply_slippage(entry_raw, "buy", cost_model)
        
        shares = int(notional_per_trade // entry_price) if entry_price > 0 else 0
        if shares <= 0:
            continue
        
        # 动态止损：基于首日波动率
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
        
        # 检查止损/止盈
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
        
        # 成本约束：如果预期利润太低，不交易
        if net_pnl < 0 and exit_reason == "holding_period":
            continue
        
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
