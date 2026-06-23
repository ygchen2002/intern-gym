"""
改进策略单元测试
功能：测试改进策略的各个模块
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_features import build_daily_ipo_features
from costs import load_cost_model, trade_cost, apply_slippage
from metrics import calculate_metrics
from strategy import generate_baseline_trades
from improved_strategy import build_improved_features, generate_improved_trades


class TestCosts:
    """成本计算测试"""
    
    def test_trade_cost_buy(self):
        """测试买入成本计算"""
        cost_model = {
            "buy_cost_bps": 12.0,
            "sell_cost_bps": 22.0,
            "slippage_bps": 10.0,
            "min_fee": 5.0,
        }
        # 买入100万，成本 = 100万 * 12/10000 = 1200
        cost = trade_cost(1_000_000, "buy", cost_model)
        assert cost == pytest.approx(1200.0, rel=1e-2)
    
    def test_trade_cost_sell(self):
        """测试卖出成本计算"""
        cost_model = {
            "buy_cost_bps": 12.0,
            "sell_cost_bps": 22.0,
            "slippage_bps": 10.0,
            "min_fee": 5.0,
        }
        # 卖出100万，成本 = 100万 * 22/10000 = 2200
        cost = trade_cost(1_000_000, "sell", cost_model)
        assert cost == pytest.approx(2200.0, rel=1e-2)
    
    def test_trade_cost_min_fee(self):
        """测试最低费用"""
        cost_model = {
            "buy_cost_bps": 12.0,
            "sell_cost_bps": 22.0,
            "slippage_bps": 10.0,
            "min_fee": 5.0,
        }
        # 小额交易，成本应不低于最低费用
        cost = trade_cost(100, "buy", cost_model)  # 100 * 12/10000 = 0.12 < 5
        assert cost == 5.0
    
    def test_apply_slippage_buy(self):
        """测试买入滑点"""
        cost_model = {"slippage_bps": 10.0}
        # 买入滑点向上
        price = apply_slippage(100.0, "buy", cost_model)
        assert price == pytest.approx(100.1, rel=1e-2)  # 100 * (1 + 10/10000)
    
    def test_apply_slippage_sell(self):
        """测试卖出滑点"""
        cost_model = {"slippage_bps": 10.0}
        # 卖出滑点向下
        price = apply_slippage(100.0, "sell", cost_model)
        assert price == pytest.approx(99.9, rel=1e-2)  # 100 * (1 - 10/10000)


class TestMetrics:
    """指标计算测试"""
    
    def test_empty_trades(self):
        """测试空交易记录"""
        trades = pd.DataFrame()
        metrics = calculate_metrics(trades)
        assert metrics["trade_count"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["total_return"] == 0.0
    
    def test_all_wins(self):
        """测试全部盈利"""
        trades = pd.DataFrame({
            "return": [0.1, 0.2, 0.15],
            "net_pnl": [1000, 2000, 1500],
            "holding_days": [3, 5, 4],
            "entry_price": [100, 100, 100],
            "shares": [1000, 1000, 1000],
        })
        metrics = calculate_metrics(trades)
        assert metrics["trade_count"] == 3
        assert metrics["win_rate"] == 1.0
        assert metrics["average_return"] == pytest.approx(0.15, rel=1e-2)
    
    def test_all_losses(self):
        """测试全部亏损"""
        trades = pd.DataFrame({
            "return": [-0.1, -0.05, -0.08],
            "net_pnl": [-1000, -500, -800],
            "holding_days": [3, 5, 4],
            "entry_price": [100, 100, 100],
            "shares": [1000, 1000, 1000],
        })
        metrics = calculate_metrics(trades)
        assert metrics["trade_count"] == 3
        assert metrics["win_rate"] == 0.0
        assert metrics["total_return"] == pytest.approx(-0.23, rel=1e-2)


class TestBaselineStrategy:
    """Baseline策略测试"""
    
    @pytest.fixture
    def sample_data(self):
        """加载样本数据"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        cost_model = load_cost_model(REPO_ROOT / "research-data" / "cost_model.json")
        return universe, daily_bars, cost_model
    
    def test_baseline_features(self, sample_data):
        """测试baseline特征构建"""
        universe, daily_bars, _ = sample_data
        features = build_daily_ipo_features(universe, daily_bars, threshold=-1.0)
        
        assert not features.empty
        assert "symbol" in features.columns
        assert "first_day_return_vs_open" in features.columns
        assert "baseline_signal" in features.columns
        assert features["baseline_signal"].any()
    
    def test_baseline_trades(self, sample_data):
        """测试baseline交易生成"""
        universe, daily_bars, cost_model = sample_data
        features = build_daily_ipo_features(universe, daily_bars, threshold=-1.0)
        trades = generate_baseline_trades(features, daily_bars, cost_model)
        
        assert not trades.empty
        assert "symbol" in trades.columns
        assert "entry_date" in trades.columns
        assert "exit_date" in trades.columns
        assert "net_pnl" in trades.columns
        assert "strategy_version" in trades.columns
        assert trades["strategy_version"].iloc[0] == "baseline_first_day_momentum_daily"
    
    def test_baseline_metrics(self, sample_data):
        """测试baseline指标计算"""
        universe, daily_bars, cost_model = sample_data
        features = build_daily_ipo_features(universe, daily_bars, threshold=-1.0)
        trades = generate_baseline_trades(features, daily_bars, cost_model)
        metrics = calculate_metrics(trades)
        
        assert "trade_count" in metrics
        assert "win_rate" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics


class TestImprovedStrategy:
    """改进策略测试"""
    
    @pytest.fixture
    def sample_data(self):
        """加载样本数据"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        cost_model = load_cost_model(REPO_ROOT / "research-data" / "cost_model.json")
        return universe, daily_bars, cost_model
    
    def test_improved_features(self, sample_data):
        """测试改进特征构建"""
        universe, daily_bars, _ = sample_data
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)
        
        assert not features.empty
        assert "improved_signal" in features.columns
        assert "volume_ratio" in features.columns
        assert "grey_change_pct" in features.columns
        assert "gap_pct" in features.columns
    
    def test_improved_trades(self, sample_data):
        """测试改进策略交易生成"""
        universe, daily_bars, cost_model = sample_data
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)
        trades = generate_improved_trades(features, daily_bars, cost_model)
        
        assert not trades.empty
        assert "strategy_version" in trades.columns
        assert trades["strategy_version"].iloc[0] == "improved_volume_grey_risk"
    
    def test_improved_metrics(self, sample_data):
        """测试改进策略指标"""
        universe, daily_bars, cost_model = sample_data
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)
        trades = generate_improved_trades(features, daily_bars, cost_model)
        metrics = calculate_metrics(trades)
        
        assert metrics["trade_count"] > 0
        assert 0 <= metrics["win_rate"] <= 1
    
    def test_signal_conditions(self, sample_data):
        """测试信号条件"""
        universe, daily_bars, _ = sample_data
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)
        
        # 检查条件列存在
        assert "condition_return" in features.columns
        assert "condition_volume" in features.columns
        assert "condition_grey" in features.columns
        assert "condition_gap" in features.columns
        
        # 检查改进信号逻辑
        improved = features[features["improved_signal"]]
        for _, row in improved.iterrows():
            # 改进信号必须满足首日收益条件
            assert row["condition_return"]
            # 至少满足2个其他条件
            other_conditions = sum([
                row["condition_volume"],
                row["condition_grey"],
                row["condition_gap"],
            ])
            assert other_conditions >= 2


class TestNoLookahead:
    """防止未来函数测试（严格检查）"""

    def test_no_future_data_in_baseline(self):
        """测试baseline没有使用未来数据"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        features = build_daily_ipo_features(universe, daily_bars, threshold=-1.0)

        for _, feature in features.iterrows():
            # 入场日期必须在第一天之后
            assert feature["entry_date"] > feature["trade_date_1"]

    def test_no_future_data_in_improved(self):
        """测试改进策略没有使用未来数据"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)

        for _, feature in features.iterrows():
            # 入场日期必须在第一天之后
            assert feature["entry_date"] > feature["trade_date_1"]

    def test_volume_ratio_uses_cross_sectional_average(self):
        """测试volume_ratio使用的是cross-sectional平均值（不使用未来数据）"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)

        # 检查sample_avg_volume列存在且所有值相同（cross-sectional比较）
        assert "sample_avg_volume" in features.columns
        sample_avg_volumes = features["sample_avg_volume"].unique()
        # 所有行应该使用相同的sample_avg_volume（整个样本的平均值）
        assert len(sample_avg_volumes) == 1

        # 验证volume_ratio = first_day_volume / sample_avg_volume
        for _, feature in features.iterrows():
            expected_ratio = feature["first_day_volume"] / feature["sample_avg_volume"]
            assert feature["volume_ratio"] == pytest.approx(expected_ratio, rel=1e-6)

    def test_volume_ratio_not_using_self_future_data(self):
        """测试volume_ratio不使用该股票自身的未来数据"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        daily = daily_bars.copy()
        daily["symbol"] = daily["symbol"].astype(str).str.upper()
        daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce").fillna(0)

        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)

        for _, feature in features.iterrows():
            symbol = feature["symbol"]
            first_day_volume = feature["first_day_volume"]

            # 获取该股票所有天的成交量
            stock_volumes = daily[daily["symbol"] == symbol]["volume"].values
            # volume_ratio不应该等于 first_day_volume / stock_volumes.mean()
            # 因为那是使用了未来数据
            self_avg = stock_volumes.mean()
            if self_avg > 0:
                wrong_ratio = first_day_volume / self_avg
                # volume_ratio应该不等于使用自身未来数据计算的值
                assert feature["volume_ratio"] != pytest.approx(wrong_ratio, rel=1e-6), \
                    f"volume_ratio={feature['volume_ratio']} 等于使用自身未来数据计算的值 {wrong_ratio}"

    def test_entry_date_before_exit_date(self):
        """测试所有交易的入场日期在出场日期之前"""
        universe = pd.read_parquet(REPO_ROOT / "research-data" / "ipo_universe.parquet")
        daily_bars = pd.read_parquet(REPO_ROOT / "research-data" / "daily_bars.parquet")
        cost_model = load_cost_model(REPO_ROOT / "research-data" / "cost_model.json")

        features = build_improved_features(universe, daily_bars, None, threshold=-1.0)
        trades = generate_improved_trades(features, daily_bars, cost_model)

        for _, trade in trades.iterrows():
            assert trade["entry_date"] < trade["exit_date"], \
                f"交易 {trade['symbol']}: 入场日期 {trade['entry_date']} >= 出场日期 {trade['exit_date']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
