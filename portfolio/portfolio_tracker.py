#!/usr/bin/env python3
"""
Portfolio Tracker - Real-time Position and Investment Monitoring
Tracks all positions, growth, P&L, and detailed investment analytics
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import os

from utils.logger import get_hft_logger


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a specific time"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    strategy: str


@dataclass
class TradeRecord:
    """Individual trade record"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    strategy: str
    order_id: str


@dataclass
class PositionSummary:
    """Detailed position summary"""
    symbol: str
    side: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    total_commission: float
    strategy: str
    first_entry: datetime
    last_update: datetime
    trade_count: int
    daily_pnl: float
    max_profit: float
    max_loss: float
    duration_hours: float


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    invested_amount: float
    total_unrealized_pnl: float
    total_unrealized_pnl_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    total_commission: float
    position_count: int
    winning_positions: int
    losing_positions: int
    win_rate: float
    largest_winner: float
    largest_loser: float
    sharpe_ratio: float
    max_drawdown: float
    portfolio_growth: float


class PortfolioTracker:
    """Comprehensive portfolio tracking and analytics"""
    
    def __init__(self, config: Dict[str, Any], initial_balance: float = 10000.0):
        self.config = config
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("portfolio_tracker")
        
        # Portfolio state
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, PositionSummary] = {}
        self.trade_history: List[TradeRecord] = []
        self.portfolio_snapshots: deque = deque(maxlen=10000)  # Keep last 10k snapshots
        self.position_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.daily_returns: deque = deque(maxlen=252)  # 1 year of daily returns
        self.hourly_values: deque = deque(maxlen=24*30)  # 30 days of hourly values
        self.start_time = datetime.now()
        
        # Analytics
        self.performance_metrics: Dict[str, float] = {}
        self.risk_metrics: Dict[str, float] = {}
        
        # Data persistence
        self.data_dir = "data/portfolio"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("Portfolio tracker initialized", 
                        initial_balance=initial_balance)
    
    def add_trade(self, trade_id: str, symbol: str, side: str, quantity: float, 
                  price: float, commission: float = 0.0, strategy: str = "", 
                  order_id: str = ""):
        """Record a new trade and update positions"""
        
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            strategy=strategy,
            order_id=order_id
        )
        
        self.trade_history.append(trade)
        self._update_position(trade)
        self._update_cash_balance(trade)
        
        self.logger.info("Trade recorded", 
                        symbol=symbol, side=side, quantity=quantity, 
                        price=price, strategy=strategy)
    
    def _update_position(self, trade: TradeRecord):
        """Update position based on trade"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = PositionSummary(
                symbol=symbol,
                side=trade.side,
                quantity=0.0,
                avg_entry_price=0.0,
                current_price=trade.price,
                market_value=0.0,
                cost_basis=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_percent=0.0,
                total_commission=0.0,
                strategy=trade.strategy,
                first_entry=trade.timestamp,
                last_update=trade.timestamp,
                trade_count=0,
                daily_pnl=0.0,
                max_profit=0.0,
                max_loss=0.0,
                duration_hours=0.0
            )
        
        position = self.positions[symbol]
        
        if trade.side == position.side or position.quantity == 0:
            # Adding to position or new position
            total_cost = (position.avg_entry_price * position.quantity) + (trade.price * trade.quantity)
            total_quantity = position.quantity + trade.quantity
            
            if total_quantity > 0:
                position.avg_entry_price = total_cost / total_quantity
                position.quantity = total_quantity
            else:
                # Position closed
                del self.positions[symbol]
                return
                
        else:
            # Reducing or closing position
            if trade.quantity >= position.quantity:
                # Position closed or reversed
                remaining_quantity = trade.quantity - position.quantity
                if remaining_quantity > 0:
                    # Position reversed
                    position.side = trade.side
                    position.quantity = remaining_quantity
                    position.avg_entry_price = trade.price
                    position.first_entry = trade.timestamp
                else:
                    # Position closed
                    del self.positions[symbol]
                    return
            else:
                # Partial close
                position.quantity -= trade.quantity
        
        # Update position metrics
        position.current_price = trade.price
        position.market_value = position.quantity * position.current_price
        position.cost_basis = position.quantity * position.avg_entry_price
        position.total_commission += trade.commission
        position.last_update = trade.timestamp
        position.trade_count += 1
        position.duration_hours = (trade.timestamp - position.first_entry).total_seconds() / 3600
        
        # Calculate P&L
        if position.side == "BUY":
            position.unrealized_pnl = position.market_value - position.cost_basis
        else:  # SELL
            position.unrealized_pnl = position.cost_basis - position.market_value
        
        position.unrealized_pnl_percent = (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0
        
        # Update max profit/loss
        position.max_profit = max(position.max_profit, position.unrealized_pnl)
        position.max_loss = min(position.max_loss, position.unrealized_pnl)
    
    def _update_cash_balance(self, trade: TradeRecord):
        """Update cash balance based on trade"""
        trade_value = trade.quantity * trade.price
        
        if trade.side == "BUY":
            self.cash_balance -= (trade_value + trade.commission)
        else:  # SELL
            self.cash_balance += (trade_value - trade.commission)
    
    def update_market_prices(self, market_data: Dict[str, float]):
        """Update current market prices for all positions"""
        for symbol, price in market_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                old_price = position.current_price
                position.current_price = price
                position.market_value = position.quantity * price
                
                # Recalculate P&L
                if position.side == "BUY":
                    position.unrealized_pnl = position.market_value - position.cost_basis
                else:  # SELL
                    position.unrealized_pnl = position.cost_basis - position.market_value
                
                position.unrealized_pnl_percent = (position.unrealized_pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0
                position.last_update = datetime.now()
                
                # Update max profit/loss
                position.max_profit = max(position.max_profit, position.unrealized_pnl)
                position.max_loss = min(position.max_loss, position.unrealized_pnl)
                
                # Daily P&L (simplified - would need yesterday's closing price)
                price_change = price - old_price
                if position.side == "BUY":
                    position.daily_pnl = price_change * position.quantity
                else:
                    position.daily_pnl = -price_change * position.quantity
                
                # Create position snapshot
                snapshot = PositionSnapshot(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    current_price=price,
                    market_value=position.market_value,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_percent=position.unrealized_pnl_percent,
                    strategy=position.strategy
                )
                
                self.position_snapshots[symbol].append(snapshot)
        
        # Update portfolio-level metrics
        self._calculate_portfolio_metrics()
    
    def _calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        now = datetime.now()
        
        # Basic calculations
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_cost_basis = sum(pos.cost_basis for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_commission = sum(pos.total_commission for pos in self.positions.values())
        total_portfolio_value = self.cash_balance + total_market_value
        
        # Performance calculations
        portfolio_growth = ((total_portfolio_value - self.initial_balance) / self.initial_balance) * 100
        total_unrealized_pnl_percent = (total_unrealized_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0
        
        # Position statistics
        winning_positions = sum(1 for pos in self.positions.values() if pos.unrealized_pnl > 0)
        losing_positions = sum(1 for pos in self.positions.values() if pos.unrealized_pnl < 0)
        win_rate = (winning_positions / len(self.positions)) * 100 if self.positions else 0
        
        # Largest winner/loser
        largest_winner = max((pos.unrealized_pnl for pos in self.positions.values()), default=0)
        largest_loser = min((pos.unrealized_pnl for pos in self.positions.values()), default=0)
        
        # Daily P&L
        daily_pnl = sum(pos.daily_pnl for pos in self.positions.values())
        daily_pnl_percent = (daily_pnl / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        
        # Risk metrics (simplified)
        returns = [snapshot.total_value / self.initial_balance - 1 
                  for snapshot in list(self.portfolio_snapshots)[-30:]]  # Last 30 snapshots
        
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Calculate max drawdown
            peak = returns[0]
            max_dd = 0
            for ret in returns:
                if ret > peak:
                    peak = ret
                dd = (peak - ret) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            max_drawdown = max_dd * 100
        
        metrics = PortfolioMetrics(
            timestamp=now,
            total_value=total_portfolio_value,
            cash_balance=self.cash_balance,
            invested_amount=total_cost_basis,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_percent=total_unrealized_pnl_percent,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            total_commission=total_commission,
            position_count=len(self.positions),
            winning_positions=winning_positions,
            losing_positions=losing_positions,
            win_rate=win_rate,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            portfolio_growth=portfolio_growth
        )
        
        # Store snapshot
        self.portfolio_snapshots.append(metrics)
        
        return metrics
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        metrics = self._calculate_portfolio_metrics()
        
        return {
            'overview': asdict(metrics),
            'positions': [asdict(pos) for pos in self.positions.values()],
            'recent_trades': [asdict(trade) for trade in self.trade_history[-10:]],
            'performance': self._get_performance_analysis(),
            'risk_analysis': self._get_risk_analysis()
        }
    
    def _get_performance_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        if not self.portfolio_snapshots:
            return {}
        
        snapshots = list(self.portfolio_snapshots)
        
        # Time-based returns
        if len(snapshots) >= 2:
            hour_return = ((snapshots[-1].total_value / snapshots[-2].total_value) - 1) * 100 if len(snapshots) >= 2 else 0
            day_return = ((snapshots[-1].total_value / snapshots[max(0, len(snapshots)-24)].total_value) - 1) * 100 if len(snapshots) >= 24 else 0
            week_return = ((snapshots[-1].total_value / snapshots[max(0, len(snapshots)-168)].total_value) - 1) * 100 if len(snapshots) >= 168 else 0
        else:
            hour_return = day_return = week_return = 0
        
        # Strategy performance
        strategy_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0})
        for trade in self.trade_history:
            strategy_performance[trade.strategy]['trades'] += 1
        
        for position in self.positions.values():
            strategy_performance[position.strategy]['pnl'] += position.unrealized_pnl
        
        return {
            'time_returns': {
                'hour': hour_return,
                'day': day_return,
                'week': week_return,
                'total': ((snapshots[-1].total_value / self.initial_balance) - 1) * 100 if snapshots else 0
            },
            'strategy_breakdown': dict(strategy_performance),
            'trade_statistics': {
                'total_trades': len(self.trade_history),
                'avg_trade_size': np.mean([trade.quantity * trade.price for trade in self.trade_history]) if self.trade_history else 0,
                'avg_commission_per_trade': np.mean([trade.commission for trade in self.trade_history]) if self.trade_history else 0
            }
        }
    
    def _get_risk_analysis(self) -> Dict[str, Any]:
        """Get risk analysis metrics"""
        if not self.positions:
            return {}
        
        # Position concentration
        total_value = sum(pos.market_value for pos in self.positions.values())
        largest_position_pct = max(pos.market_value / total_value for pos in self.positions.values()) * 100 if total_value > 0 else 0
        
        # VaR calculation (simplified)
        position_returns = []
        for symbol, snapshots in self.position_snapshots.items():
            if len(snapshots) >= 2:
                returns = [(snapshots[i].unrealized_pnl_percent - snapshots[i-1].unrealized_pnl_percent) 
                          for i in range(1, len(snapshots))]
                position_returns.extend(returns)
        
        var_95 = np.percentile(position_returns, 5) if position_returns else 0
        
        return {
            'concentration': {
                'largest_position_percent': largest_position_pct,
                'position_count': len(self.positions)
            },
            'value_at_risk': {
                'var_95_percent': var_95,
                'estimated_loss_95': (var_95 / 100) * total_value if total_value > 0 else 0
            },
            'exposure': {
                'total_exposure': total_value,
                'cash_ratio': (self.cash_balance / (self.cash_balance + total_value)) * 100 if (self.cash_balance + total_value) > 0 else 0
            }
        }
    
    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        snapshots = list(self.position_snapshots[symbol])
        
        # Price history
        price_history = [{'timestamp': s.timestamp, 'price': s.current_price, 'pnl': s.unrealized_pnl} 
                        for s in snapshots[-100:]]  # Last 100 snapshots
        
        # Related trades
        related_trades = [asdict(trade) for trade in self.trade_history 
                         if trade.symbol == symbol]
        
        return {
            'position': asdict(position),
            'price_history': price_history,
            'trades': related_trades,
            'statistics': {
                'volatility': np.std([s.unrealized_pnl_percent for s in snapshots[-50:]]) if len(snapshots) >= 2 else 0,
                'avg_daily_change': np.mean([s.unrealized_pnl_percent for s in snapshots[-24:]]) if len(snapshots) >= 2 else 0,
                'time_in_position': position.duration_hours
            }
        }
    
    def export_data(self, format: str = 'json') -> str:
        """Export portfolio data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = self.get_portfolio_summary()
        
        if format.lower() == 'json':
            filename = f"{self.data_dir}/portfolio_export_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            filename = f"{self.data_dir}/portfolio_export_{timestamp}.csv"
            # Export positions to CSV
            positions_df = pd.DataFrame([asdict(pos) for pos in self.positions.values()])
            positions_df.to_csv(filename, index=False)
        
        self.logger.info(f"Portfolio data exported to {filename}")
        return filename
    
    def save_snapshot(self):
        """Save current portfolio snapshot to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_file = f"{self.data_dir}/snapshot_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'positions': [asdict(pos) for pos in self.positions.values()],
            'cash_balance': self.cash_balance,
            'trade_history': [asdict(trade) for trade in self.trade_history[-100:]],  # Last 100 trades
            'portfolio_metrics': asdict(self._calculate_portfolio_metrics())
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return snapshot_file 