#!/usr/bin/env python3
"""
Take Profit Strategy - Automatically sell positions when profit targets are met
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import get_hft_logger
from strategies.mean_reversion import Signal


class TakeProfitStrategy:
    """Take profit strategy - sells positions when profit targets are reached"""
    
    def __init__(self, config: Dict):
        self.config = config['strategies'].get('take_profit', {})
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("take_profit")
        
        # Strategy parameters - BALANCED RISK/REWARD
        self.profit_target_percent = self.config.get('profit_target_percent', 0.8)  # 0.8% profit target
        self.min_hold_time_minutes = self.config.get('min_hold_time_minutes', 2)  # Hold for at least 2 minutes
        self.max_hold_time_hours = self.config.get('max_hold_time_hours', 4)  # Force sell after 4 hours
        self.stop_loss_percent = self.config.get('stop_loss_percent', -0.5)  # 0.5% stop loss
        self.trailing_stop_percent = self.config.get('trailing_stop_percent', 0.3)  # 0.3% trailing stop
        self.position_size = self.config.get('position_size', 1.0)  # Sell 100% of position
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 1.6)  # 1.6:1 risk/reward
        
        # Enhanced risk management
        self.max_daily_loss = self.config.get('max_daily_loss', 0.01)  # 1% max daily loss
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.daily_pnl = 0
        
        # State tracking
        self.monitored_positions = {}  # Track positions we're monitoring
        self.profit_targets = {}  # Store profit targets for each position
        self.last_signals = {}  # Avoid duplicate signals
        
        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        self.profits_taken = 0
        self.stop_losses_hit = 0
        
        self.logger.info(f"Take Profit Strategy initialized with {self.profit_target_percent}% profit target")
    
    def analyze_positions(self, portfolio_summary: Dict) -> List[Signal]:
        """Analyze current positions and generate sell signals when profit targets are met"""
        signals = []
        
        try:
            positions = portfolio_summary.get('positions', [])
            current_time = datetime.now()
            
            for position in positions:
                symbol = position['symbol']
                
                # Track new positions
                if symbol not in self.monitored_positions:
                    self._add_position_to_monitor(position)
                
                # Check if we should sell this position
                signal = self._check_profit_target(position, current_time)
                if signal:
                    signals.append(signal)
                    
        except Exception as e:
            self.hft_logger.log_error("analyze_positions", e, {"positions_count": len(positions) if 'positions' in locals() else 0})
        
        return signals
    
    def _add_position_to_monitor(self, position: Dict):
        """Add a new position to monitoring"""
        symbol = position['symbol']
        
        self.monitored_positions[symbol] = {
            'entry_price': position['avg_entry_price'],
            'entry_time': datetime.now(),
            'quantity': position['quantity'],
            'cost_basis': position['cost_basis'],
            'profit_target_price': position['avg_entry_price'] * (1 + self.profit_target_percent / 100),
            'stop_loss_price': position['avg_entry_price'] * (1 + self.stop_loss_percent / 100),
            'highest_price': position['avg_entry_price'],  # Track highest price for trailing stop
            'initial_monitoring': True
        }
        
        self.logger.info(f"Monitoring new position: {symbol} @ ${position['avg_entry_price']:.4f}", 
                        profit_target=f"${self.monitored_positions[symbol]['profit_target_price']:.4f}",
                        stop_loss=f"${self.monitored_positions[symbol]['stop_loss_price']:.4f}")
    
    def _check_profit_target(self, position: Dict, current_time: datetime) -> Optional[Signal]:
        """Check if position meets profit target or stop loss criteria"""
        symbol = position['symbol']
        current_price = position['current_price']
        
        if symbol not in self.monitored_positions:
            return None
        
        monitored = self.monitored_positions[symbol]
        entry_price = monitored['entry_price']
        profit_target_price = monitored['profit_target_price']
        stop_loss_price = monitored['stop_loss_price']
        entry_time = monitored['entry_time']
        
        # Calculate current profit/loss percentage
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        time_held = (current_time - entry_time).total_seconds() / 60  # Minutes
        
        signal_type = None
        signal_reason = None
        strength = 0.8  # High confidence for mechanical signals
        
        # ENHANCED RISK MANAGEMENT CHECKS
        
        # 1. IMMEDIATE STOP LOSS (highest priority)
        if current_price <= stop_loss_price:
            signal_type = "SELL"
            signal_reason = "stop_loss_triggered"
            self.stop_losses_hit += 1
            self.consecutive_losses += 1
            strength = 1.0  # Maximum confidence for risk management
            
        # 2. TRAILING STOP (protect profits)
        elif symbol in self.monitored_positions and 'highest_price' in monitored:
            highest_price = max(monitored.get('highest_price', entry_price), current_price)
            self.monitored_positions[symbol]['highest_price'] = highest_price
            trailing_stop_price = highest_price * (1 - self.trailing_stop_percent / 100)
            
            if current_price <= trailing_stop_price and profit_percent > 0:
                signal_type = "SELL"
                signal_reason = "trailing_stop_triggered"
                self.profits_taken += 1
                self.consecutive_losses = 0  # Reset consecutive losses on profit
                strength = 0.9  # High confidence for profit protection
        
        # 3. PROFIT TARGET (balanced approach)
        elif current_price >= profit_target_price and time_held >= self.min_hold_time_minutes:
            signal_type = "SELL"
            signal_reason = "profit_target_reached"
            self.profits_taken += 1
            self.consecutive_losses = 0  # Reset consecutive losses on profit
            strength = 0.95  # Very high confidence for profit taking
            
        # 4. RISK/REWARD BASED EXIT (smart exit)
        elif profit_percent >= (abs(self.stop_loss_percent) * self.risk_reward_ratio) and time_held >= self.min_hold_time_minutes:
            signal_type = "SELL"
            signal_reason = "risk_reward_target_met"
            self.profits_taken += 1
            self.consecutive_losses = 0
            strength = 0.85  # High confidence for risk/reward exit
            
        # 5. CONSECUTIVE LOSS PROTECTION
        elif self.consecutive_losses >= self.max_consecutive_losses and profit_percent >= -abs(self.stop_loss_percent) * 0.5:
            signal_type = "SELL"
            signal_reason = "consecutive_loss_protection"
            strength = 0.75  # Moderate confidence for protection
            
        # 6. FORCE SELL AFTER MAX HOLD TIME
        elif time_held >= (self.max_hold_time_hours * 60):
            signal_type = "SELL"
            signal_reason = "max_hold_time_exceeded"
            strength = 0.7  # Medium confidence for time-based exit
        
        # Generate signal if criteria met
        if signal_type:
            # Avoid duplicate signals (REDUCED for more frequent selling)
            if symbol in self.last_signals:
                last_signal_time = self.last_signals[symbol].timestamp
                if (current_time - last_signal_time).seconds < 30:  # Only 30 seconds (was 5 minutes)
                    return None
            
            metadata = {
                'entry_price': entry_price,
                'current_price': current_price,
                'profit_percent': profit_percent,
                'profit_target_price': profit_target_price,
                'stop_loss_price': stop_loss_price,
                'time_held_minutes': time_held,
                'signal_reason': signal_reason,
                'position_quantity': position['quantity'],
                'unrealized_pnl': position['unrealized_pnl']
            }
            
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                timestamp=current_time,
                metadata=metadata
            )
            
            self.signal_count += 1
            self.last_signals[symbol] = signal
            
            # Remove from monitoring after signal
            if signal_reason == "profit_target_reached":
                del self.monitored_positions[symbol]
            
            self.hft_logger.log_signal(
                symbol, signal_type, strength, "take_profit", metadata
            )
            
            self.logger.info(f"TAKE PROFIT SIGNAL: {symbol} {signal_reason}", 
                           entry_price=f"${entry_price:.4f}",
                           current_price=f"${current_price:.4f}",
                           profit_percent=f"{profit_percent:+.2f}%",
                           time_held=f"{time_held:.1f}min")
            
            return signal
        
        return None
    
    def calculate_position_size(self, symbol: str, signal: Signal, available_capital: float) -> float:
        """Calculate position size for take profit (usually sell full position)"""
        if signal.signal_type != "SELL":
            return 0.0
        
        # Get the quantity from the signal metadata
        position_quantity = signal.metadata.get('position_quantity', 0)
        
        # Sell percentage of position (default 100%)
        sell_quantity = position_quantity * self.position_size
        
        return round(sell_quantity, 6)
    
    def get_strategy_stats(self) -> Dict:
        """Get take profit strategy performance statistics"""
        total_exits = self.profits_taken + self.stop_losses_hit
        win_rate = (self.profits_taken / max(total_exits, 1)) * 100
        
        return {
            'strategy_name': 'take_profit',
            'total_signals': self.signal_count,
            'profits_taken': self.profits_taken,
            'stop_losses_hit': self.stop_losses_hit,
            'win_rate_percentage': round(win_rate, 2),
            'monitored_positions': len(self.monitored_positions),
            'parameters': {
                'profit_target_percent': self.profit_target_percent,
                'stop_loss_percent': self.stop_loss_percent,
                'min_hold_time_minutes': self.min_hold_time_minutes,
                'max_hold_time_hours': self.max_hold_time_hours
            }
        }
    
    def update_config(self, new_config: Dict):
        """Update strategy parameters"""
        old_target = self.profit_target_percent
        
        self.profit_target_percent = new_config.get('profit_target_percent', self.profit_target_percent)
        self.stop_loss_percent = new_config.get('stop_loss_percent', self.stop_loss_percent)
        self.min_hold_time_minutes = new_config.get('min_hold_time_minutes', self.min_hold_time_minutes)
        self.max_hold_time_hours = new_config.get('max_hold_time_hours', self.max_hold_time_hours)
        
        self.logger.info(f"Take profit target updated: {old_target}% → {self.profit_target_percent}%")
        
        # Update existing monitored positions with new targets
        for symbol, monitored in self.monitored_positions.items():
            monitored['profit_target_price'] = monitored['entry_price'] * (1 + self.profit_target_percent / 100)
            monitored['stop_loss_price'] = monitored['entry_price'] * (1 + self.stop_loss_percent / 100) 