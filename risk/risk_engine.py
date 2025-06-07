import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from utils.logger import get_hft_logger


class RiskCheckType(Enum):
    """Types of risk checks"""
    PRE_TRADE = "pre_trade"
    POST_TRADE = "post_trade"
    POSITION = "position"
    PORTFOLIO = "portfolio"
    MARKET = "market"


@dataclass
class RiskCheck:
    """Risk check result"""
    check_type: RiskCheckType
    passed: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    strategy: str


class RiskEngine:
    """Comprehensive risk management engine for HFT system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get('risk', {})
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("risk_engine")  # Get BoundLogger for regular logging
        
        # Risk limits
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk', 0.02)
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.05)
        self.stop_loss = self.risk_config.get('stop_loss', 0.02)
        self.take_profit = self.risk_config.get('take_profit', 0.04)
        self.max_open_positions = self.risk_config.get('max_open_positions', 3)
        self.max_leverage = self.risk_config.get('max_leverage', 1.0)
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.daily_trades = 0
        self.risk_breaches = []
        
        # Portfolio tracking
        self.portfolio_value = config.get('trading', {}).get('initial_balance', 10000.0)
        self.available_capital = self.portfolio_value
        self.max_position_size = self.portfolio_value * 0.2  # 20% max per position
        
        # Performance metrics
        self.risk_checks_performed = 0
        self.risk_checks_failed = 0
        
        # Circuit breaker
        self.trading_halted = False
        self.halt_reason = None
        
    def perform_pre_trade_check(self, symbol: str, side: str, quantity: float, 
                               price: float, strategy: str) -> RiskCheck:
        """Perform comprehensive pre-trade risk checks"""
        self.risk_checks_performed += 1
        
        checks = []
        overall_passed = True
        risk_level = "LOW"
        
        # 1. Position limits check
        position_check = self._check_position_limits(symbol, side, quantity, price)
        checks.append(position_check)
        if not position_check.passed:
            overall_passed = False
            risk_level = max(risk_level, position_check.risk_level, key=self._risk_level_priority)
        
        # 2. Portfolio exposure check
        exposure_check = self._check_portfolio_exposure(quantity, price)
        checks.append(exposure_check)
        if not exposure_check.passed:
            overall_passed = False
            risk_level = max(risk_level, exposure_check.risk_level, key=self._risk_level_priority)
        
        # 3. Daily loss limit check
        daily_loss_check = self._check_daily_loss_limit()
        checks.append(daily_loss_check)
        if not daily_loss_check.passed:
            overall_passed = False
            risk_level = max(risk_level, daily_loss_check.risk_level, key=self._risk_level_priority)
        
        # 4. Capital adequacy check
        capital_check = self._check_capital_adequacy(quantity, price)
        checks.append(capital_check)
        if not capital_check.passed:
            overall_passed = False
            risk_level = max(risk_level, capital_check.risk_level, key=self._risk_level_priority)
        
        # 5. Trading halt check
        halt_check = self._check_trading_halt()
        checks.append(halt_check)
        if not halt_check.passed:
            overall_passed = False
            risk_level = "CRITICAL"
        
        # Aggregate results
        if not overall_passed:
            self.risk_checks_failed += 1
        
        details = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'strategy': strategy,
            'individual_checks': [check.__dict__ for check in checks],
            'current_positions': len(self.positions),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl
        }
        
        message = f"Pre-trade check {'PASSED' if overall_passed else 'FAILED'} for {symbol}"
        if not overall_passed:
            failed_checks = [check.message for check in checks if not check.passed]
            message += f". Failed: {', '.join(failed_checks)}"
        
        risk_check = RiskCheck(
            check_type=RiskCheckType.PRE_TRADE,
            passed=overall_passed,
            risk_level=risk_level,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
        
        self.hft_logger.log_risk_check("pre_trade", overall_passed, details)
        return risk_check
    
    def perform_post_trade_check(self, symbol: str, side: str, quantity: float, 
                                price: float, trade_id: str) -> RiskCheck:
        """Perform post-trade risk checks and update positions"""
        # Update position
        self._update_position(symbol, side, quantity, price)
        
        # Update daily metrics
        self.daily_trades += 1
        
        checks = []
        overall_passed = True
        risk_level = "LOW"
        
        # 1. Position size check
        if symbol in self.positions:
            position_value = abs(self.positions[symbol].quantity * self.positions[symbol].current_price)
            if position_value > self.max_position_size:
                checks.append(RiskCheck(
                    check_type=RiskCheckType.POSITION,
                    passed=False,
                    risk_level="HIGH",
                    message=f"Position size exceeded limit: {position_value:.2f} > {self.max_position_size:.2f}",
                    details={'position_value': position_value, 'limit': self.max_position_size},
                    timestamp=datetime.now()
                ))
                overall_passed = False
                risk_level = "HIGH"
        
        # 2. Stop loss/take profit check
        sl_tp_check = self._check_stop_loss_take_profit(symbol)
        if sl_tp_check:
            checks.append(sl_tp_check)
            if not sl_tp_check.passed:
                overall_passed = False
                risk_level = max(risk_level, sl_tp_check.risk_level, key=self._risk_level_priority)
        
        details = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'trade_id': trade_id,
            'updated_position': self.positions.get(symbol).__dict__ if symbol in self.positions else None,
            'portfolio_exposure': self.total_exposure,
            'daily_trades': self.daily_trades
        }
        
        message = f"Post-trade check {'PASSED' if overall_passed else 'FAILED'} for {symbol}"
        
        risk_check = RiskCheck(
            check_type=RiskCheckType.POST_TRADE,
            passed=overall_passed,
            risk_level=risk_level,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
        
        self.hft_logger.log_risk_check("post_trade", overall_passed, details)
        return risk_check
    
    def monitor_positions(self, market_data: Dict[str, float]) -> List[RiskCheck]:
        """Continuously monitor positions for risk violations"""
        risk_checks = []
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                # Update current price
                current_price = market_data[symbol]
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.side == "BUY":
                    position.unrealized_pnl = (current_price - position.entry_price) / position.entry_price
                else:  # SELL
                    position.unrealized_pnl = (position.entry_price - current_price) / position.entry_price
                
                # Check stop loss
                if position.unrealized_pnl <= -self.stop_loss:
                    risk_check = RiskCheck(
                        check_type=RiskCheckType.POSITION,
                        passed=False,
                        risk_level="HIGH",
                        message=f"Stop loss triggered for {symbol}: {position.unrealized_pnl:.4f}",
                        details={
                            'symbol': symbol,
                            'unrealized_pnl': position.unrealized_pnl,
                            'stop_loss_limit': -self.stop_loss,
                            'action_required': 'CLOSE_POSITION'
                        },
                        timestamp=datetime.now()
                    )
                    risk_checks.append(risk_check)
                
                # Check take profit
                elif position.unrealized_pnl >= self.take_profit:
                    risk_check = RiskCheck(
                        check_type=RiskCheckType.POSITION,
                        passed=True,
                        risk_level="LOW",
                        message=f"Take profit triggered for {symbol}: {position.unrealized_pnl:.4f}",
                        details={
                            'symbol': symbol,
                            'unrealized_pnl': position.unrealized_pnl,
                            'take_profit_limit': self.take_profit,
                            'action_required': 'CLOSE_POSITION'
                        },
                        timestamp=datetime.now()
                    )
                    risk_checks.append(risk_check)
        
        # Update portfolio metrics
        self._update_portfolio_metrics()
        
        # Check portfolio-level risks
        portfolio_check = self._check_portfolio_risk()
        if portfolio_check:
            risk_checks.append(portfolio_check)
        
        return risk_checks
    
    def _check_position_limits(self, symbol: str, side: str, quantity: float, price: float) -> RiskCheck:
        """Check position limits"""
        current_positions = len(self.positions)
        position_value = quantity * price
        
        if current_positions >= self.max_open_positions:
            return RiskCheck(
                check_type=RiskCheckType.POSITION,
                passed=False,
                risk_level="MEDIUM",
                message=f"Maximum open positions exceeded: {current_positions}/{self.max_open_positions}",
                details={'current_positions': current_positions, 'limit': self.max_open_positions},
                timestamp=datetime.now()
            )
        
        if position_value > self.max_position_size:
            return RiskCheck(
                check_type=RiskCheckType.POSITION,
                passed=False,
                risk_level="HIGH",
                message=f"Position size too large: {position_value:.2f} > {self.max_position_size:.2f}",
                details={'position_value': position_value, 'limit': self.max_position_size},
                timestamp=datetime.now()
            )
        
        return RiskCheck(
            check_type=RiskCheckType.POSITION,
            passed=True,
            risk_level="LOW",
            message="Position limits check passed",
            details={'position_value': position_value, 'current_positions': current_positions},
            timestamp=datetime.now()
        )
    
    def _check_portfolio_exposure(self, quantity: float, price: float) -> RiskCheck:
        """Check portfolio exposure limits"""
        new_exposure = quantity * price
        total_new_exposure = self.total_exposure + new_exposure
        exposure_ratio = total_new_exposure / self.portfolio_value
        
        if exposure_ratio > self.max_leverage:
            return RiskCheck(
                check_type=RiskCheckType.PORTFOLIO,
                passed=False,
                risk_level="HIGH",
                message=f"Portfolio exposure too high: {exposure_ratio:.2%} > {self.max_leverage:.2%}",
                details={
                    'current_exposure': self.total_exposure,
                    'new_exposure': new_exposure,
                    'total_exposure': total_new_exposure,
                    'exposure_ratio': exposure_ratio,
                    'limit': self.max_leverage
                },
                timestamp=datetime.now()
            )
        
        return RiskCheck(
            check_type=RiskCheckType.PORTFOLIO,
            passed=True,
            risk_level="LOW",
            message="Portfolio exposure check passed",
            details={'exposure_ratio': exposure_ratio},
            timestamp=datetime.now()
        )
    
    def _check_daily_loss_limit(self) -> RiskCheck:
        """Check daily loss limits"""
        daily_loss_ratio = abs(self.daily_pnl) / self.portfolio_value
        
        if self.daily_pnl < 0 and daily_loss_ratio > self.max_daily_loss:
            return RiskCheck(
                check_type=RiskCheckType.PORTFOLIO,
                passed=False,
                risk_level="CRITICAL",
                message=f"Daily loss limit exceeded: {daily_loss_ratio:.2%} > {self.max_daily_loss:.2%}",
                details={
                    'daily_pnl': self.daily_pnl,
                    'daily_loss_ratio': daily_loss_ratio,
                    'limit': self.max_daily_loss,
                    'action_required': 'HALT_TRADING'
                },
                timestamp=datetime.now()
            )
        
        return RiskCheck(
            check_type=RiskCheckType.PORTFOLIO,
            passed=True,
            risk_level="LOW",
            message="Daily loss limit check passed",
            details={'daily_pnl': self.daily_pnl, 'daily_loss_ratio': daily_loss_ratio},
            timestamp=datetime.now()
        )
    
    def _check_capital_adequacy(self, quantity: float, price: float) -> RiskCheck:
        """Check if sufficient capital is available"""
        required_capital = quantity * price
        
        if required_capital > self.available_capital:
            return RiskCheck(
                check_type=RiskCheckType.PORTFOLIO,
                passed=False,
                risk_level="HIGH",
                message=f"Insufficient capital: {required_capital:.2f} > {self.available_capital:.2f}",
                details={
                    'required_capital': required_capital,
                    'available_capital': self.available_capital
                },
                timestamp=datetime.now()
            )
        
        return RiskCheck(
            check_type=RiskCheckType.PORTFOLIO,
            passed=True,
            risk_level="LOW",
            message="Capital adequacy check passed",
            details={'required_capital': required_capital, 'available_capital': self.available_capital},
            timestamp=datetime.now()
        )
    
    def _check_trading_halt(self) -> RiskCheck:
        """Check if trading is halted"""
        if self.trading_halted:
            return RiskCheck(
                check_type=RiskCheckType.MARKET,
                passed=False,
                risk_level="CRITICAL",
                message=f"Trading halted: {self.halt_reason}",
                details={'halt_reason': self.halt_reason},
                timestamp=datetime.now()
            )
        
        return RiskCheck(
            check_type=RiskCheckType.MARKET,
            passed=True,
            risk_level="LOW",
            message="Trading halt check passed",
            details={},
            timestamp=datetime.now()
        )
    
    def _check_stop_loss_take_profit(self, symbol: str) -> Optional[RiskCheck]:
        """Check stop loss and take profit for a position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position.unrealized_pnl <= -self.stop_loss:
            return RiskCheck(
                check_type=RiskCheckType.POSITION,
                passed=False,
                risk_level="HIGH",
                message=f"Stop loss triggered for {symbol}",
                details={
                    'unrealized_pnl': position.unrealized_pnl,
                    'stop_loss_limit': -self.stop_loss,
                    'action_required': 'CLOSE_POSITION'
                },
                timestamp=datetime.now()
            )
        
        return None
    
    def _check_portfolio_risk(self) -> Optional[RiskCheck]:
        """Check overall portfolio risk"""
        total_unrealized_pnl = sum(pos.unrealized_pnl * pos.quantity * pos.current_price 
                                  for pos in self.positions.values())
        portfolio_risk = abs(total_unrealized_pnl) / self.portfolio_value
        
        if portfolio_risk > self.max_portfolio_risk:
            return RiskCheck(
                check_type=RiskCheckType.PORTFOLIO,
                passed=False,
                risk_level="HIGH",
                message=f"Portfolio risk exceeded: {portfolio_risk:.2%} > {self.max_portfolio_risk:.2%}",
                details={
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'portfolio_risk': portfolio_risk,
                    'limit': self.max_portfolio_risk
                },
                timestamp=datetime.now()
            )
        
        return None
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update or create position"""
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            if position.side == side:
                # Same direction - add to position
                total_quantity = position.quantity + quantity
                weighted_price = ((position.quantity * position.entry_price) + 
                                (quantity * price)) / total_quantity
                position.quantity = total_quantity
                position.entry_price = weighted_price
            else:
                # Opposite direction - reduce or reverse position
                if quantity >= position.quantity:
                    # Close and potentially reverse
                    remaining_quantity = quantity - position.quantity
                    if remaining_quantity > 0:
                        position.side = side
                        position.quantity = remaining_quantity
                        position.entry_price = price
                    else:
                        del self.positions[symbol]
                else:
                    # Partial close
                    position.quantity -= quantity
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now(),
                strategy=""
            )
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        self.total_exposure = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        
        # Update available capital
        unrealized_pnl_value = sum(
            pos.unrealized_pnl * pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        self.available_capital = self.portfolio_value + unrealized_pnl_value - self.total_exposure
    
    def _risk_level_priority(self, risk_level: str) -> int:
        """Get priority value for risk level comparison"""
        priorities = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return priorities.get(risk_level, 0)
    
    def halt_trading(self, reason: str):
        """Halt all trading activities"""
        self.trading_halted = True
        self.halt_reason = reason
        self.logger.warning(f"Trading halted: {reason}")
    
    def resume_trading(self):
        """Resume trading activities"""
        self.trading_halted = False
        self.halt_reason = None
        self.logger.info("Trading resumed")
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics"""
        win_rate = (self.risk_checks_performed - self.risk_checks_failed) / max(self.risk_checks_performed, 1)
        
        return {
            'risk_checks_performed': self.risk_checks_performed,
            'risk_checks_failed': self.risk_checks_failed,
            'risk_check_success_rate': round(win_rate * 100, 2),
            'current_positions': len(self.positions),
            'total_exposure': self.total_exposure,
            'available_capital': self.available_capital,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_daily_loss': self.max_daily_loss,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'max_open_positions': self.max_open_positions
            }
        } 