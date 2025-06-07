#!/usr/bin/env python3
"""
Options Trading Strategy for HFT System
Implements various options strategies with Greeks calculation and risk management
"""

import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import norm
import math

# Import from parent modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_hft_logger


@dataclass
class OptionContract:
    """Options contract data structure"""
    symbol: str                 # Underlying symbol (e.g., 'BTC-29DEC23-45000-C')
    underlying: str            # Underlying asset (e.g., 'BTCUSDT')
    option_type: str           # 'CALL' or 'PUT'
    strike_price: float        # Strike price
    expiry_date: datetime      # Expiration date
    premium: float             # Current premium/price
    bid: float                 # Bid price
    ask: float                 # Ask price
    volume: int                # Trading volume
    open_interest: int         # Open interest
    implied_volatility: float  # Implied volatility
    
    # Greeks
    delta: float = 0.0         # Price sensitivity
    gamma: float = 0.0         # Delta sensitivity
    theta: float = 0.0         # Time decay
    vega: float = 0.0          # Volatility sensitivity
    rho: float = 0.0           # Interest rate sensitivity


@dataclass
class OptionsPosition:
    """Options position tracking"""
    contract: OptionContract
    side: str                  # 'BUY' or 'SELL'
    quantity: int              # Number of contracts
    entry_price: float         # Entry premium
    current_price: float       # Current premium
    entry_time: datetime       # Entry timestamp
    pnl: float = 0.0          # Current P&L
    pnl_percent: float = 0.0   # P&L percentage
    days_to_expiry: int = 0    # Days until expiration
    
    # Position Greeks (quantity-adjusted)
    position_delta: float = 0.0
    position_gamma: float = 0.0
    position_theta: float = 0.0
    position_vega: float = 0.0


@dataclass
class OptionsSignal:
    """Options trading signal"""
    underlying: str
    strategy: str              # Strategy name
    signal_type: str           # 'BUY_CALL', 'SELL_CALL', 'BUY_PUT', 'SELL_PUT', 'CLOSE'
    contract: OptionContract
    quantity: int
    max_risk: float           # Maximum risk for the trade
    expected_profit: float    # Expected profit
    probability: float        # Success probability
    rationale: str           # Trade rationale
    timestamp: datetime
    metadata: Dict[str, Any]


class OptionsGreeksCalculator:
    """Black-Scholes options pricing and Greeks calculator"""
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        try:
            if T <= 0:
                # Option has expired
                if option_type.upper() == 'CALL':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.upper() == 'CALL':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0.01)  # Minimum price to avoid negative
            
        except Exception:
            return 0.01
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate all option Greeks"""
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.upper() == 'CALL':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type.upper() == 'CALL':
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type.upper() == 'CALL':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}


class OptionsStrategy:
    """Advanced options trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_hft_logger().get_logger("options_strategy")
        
        # Strategy parameters
        self.risk_free_rate = config.get('options', {}).get('risk_free_rate', 0.05)
        self.max_days_to_expiry = config.get('options', {}).get('max_days_to_expiry', 45)
        self.min_days_to_expiry = config.get('options', {}).get('min_days_to_expiry', 7)
        
        # Portfolio limits
        self.max_position_size = config.get('options', {}).get('max_position_size', 10)
        self.max_delta_exposure = config.get('options', {}).get('max_delta_exposure', 5.0)
        
        # Current positions and portfolio Greeks
        self.positions: Dict[str, OptionsPosition] = {}
        self.portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        self.greeks_calc = OptionsGreeksCalculator()
        
        self.logger.info("Options trading strategy initialized")
    
    def generate_mock_options_chain(self, underlying: str, spot_price: float) -> List[OptionContract]:
        """Generate mock options chain for demo purposes"""
        contracts = []
        
        # Generate expiry dates (weekly options)
        expiry_dates = [
            datetime.now() + timedelta(days=7),   # 1 week
            datetime.now() + timedelta(days=14),  # 2 weeks
            datetime.now() + timedelta(days=21),  # 3 weeks
        ]
        
        for expiry in expiry_dates:
            # Generate strike prices around current spot price
            strikes = [
                spot_price * 0.95,  # OTM Put
                spot_price * 0.98,  # Near ATM Put
                spot_price * 1.00,  # ATM
                spot_price * 1.02,  # Near ATM Call
                spot_price * 1.05,  # OTM Call
            ]
            
            for strike in strikes:
                # Generate calls and puts
                for option_type in ['CALL', 'PUT']:
                    # Mock pricing
                    days_to_expiry = (expiry - datetime.now()).days
                    time_value = max(0.01, days_to_expiry / 365.0)
                    
                    if option_type == 'CALL':
                        intrinsic = max(0, spot_price - strike)
                    else:
                        intrinsic = max(0, strike - spot_price)
                    
                    # Mock premium calculation
                    iv = 0.3 + np.random.normal(0, 0.05)  # Base 30% IV
                    iv = max(0.1, min(1.0, iv))
                    
                    premium = intrinsic + (time_value * iv * spot_price * 0.05)
                    premium = max(0.01, premium)
                    
                    symbol = f"{underlying}-{expiry.strftime('%d%b%y').upper()}-{int(strike)}-{option_type[0]}"
                    
                    contract = OptionContract(
                        symbol=symbol,
                        underlying=underlying,
                        option_type=option_type,
                        strike_price=strike,
                        expiry_date=expiry,
                        premium=premium,
                        bid=premium * 0.95,
                        ask=premium * 1.05,
                        volume=int(np.random.exponential(100)),
                        open_interest=int(np.random.exponential(500)),
                        implied_volatility=iv
                    )
                    contracts.append(contract)
        
        return contracts
    
    def analyze_options_chain(self, underlying: str, spot_price: float) -> List[OptionsSignal]:
        """Analyze options chain and generate trading signals"""
        signals = []
        
        try:
            # Generate mock options chain
            options_chain = self.generate_mock_options_chain(underlying, spot_price)
            
            # Update Greeks for all contracts
            updated_contracts = self._update_options_greeks(options_chain, spot_price)
            
            # Generate signals for different strategies
            signals.extend(self._momentum_strategy(underlying, spot_price, updated_contracts))
            signals.extend(self._volatility_strategy(underlying, spot_price, updated_contracts))
            
            # Filter signals based on risk management
            filtered_signals = self._filter_signals_by_risk(signals)
            
            self.logger.info(f"Generated {len(filtered_signals)} options signals for {underlying}")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing options chain for {underlying}: {e}")
            return []
    
    def _momentum_strategy(self, underlying: str, spot_price: float, 
                          contracts: List[OptionContract]) -> List[OptionsSignal]:
        """Momentum breakout options strategy"""
        signals = []
        
        # Buy ATM calls for upward momentum
        for contract in contracts:
            if (contract.option_type == 'CALL' and 
                abs(contract.strike_price - spot_price) <= spot_price * 0.02 and
                contract.days_to_expiry >= self.min_days_to_expiry and
                contract.delta > 0.4):
                
                signal = OptionsSignal(
                    underlying=underlying,
                    strategy='momentum_breakout',
                    signal_type='BUY_CALL',
                    contract=contract,
                    quantity=min(5, self.max_position_size),
                    max_risk=contract.premium * 5,
                    expected_profit=contract.premium * 10,
                    probability=0.4,
                    rationale=f'Momentum play - buying {contract.strike_price} call',
                    timestamp=datetime.now(),
                    metadata={'delta': contract.delta, 'gamma': contract.gamma}
                )
                signals.append(signal)
                break  # Limit to one signal per strategy
        
        return signals
    
    def _volatility_strategy(self, underlying: str, spot_price: float, 
                           contracts: List[OptionContract]) -> List[OptionsSignal]:
        """Volatility trading strategy"""
        signals = []
        
        # Find ATM options for straddle
        atm_calls = [c for c in contracts if c.option_type == 'CALL' and 
                    abs(c.strike_price - spot_price) <= spot_price * 0.02]
        
        if atm_calls:
            call = min(atm_calls, key=lambda x: abs(x.strike_price - spot_price))
            
            if call.implied_volatility < 0.4:  # Low IV opportunity
                signal = OptionsSignal(
                    underlying=underlying,
                    strategy='volatility_expansion',
                    signal_type='BUY_CALL',
                    contract=call,
                    quantity=2,
                    max_risk=call.premium * 2,
                    expected_profit=call.premium * 6,
                    probability=0.5,
                    rationale=f'Low IV opportunity - expecting volatility expansion',
                    timestamp=datetime.now(),
                    metadata={'iv': call.implied_volatility, 'vega': call.vega}
                )
                signals.append(signal)
        
        return signals
    
    def _filter_signals_by_risk(self, signals: List[OptionsSignal]) -> List[OptionsSignal]:
        """Filter signals based on risk management rules"""
        filtered = []
        
        for signal in signals:
            # Skip if would exceed position limits
            if len(self.positions) >= 5:  # Max 5 options positions for demo
                continue
            
            # Skip if premium is too expensive
            if signal.contract.premium > signal.contract.strike_price * 0.05:
                continue
            
            # Skip if time to expiry is too short
            if signal.contract.days_to_expiry < 3:
                continue
            
            filtered.append(signal)
        
        return filtered[:2]  # Limit to 2 signals per analysis
    
    def _update_options_greeks(self, contracts: List[OptionContract], 
                             spot_price: float) -> List[OptionContract]:
        """Update Greeks for all option contracts"""
        updated = []
        
        for contract in contracts:
            # Calculate time to expiry
            time_to_expiry = (contract.expiry_date - datetime.now()).days / 365.0
            contract.days_to_expiry = max(0, (contract.expiry_date - datetime.now()).days)
            
            # Calculate Greeks
            greeks = self.greeks_calc.calculate_greeks(
                S=spot_price,
                K=contract.strike_price,
                T=time_to_expiry,
                r=self.risk_free_rate,
                sigma=contract.implied_volatility,
                option_type=contract.option_type
            )
            
            contract.delta = greeks['delta']
            contract.gamma = greeks['gamma']
            contract.theta = greeks['theta']
            contract.vega = greeks['vega']
            contract.rho = greeks['rho']
            
            updated.append(contract)
        
        return updated
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get options portfolio summary"""
        total_positions = len(self.positions)
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        
        return {
            'total_positions': total_positions,
            'total_pnl': total_pnl,
            'portfolio_greeks': self.portfolio_greeks,
            'risk_metrics': {
                'max_delta_capacity': self.max_delta_exposure - abs(self.portfolio_greeks['delta']),
                'daily_theta_decay': self.portfolio_greeks['theta'],
                'volatility_exposure': self.portfolio_greeks['vega']
            }
        }
