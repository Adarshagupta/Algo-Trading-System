import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
from utils.logger import get_hft_logger
from strategies.mean_reversion import Signal


class MomentumStrategy:
    """Momentum trading strategy using trend following and breakout detection"""
    
    def __init__(self, config: Dict):
        self.config = config['strategies']['momentum']
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("momentum")  # Get BoundLogger for regular logging
        
        # Strategy parameters
        self.fast_ma = self.config.get('fast_ma', 5)
        self.slow_ma = self.config.get('slow_ma', 20)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.position_size = self.config.get('position_size', 0.15)
        
        # State tracking
        self.positions = {}
        self.last_signals = {}
        self.trend_history = {}  # Track trend direction
        self.breakout_levels = {}  # Support/Resistance levels
        
        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        
    def analyze_market_data(self, symbol: str, ohlcv_data: pd.DataFrame) -> Optional[Signal]:
        """Analyze market data and generate momentum-based trading signals"""
        try:
            if len(ohlcv_data) < max(self.slow_ma, self.rsi_period):
                return None
            
            close_prices = ohlcv_data['close'].values
            high_prices = ohlcv_data['high'].values
            low_prices = ohlcv_data['low'].values
            volume = ohlcv_data['volume'].values
            
            current_price = close_prices[-1]
            
            # Calculate moving averages
            fast_ma = talib.SMA(close_prices, timeperiod=self.fast_ma)
            slow_ma = talib.SMA(close_prices, timeperiod=self.slow_ma)
            
            # Calculate RSI
            rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            
            # Calculate Bollinger Bands for volatility
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            
            # Calculate volume indicators
            volume_sma = talib.SMA(volume, timeperiod=20)
            volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
            
            # Detect support/resistance levels
            support, resistance = self._detect_support_resistance(high_prices, low_prices)
            
            # Generate signal
            signal = self._generate_momentum_signal(
                symbol, current_price, fast_ma[-1], slow_ma[-1], 
                rsi[-1], macd[-1], macd_signal[-1], macd_hist[-1],
                volume_ratio, support, resistance, bb_upper[-1], bb_lower[-1]
            )
            
            if signal:
                self.signal_count += 1
                self.hft_logger.log_signal(
                    symbol, signal.signal_type, signal.strength,
                    "momentum", signal.metadata
                )
            
            return signal
            
        except Exception as e:
            self.hft_logger.log_error("analyze_market_data", e, {"symbol": symbol})
            return None
    
    def _generate_momentum_signal(self, symbol: str, current_price: float, 
                                 fast_ma: float, slow_ma: float, rsi: float,
                                 macd: float, macd_signal: float, macd_hist: float,
                                 volume_ratio: float, support: float, 
                                 resistance: float, bb_upper: float, bb_lower: float) -> Optional[Signal]:
        """Generate momentum-based trading signal"""
        
        signal_type = "HOLD"
        strength = 0.0
        
        metadata = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'volume_ratio': volume_ratio,
            'support': support,
            'resistance': resistance,
            'current_price': current_price
        }
        
        # Trend detection
        ma_crossover = fast_ma > slow_ma
        macd_bullish = macd > macd_signal and macd_hist > 0
        
        # Momentum conditions
        strong_volume = volume_ratio > 1.5  # Above average volume
        breakout_above_resistance = current_price > resistance * 1.001  # 0.1% buffer
        breakdown_below_support = current_price < support * 0.999  # 0.1% buffer
        
        # Bullish momentum signal
        if (ma_crossover and macd_bullish and 
            rsi > 40 and rsi < self.rsi_overbought and volume_ratio > 1.2):  # Lower thresholds for scalping
            signal_type = "BUY"
            strength = 0.4  # Lower base strength for scalping
            
            # Boost strength for breakout
            if breakout_above_resistance:
                strength = min(strength * 1.2, 0.8)  # Cap at 0.8 for scalping
                metadata['signal_reason'] = "scalp_breakout"
            else:
                metadata['signal_reason'] = "scalp_momentum"
        
        # Bearish momentum signal
        elif (not ma_crossover and not macd_bullish and 
              rsi < 60 and rsi > self.rsi_oversold and volume_ratio > 1.2):  # Lower thresholds for scalping
            signal_type = "SELL"
            strength = 0.4  # Lower base strength for scalping
            
            # Boost strength for breakdown
            if breakdown_below_support:
                strength = min(strength * 1.2, 0.8)  # Cap at 0.8 for scalping
                metadata['signal_reason'] = "scalp_breakdown"
            else:
                metadata['signal_reason'] = "scalp_momentum"
        
        # Additional confirmation filters
        if signal_type != "HOLD":
            # RSI filter
            if signal_type == "BUY" and rsi > self.rsi_overbought:
                strength *= 0.5  # Reduce strength if overbought
            elif signal_type == "SELL" and rsi < self.rsi_oversold:
                strength *= 0.5  # Reduce strength if oversold
            
            # Volume confirmation
            if not strong_volume:
                strength *= 0.7  # Reduce strength without volume confirmation
            
            # Trend consistency check
            trend_score = self._calculate_trend_consistency(symbol, signal_type)
            strength *= trend_score
        
        # Filter weak signals - lower threshold for scalping
        if strength < 0.25:  # Lower from 0.4 to 0.25 for scalping
            signal_type = "HOLD"
            strength = 0.0
        
        # Avoid repeated signals - shorter cooldown for scalping
        if symbol in self.last_signals:
            last_signal = self.last_signals[symbol]
            time_diff = (datetime.now() - last_signal.timestamp).seconds
            if last_signal.signal_type == signal_type and time_diff < 60:  # 1 minute cooldown for scalping
                return None
        
        if signal_type != "HOLD":
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            self.last_signals[symbol] = signal
            self._update_trend_history(symbol, signal_type)
            return signal
        
        return None
    
    def _detect_support_resistance(self, high_prices: np.ndarray, 
                                  low_prices: np.ndarray, lookback: int = 20) -> Tuple[float, float]:
        """Detect support and resistance levels"""
        if len(high_prices) < lookback:
            return low_prices[-1], high_prices[-1]
        
        recent_highs = high_prices[-lookback:]
        recent_lows = low_prices[-lookback:]
        
        # Find pivot points
        resistance_candidates = []
        support_candidates = []
        
        for i in range(2, len(recent_highs) - 2):
            # Resistance: local maximum
            if (recent_highs[i] > recent_highs[i-1] and 
                recent_highs[i] > recent_highs[i+1] and
                recent_highs[i] > recent_highs[i-2] and 
                recent_highs[i] > recent_highs[i+2]):
                resistance_candidates.append(recent_highs[i])
            
            # Support: local minimum
            if (recent_lows[i] < recent_lows[i-1] and 
                recent_lows[i] < recent_lows[i+1] and
                recent_lows[i] < recent_lows[i-2] and 
                recent_lows[i] < recent_lows[i+2]):
                support_candidates.append(recent_lows[i])
        
        # Get strongest levels
        resistance = max(resistance_candidates) if resistance_candidates else max(recent_highs)
        support = min(support_candidates) if support_candidates else min(recent_lows)
        
        return support, resistance
    
    def _calculate_trend_consistency(self, symbol: str, signal_type: str) -> float:
        """Calculate trend consistency score"""
        if symbol not in self.trend_history:
            return 0.8  # Neutral score for new symbols
        
        recent_trends = self.trend_history[symbol][-5:]  # Last 5 signals
        
        if len(recent_trends) < 2:
            return 0.8
        
        # Count consistent signals
        consistent_count = sum(1 for trend in recent_trends if trend == signal_type)
        consistency_ratio = consistent_count / len(recent_trends)
        
        # Penalize flip-flopping, reward trend consistency
        if consistency_ratio > 0.6:
            return min(1.0, 0.7 + consistency_ratio * 0.3)
        else:
            return max(0.3, consistency_ratio)
    
    def _update_trend_history(self, symbol: str, signal_type: str):
        """Update trend history for consistency tracking"""
        if symbol not in self.trend_history:
            from collections import deque
            self.trend_history[symbol] = deque(maxlen=10)
        
        self.trend_history[symbol].append(signal_type)
    
    def calculate_position_size(self, symbol: str, signal: Signal, 
                              available_capital: float) -> float:
        """Calculate position size with momentum-specific adjustments"""
        base_size = available_capital * self.position_size
        
        # Adjust for signal strength
        adjusted_size = base_size * signal.strength
        
        # Boost size for breakout signals
        if signal.metadata.get('signal_reason') in ['momentum_breakout', 'momentum_breakdown']:
            adjusted_size *= 1.2
        
        # Volume-based adjustment
        volume_ratio = signal.metadata.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:  # Very high volume
            adjusted_size *= 1.1
        elif volume_ratio < 1.0:  # Below average volume
            adjusted_size *= 0.8
        
        return round(adjusted_size, 2)
    
    def evaluate_signal_performance(self, symbol: str, entry_signal: Signal, 
                                   exit_price: float) -> Dict:
        """Evaluate momentum signal performance"""
        if entry_signal.signal_type == "BUY":
            pnl = (exit_price - entry_signal.price) / entry_signal.price
        else:  # SELL
            pnl = (entry_signal.price - exit_price) / entry_signal.price
        
        if pnl > 0:
            self.successful_signals += 1
        
        performance = {
            'symbol': symbol,
            'strategy': 'momentum',
            'entry_price': entry_signal.price,
            'exit_price': exit_price,
            'pnl_percentage': pnl * 100,
            'signal_strength': entry_signal.strength,
            'signal_reason': entry_signal.metadata.get('signal_reason', 'momentum_trend'),
            'profitable': pnl > 0,
            'metadata': entry_signal.metadata
        }
        
        self.logger.info("Momentum signal performance evaluated", **performance)
        return performance
    
    def get_strategy_stats(self) -> Dict:
        """Get momentum strategy performance statistics"""
        win_rate = (self.successful_signals / max(self.signal_count, 1)) * 100
        
        return {
            'strategy_name': 'momentum',
            'total_signals': self.signal_count,
            'successful_signals': self.successful_signals,
            'win_rate_percentage': round(win_rate, 2),
            'active_positions': len(self.positions),
            'parameters': {
                'fast_ma': self.fast_ma,
                'slow_ma': self.slow_ma,
                'rsi_period': self.rsi_period,
                'position_size': self.position_size
            }
        }
    
    def reset_strategy(self):
        """Reset strategy state"""
        self.positions.clear()
        self.last_signals.clear()
        self.trend_history.clear()
        self.breakout_levels.clear()
        self.signal_count = 0
        self.successful_signals = 0
        
        self.logger.info("Momentum strategy reset") 