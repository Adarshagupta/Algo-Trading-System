import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import talib
from utils.logger import get_hft_logger


@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # Signal strength 0-1
    price: float
    timestamp: datetime
    metadata: Dict


class MeanReversionStrategy:
    """Mean reversion trading strategy using statistical analysis"""
    
    def __init__(self, config: Dict):
        self.config = config['strategies']['mean_reversion']
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("mean_reversion")  # Get BoundLogger for regular logging
        
        # Strategy parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.std_dev_threshold = self.config.get('std_dev_threshold', 2.0)
        self.position_size = self.config.get('position_size', 0.1)
        
        # State tracking
        self.positions = {}  # Symbol -> position info
        self.last_signals = {}  # Symbol -> last signal
        self.price_history = {}  # Symbol -> price deque
        
        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        
    def analyze_market_data(self, symbol: str, ohlcv_data: pd.DataFrame) -> Optional[Signal]:
        """Analyze market data and generate trading signals"""
        try:
            if len(ohlcv_data) < self.lookback_period:
                return None
            
            # Calculate technical indicators
            close_prices = ohlcv_data['close'].values
            current_price = close_prices[-1]
            
            # Calculate moving average and standard deviation
            ma = np.mean(close_prices[-self.lookback_period:])
            std = np.std(close_prices[-self.lookback_period:])
            
            # Calculate z-score (how many standard deviations from mean)
            z_score = (current_price - ma) / std if std > 0 else 0
            
            # Additional technical indicators
            rsi = talib.RSI(close_prices, timeperiod=14)[-1] if len(close_prices) >= 14 else 50
            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            ) if len(close_prices) >= 20 else (current_price, current_price, current_price)
            
            # Generate signal based on mean reversion logic
            signal = self._generate_signal(
                symbol, current_price, z_score, rsi, 
                bollinger_upper[-1], bollinger_lower[-1], ma
            )
            
            if signal:
                self.signal_count += 1
                self.hft_logger.log_signal(
                    symbol, signal.signal_type, signal.strength, 
                    "mean_reversion", signal.metadata
                )
            
            return signal
            
        except Exception as e:
            self.hft_logger.log_error("analyze_market_data", e, {"symbol": symbol})
            return None
    
    def _generate_signal(self, symbol: str, current_price: float, z_score: float, 
                        rsi: float, bb_upper: float, bb_lower: float, ma: float) -> Optional[Signal]:
        """Generate trading signal based on mean reversion analysis"""
        
        signal_type = "HOLD"
        strength = 0.0
        metadata = {
            'z_score': z_score,
            'rsi': rsi,
            'bollinger_upper': bb_upper,
            'bollinger_lower': bb_lower,
            'moving_average': ma,
            'current_price': current_price
        }
        
        # Mean reversion logic - more sensitive for scalping
        if z_score < -self.std_dev_threshold and rsi < 40:  # Less extreme RSI for scalping
            # Price is significantly below mean and oversold - potential BUY
            signal_type = "BUY"
            strength = min(abs(z_score) / self.std_dev_threshold * 0.6, 0.8)  # Lower strength for scalping
            
        elif z_score > self.std_dev_threshold and rsi > 60:  # Less extreme RSI for scalping
            # Price is significantly above mean and overbought - potential SELL
            signal_type = "SELL"
            strength = min(abs(z_score) / self.std_dev_threshold * 0.6, 0.8)  # Lower strength for scalping
        
        # Additional confirmation using Bollinger Bands
        if signal_type == "BUY" and current_price <= bb_lower:
            strength = min(strength * 1.2, 1.0)  # Increase confidence
        elif signal_type == "SELL" and current_price >= bb_upper:
            strength = min(strength * 1.2, 1.0)  # Increase confidence
        
        # Filter weak signals - lower threshold for scalping
        if strength < 0.2:  # Lower from 0.3 to 0.2 for scalping
            signal_type = "HOLD"
            strength = 0.0
        
        # Avoid repeated signals - shorter cooldown for scalping
        if symbol in self.last_signals:
            last_signal = self.last_signals[symbol]
            if (last_signal.signal_type == signal_type and 
                (datetime.now() - last_signal.timestamp).seconds < 60):  # 1 minute for scalping
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
            return signal
        
        return None
    
    def calculate_position_size(self, symbol: str, signal: Signal, 
                              available_capital: float) -> float:
        """Calculate optimal position size based on risk management"""
        base_size = available_capital * self.position_size
        
        # Adjust size based on signal strength
        adjusted_size = base_size * signal.strength
        
        # Apply additional risk adjustments
        volatility_adjustment = self._calculate_volatility_adjustment(symbol)
        final_size = adjusted_size * volatility_adjustment
        
        return round(final_size, 2)
    
    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on volatility"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return 0.5  # Conservative default
        
        prices = list(self.price_history[symbol])
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(1440)  # Annualized volatility (1440 minutes in a day)
        
        # Reduce position size for high volatility assets
        if volatility > 0.1:  # 10% daily volatility
            return 0.3
        elif volatility > 0.05:  # 5% daily volatility
            return 0.6
        else:
            return 1.0
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for volatility calculations"""
        if symbol not in self.price_history:
            from collections import deque
            self.price_history[symbol] = deque(maxlen=100)
        
        self.price_history[symbol].append(price)
    
    def evaluate_signal_performance(self, symbol: str, entry_signal: Signal, 
                                   exit_price: float) -> Dict:
        """Evaluate the performance of a completed trade"""
        if entry_signal.signal_type == "BUY":
            pnl = (exit_price - entry_signal.price) / entry_signal.price
        else:  # SELL
            pnl = (entry_signal.price - exit_price) / entry_signal.price
        
        if pnl > 0:
            self.successful_signals += 1
        
        performance = {
            'symbol': symbol,
            'strategy': 'mean_reversion',
            'entry_price': entry_signal.price,
            'exit_price': exit_price,
            'pnl_percentage': pnl * 100,
            'signal_strength': entry_signal.strength,
            'profitable': pnl > 0,
            'metadata': entry_signal.metadata
        }
        
        self.logger.info("Signal performance evaluated", **performance)
        return performance
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        win_rate = (self.successful_signals / max(self.signal_count, 1)) * 100
        
        return {
            'strategy_name': 'mean_reversion',
            'total_signals': self.signal_count,
            'successful_signals': self.successful_signals,
            'win_rate_percentage': round(win_rate, 2),
            'active_positions': len(self.positions),
            'parameters': {
                'lookback_period': self.lookback_period,
                'std_dev_threshold': self.std_dev_threshold,
                'position_size': self.position_size
            }
        }
    
    def reset_strategy(self):
        """Reset strategy state (useful for backtesting)"""
        self.positions.clear()
        self.last_signals.clear()
        self.price_history.clear()
        self.signal_count = 0
        self.successful_signals = 0
        
        self.logger.info("Mean reversion strategy reset")


class EnhancedMeanReversionStrategy(MeanReversionStrategy):
    """Enhanced mean reversion strategy with machine learning features"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.feature_history = {}  # Store feature vectors for ML
        
    def extract_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning models"""
        if len(ohlcv_data) < 50:
            return np.array([])
        
        close = ohlcv_data['close'].values
        volume = ohlcv_data['volume'].values
        
        features = []
        
        # Price-based features
        returns = np.diff(np.log(close))
        features.extend([
            np.mean(returns[-20:]),  # Recent average return
            np.std(returns[-20:]),   # Recent volatility
            np.skew(returns[-20:]) if len(returns) >= 20 else 0,  # Skewness
        ])
        
        # Technical indicators
        rsi = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
        macd, macd_signal, macd_hist = talib.MACD(close) if len(close) >= 26 else (0, 0, 0)
        
        features.extend([
            rsi / 100,  # Normalized RSI
            macd[-1] if len(macd) > 0 else 0,
            macd_hist[-1] if len(macd_hist) > 0 else 0,
        ])
        
        # Volume features
        volume_ma = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
        volume_ratio = volume[-1] / volume_ma if volume_ma > 0 else 1
        
        features.append(min(volume_ratio, 5))  # Cap extreme values
        
        return np.array(features) 