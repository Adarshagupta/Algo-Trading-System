import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Callable, Optional, Any
from binance import ThreadedWebsocketManager, BinanceSocketManager
from binance.client import Client
import queue
import threading
from collections import deque
import time
import numpy as np
from utils.logger import get_hft_logger


class BinanceFeedHandler:
    """Real-time market data feed handler for Binance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("feed_handler")  # Get BoundLogger for regular logging
        
        # Initialize Binance client (testnet)
        self.client = Client(
            api_key=config['binance']['api_key'],
            api_secret=config['binance']['api_secret'],
            testnet=config['binance'].get('testnet', True)
        )
        
        # WebSocket manager
        self.bsm = None
        self.twm = None
        
        # Data storage
        self.tick_data = {}  # Symbol -> deque of ticks
        self.orderbook_data = {}  # Symbol -> latest orderbook
        self.kline_data = {}  # Symbol -> timeframe -> deque of klines
        
        # Callbacks
        self.tick_callbacks = []
        self.orderbook_callbacks = []
        self.kline_callbacks = []
        
        # Performance tracking
        self.message_count = 0
        self.last_latency_check = time.time()
        
        # Initialize data structures
        self._initialize_data_structures()
        
    def _initialize_data_structures(self):
        """Initialize data storage structures for all symbols"""
        symbols = self.config['trading']['symbols']
        timeframes = self.config['trading']['timeframes']
        
        for symbol in symbols:
            # Tick data: keep last 1000 ticks per symbol
            self.tick_data[symbol] = deque(maxlen=1000)
            
            # Orderbook data: keep latest snapshot
            self.orderbook_data[symbol] = {
                'bids': [],
                'asks': [],
                'timestamp': None
            }
            
            # Kline data: keep last 500 candles per timeframe
            self.kline_data[symbol] = {}
            for timeframe in timeframes:
                self.kline_data[symbol][timeframe] = deque(maxlen=500)
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick data updates"""
        self.tick_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable):
        """Add callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_kline_callback(self, callback: Callable):
        """Add callback for kline updates"""
        self.kline_callbacks.append(callback)
    
    def _handle_socket_message(self, msg):
        """Handle incoming WebSocket messages"""
        try:
            self.message_count += 1
            
            # Calculate latency
            current_time = time.time() * 1000  # ms
            if 'E' in msg:  # Event time in message
                latency = current_time - msg['E']
                if latency > self.config.get('monitoring', {}).get('latency_threshold_ms', 100):
                    self.logger.warning("High latency detected", latency_ms=latency)
            
            # Route message based on stream type
            if 'e' in msg:
                event_type = msg['e']
                
                if event_type == 'trade':
                    self._handle_trade_update(msg)
                elif event_type == 'depthUpdate':
                    self._handle_depth_update(msg)
                elif event_type == 'kline':
                    self._handle_kline_update(msg)
                    
        except Exception as e:
            self.hft_logger.log_error("feed_handler", e, {"message": msg})
    
    def _handle_trade_update(self, msg):
        """Handle trade/tick updates"""
        symbol = msg['s']
        
        tick_data = {
            'symbol': symbol,
            'price': float(msg['p']),
            'quantity': float(msg['q']),
            'timestamp': datetime.fromtimestamp(msg['T'] / 1000, tz=timezone.utc),
            'is_buyer_maker': msg['m'],
            'trade_id': msg['t']
        }
        
        # Store tick data
        self.tick_data[symbol].append(tick_data)
        
        # Execute callbacks
        for callback in self.tick_callbacks:
            try:
                callback(tick_data)
            except Exception as e:
                self.hft_logger.log_error("tick_callback", e)
    
    def _handle_depth_update(self, msg):
        """Handle orderbook depth updates"""
        symbol = msg['s']
        
        # Update orderbook
        orderbook = {
            'symbol': symbol,
            'bids': [[float(bid[0]), float(bid[1])] for bid in msg['b']],
            'asks': [[float(ask[0]), float(ask[1])] for ask in msg['a']],
            'timestamp': datetime.fromtimestamp(msg['E'] / 1000, tz=timezone.utc),
            'last_update_id': msg['u']
        }
        
        self.orderbook_data[symbol] = orderbook
        
        # Execute callbacks
        for callback in self.orderbook_callbacks:
            try:
                callback(orderbook)
            except Exception as e:
                self.hft_logger.log_error("orderbook_callback", e)
    
    def _handle_kline_update(self, msg):
        """Handle kline/candlestick updates"""
        kline = msg['k']
        symbol = kline['s']
        interval = kline['i']
        
        kline_data = {
            'symbol': symbol,
            'interval': interval,
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'timestamp': datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc),
            'is_closed': kline['x']  # Whether this kline is closed
        }
        
        # Store kline data
        if interval in self.kline_data[symbol]:
            self.kline_data[symbol][interval].append(kline_data)
        
        # Execute callbacks
        for callback in self.kline_callbacks:
            try:
                callback(kline_data)
            except Exception as e:
                self.hft_logger.log_error("kline_callback", e)
    
    def start_streams(self):
        """Start all WebSocket streams"""
        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.config['binance']['api_key'],
                api_secret=self.config['binance']['api_secret'],
                testnet=self.config['binance'].get('testnet', True)
            )
            
            self.twm.start()
            
            symbols = self.config['trading']['symbols']
            timeframes = self.config['trading']['timeframes']
            
            # Start trade streams
            for symbol in symbols:
                self.twm.start_trade_socket(
                    callback=self._handle_socket_message,
                    symbol=symbol.lower()
                )
                self.logger.info(f"Started trade stream for {symbol}")
            
            # Start depth streams
            for symbol in symbols:
                self.twm.start_depth_socket(
                    callback=self._handle_socket_message,
                    symbol=symbol.lower(),
                    depth=20
                )
                self.logger.info(f"Started depth stream for {symbol}")
            
            # Start kline streams
            for symbol in symbols:
                for timeframe in timeframes:
                    self.twm.start_kline_socket(
                        callback=self._handle_socket_message,
                        symbol=symbol.lower(),
                        interval=timeframe
                    )
                    self.logger.info(f"Started kline stream for {symbol} {timeframe}")
            
            self.logger.info("All WebSocket streams started successfully")
            
        except Exception as e:
            self.hft_logger.log_error("start_streams", e)
            raise
    
    def stop_streams(self):
        """Stop all WebSocket streams"""
        try:
            if self.twm:
                self.twm.stop()
                self.logger.info("All WebSocket streams stopped")
        except Exception as e:
            self.hft_logger.log_error("stop_streams", e)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        if symbol in self.tick_data and self.tick_data[symbol]:
            return self.tick_data[symbol][-1]['price']
        return None
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook for a symbol"""
        return self.orderbook_data.get(symbol)
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data as pandas DataFrame"""
        if symbol in self.kline_data and timeframe in self.kline_data[symbol]:
            data = list(self.kline_data[symbol][timeframe])[-limit:]
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df[['open', 'high', 'low', 'close', 'volume']]
        
        return pd.DataFrame()
    
    def get_tick_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent tick data for a symbol"""
        if symbol in self.tick_data:
            return list(self.tick_data[symbol])[-limit:]
        return []
    
    def get_performance_stats(self) -> Dict:
        """Get feed handler performance statistics"""
        current_time = time.time()
        time_elapsed = current_time - self.last_latency_check
        
        return {
            'messages_per_second': self.message_count / max(time_elapsed, 1),
            'total_messages': self.message_count,
            'active_symbols': len(self.tick_data),
            'uptime_seconds': time_elapsed
        } 