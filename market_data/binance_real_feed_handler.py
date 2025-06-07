#!/usr/bin/env python3
"""
Binance Real Market Data Feed Handler - Live Data with Demo Money
Uses Binance WebSocket and REST APIs for real-time market data
"""

import asyncio
import websockets
import json
import pandas as pd
import aiohttp
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Callable, Optional, Any
import queue
import threading
from collections import deque
import time
import numpy as np
from binance.client import Client
from binance import ThreadedWebsocketManager
from utils.logger import get_hft_logger


class BinanceRealFeedHandler:
    """Real-time market data feed handler using Binance APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("binance_real_feed_handler")
        
        # Binance configuration
        self.api_key = config['binance']['api_key']
        self.api_secret = config['binance']['api_secret']
        self.testnet = config['binance'].get('testnet', False)
        
        # Initialize Binance client (read-only)
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )
        
        # WebSocket manager
        self.twm = None
        self.running = False
        
        # Data storage
        self.tick_data = {}  # Symbol -> deque of ticks
        self.orderbook_data = {}  # Symbol -> latest orderbook
        self.ohlcv_data = {}  # Symbol -> timeframe -> deque of candles
        self.latest_prices = {}  # Symbol -> latest price
        
        # Callbacks
        self.tick_callbacks = []
        self.orderbook_callbacks = []
        self.kline_callbacks = []
        
        # Performance tracking
        self.message_count = 0
        self.last_latency_check = time.time()
        self.connection_attempts = 0
        self.last_ping = time.time()
        
        # Initialize data structures
        self._initialize_data_structures()
        
        self.logger.info("Binance real feed handler initialized", 
                        api_key_prefix=self.api_key[:8] + "...",
                        testnet=self.testnet)
    
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
            
            # OHLCV data: keep last 500 candles per timeframe
            self.ohlcv_data[symbol] = {}
            for timeframe in timeframes:
                self.ohlcv_data[symbol][timeframe] = deque(maxlen=500)
            
            # Latest prices
            self.latest_prices[symbol] = None
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick data updates"""
        self.tick_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable):
        """Add callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_kline_callback(self, callback: Callable):
        """Add callback for OHLCV updates"""
        self.kline_callbacks.append(callback)
    
    def _fetch_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
        """Fetch historical OHLCV data from Binance REST API"""
        try:
            # Convert timeframe to Binance format
            interval = self._convert_timeframe_to_binance(timeframe)
            
            # Fetch klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Process and store historical data
            for kline in klines:
                ohlcv_record = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'is_closed': True
                }
                
                self.ohlcv_data[symbol][timeframe].append(ohlcv_record)
            
            self.logger.info(f"Loaded {len(klines)} historical candles for {symbol} {timeframe}")
            
        except Exception as e:
            self.hft_logger.log_error("fetch_historical_data", e, {
                "symbol": symbol,
                "timeframe": timeframe
            })
    
    def _convert_timeframe_to_binance(self, timeframe: str) -> str:
        """Convert internal timeframe to Binance interval"""
        mapping = {
            "1MIN": "1m",
            "5MIN": "5m",
            "15MIN": "15m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1d"
        }
        return mapping.get(timeframe, "1m")
    
    def _handle_trade_message(self, msg):
        """Handle trade/tick updates from WebSocket"""
        try:
            symbol = msg['s']
            
            tick_data = {
                'symbol': symbol,
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'timestamp': datetime.fromtimestamp(msg['T'] / 1000, tz=timezone.utc),
                'trade_id': msg['t'],
                'side': 'buy' if msg['m'] else 'sell'  # m = true if buyer is market maker
            }
            
            # Store tick data
            if symbol in self.tick_data:
                self.tick_data[symbol].append(tick_data)
                
                # Update latest price
                self.latest_prices[symbol] = tick_data['price']
            
            # Execute callbacks
            for callback in self.tick_callbacks:
                try:
                    asyncio.create_task(callback(tick_data))
                except Exception as e:
                    self.hft_logger.log_error("tick_callback", e)
                    
        except Exception as e:
            self.hft_logger.log_error("handle_trade_message", e, {"message": str(msg)[:200]})
    
    def _handle_depth_message(self, msg):
        """Handle orderbook depth updates from WebSocket"""
        try:
            symbol = msg['s']
            
            # Process bids and asks
            bids = [[float(bid[0]), float(bid[1])] for bid in msg['b']]
            asks = [[float(ask[0]), float(ask[1])] for ask in msg['a']]
            
            orderbook = {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.fromtimestamp(msg['E'] / 1000, tz=timezone.utc)
            }
            
            # Store orderbook data
            if symbol in self.orderbook_data:
                self.orderbook_data[symbol] = orderbook
            
            # Execute callbacks
            for callback in self.orderbook_callbacks:
                try:
                    asyncio.create_task(callback(orderbook))
                except Exception as e:
                    self.hft_logger.log_error("orderbook_callback", e)
                    
        except Exception as e:
            self.hft_logger.log_error("handle_depth_message", e, {"message": str(msg)[:200]})
    
    def _handle_kline_message(self, msg):
        """Handle kline/candlestick updates from WebSocket"""
        try:
            kline = msg['k']
            symbol = kline['s']
            
            # Convert interval back to our format
            timeframe = self._convert_binance_to_timeframe(kline['i'])
            
            ohlcv_record = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': kline['x']  # True if kline is closed
            }
            
            # Store OHLCV data
            if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
                # Remove last candle if it's the same timestamp (update in progress)
                candles = self.ohlcv_data[symbol][timeframe]
                if candles and candles[-1]['timestamp'] == ohlcv_record['timestamp']:
                    candles.pop()
                
                candles.append(ohlcv_record)
            
            # Execute callbacks
            for callback in self.kline_callbacks:
                try:
                    asyncio.create_task(callback(ohlcv_record))
                except Exception as e:
                    self.hft_logger.log_error("kline_callback", e)
                    
        except Exception as e:
            self.hft_logger.log_error("handle_kline_message", e, {"message": str(msg)[:200]})
    
    def _convert_binance_to_timeframe(self, interval: str) -> str:
        """Convert Binance interval to internal timeframe"""
        mapping = {
            "1m": "1MIN",
            "5m": "5MIN", 
            "15m": "15MIN",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }
        return mapping.get(interval, "1MIN")
    
    def start_streams(self):
        """Start all data streams"""
        try:
            self.running = True
            self.connection_attempts += 1
            
            # Initialize WebSocket manager
            self.twm = ThreadedWebsocketManager(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.twm.start()
            
            symbols = self.config['trading']['symbols']
            timeframes = self.config['trading']['timeframes']
            
            # Load historical data first
            self.logger.info("Loading historical data...")
            for symbol in symbols:
                for timeframe in timeframes:
                    self._fetch_historical_data(symbol, timeframe)
            
            # Start real-time streams
            self.logger.info("Starting real-time streams...")
            
            # Start trade streams for each symbol
            for symbol in symbols:
                # Trade stream
                self.twm.start_trade_socket(
                    callback=self._handle_trade_message,
                    symbol=symbol
                )
                
                # Depth stream (orderbook)
                self.twm.start_depth_socket(
                    callback=self._handle_depth_message,
                    symbol=symbol,
                    depth=20  # Top 20 levels
                )
                
                # Kline streams for each timeframe
                for timeframe in timeframes:
                    interval = self._convert_timeframe_to_binance(timeframe)
                    self.twm.start_kline_socket(
                        callback=self._handle_kline_message,
                        symbol=symbol,
                        interval=interval
                    )
            
            self.logger.info("Binance real-time streams started successfully")
            
        except Exception as e:
            self.hft_logger.log_error("start_streams", e)
    
    def stop_streams(self):
        """Stop all data streams"""
        self.running = False
        
        if self.twm:
            self.twm.stop()
        
        self.logger.info("Binance real-time streams stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        return self.latest_prices.get(symbol)
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook for a symbol"""
        return self.orderbook_data.get(symbol)
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data as pandas DataFrame"""
        try:
            if symbol in self.ohlcv_data and timeframe in self.ohlcv_data[symbol]:
                data = list(self.ohlcv_data[symbol][timeframe])[-limit:]
                
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            self.hft_logger.log_error("get_ohlcv_data", e)
            return pd.DataFrame()
    
    def get_tick_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent tick data for a symbol"""
        if symbol in self.tick_data:
            return list(self.tick_data[symbol])[-limit:]
        return []
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        current_time = time.time()
        uptime = current_time - self.last_latency_check
        
        return {
            'messages_received': self.message_count,
            'uptime_seconds': uptime,
            'messages_per_second': self.message_count / max(uptime, 1),
            'connection_attempts': self.connection_attempts,
            'last_ping_seconds_ago': current_time - self.last_ping,
            'symbols_tracked': len(self.latest_prices),
            'websocket_connected': self.twm is not None and self.running
        }
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols using REST API"""
        try:
            symbols = self.config['trading']['symbols']
            prices = {}
            
            # Get ticker prices
            tickers = self.client.get_all_tickers()
            
            for ticker in tickers:
                if ticker['symbol'] in symbols:
                    prices[ticker['symbol']] = float(ticker['price'])
            
            return prices
            
        except Exception as e:
            self.hft_logger.log_error("get_current_prices", e)
            return {}
    
    def test_connection(self) -> bool:
        """Test connection to Binance API"""
        try:
            # Test REST API connection
            server_time = self.client.get_server_time()
            self.logger.info("Binance connection test successful", 
                           server_time=server_time['serverTime'])
            return True
            
        except Exception as e:
            self.hft_logger.log_error("test_connection", e)
            return False 