#!/usr/bin/env python3
"""
CoinAPI.io Feed Handler - Real-time Market Data from CoinAPI.io
Provides tick data, OHLCV data, and order book feeds using CoinAPI.io WebSocket and REST APIs
"""

import asyncio
import websockets
import json
import pandas as pd
import aiohttp
import ssl
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Callable, Optional, Any
import queue
import threading
from collections import deque
import time
import numpy as np
from utils.logger import get_hft_logger


class CoinAPIFeedHandler:
    """Real-time market data feed handler for CoinAPI.io"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("coinapi_feed_handler")
        
        # CoinAPI.io configuration
        self.api_key = config['coinapi']['api_key']
        self.base_url = config['coinapi']['base_url']
        self.websocket_url = config['coinapi']['websocket_url']
        self.sandbox = config['coinapi'].get('sandbox', False)
        
        # WebSocket connection
        self.websocket = None
        self.websocket_task = None
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
        
        # Headers for API requests
        self.headers = {
            'X-CoinAPI-Key': self.api_key,
            'Accept': 'application/json'
        }
        
        # Initialize data structures
        self._initialize_data_structures()
        
        self.logger.info("CoinAPI.io feed handler initialized", 
                        api_key_prefix=self.api_key[:8] + "...",
                        sandbox=self.sandbox)
    
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
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
        """Fetch historical OHLCV data from REST API"""
        try:
            # Convert timeframe to CoinAPI format
            period_id = self._convert_timeframe(timeframe)
            
            # Calculate time range
            end_time = datetime.now(timezone.utc)
            
            # Estimate time delta based on timeframe
            if timeframe == "1MIN":
                delta = timedelta(minutes=limit)
            elif timeframe == "5MIN":
                delta = timedelta(minutes=limit * 5)
            elif timeframe == "15MIN":
                delta = timedelta(minutes=limit * 15)
            else:
                delta = timedelta(hours=limit)
            
            start_time = end_time - delta
            
            url = f"{self.base_url}/v1/ohlcv/{symbol}/history"
            params = {
                'period_id': period_id,
                'time_start': start_time.isoformat(),
                'time_end': end_time.isoformat(),
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process and store historical data
                        for candle in data:
                            ohlcv_record = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'timestamp': datetime.fromisoformat(candle['time_period_start'].replace('Z', '+00:00')),
                                'open': float(candle['price_open']),
                                'high': float(candle['price_high']),
                                'low': float(candle['price_low']),
                                'close': float(candle['price_close']),
                                'volume': float(candle.get('volume_traded', 0)),
                                'is_closed': True
                            }
                            
                            self.ohlcv_data[symbol][timeframe].append(ohlcv_record)
                        
                        self.logger.info(f"Loaded {len(data)} historical candles for {symbol} {timeframe}")
                        
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to fetch historical data: {response.status} - {error_text}")
                        
        except Exception as e:
            self.hft_logger.log_error("fetch_historical_data", e, {
                "symbol": symbol,
                "timeframe": timeframe
            })
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe to CoinAPI period ID"""
        mapping = {
            "1MIN": "1MIN",
            "5MIN": "5MIN",
            "15MIN": "15MIN",
            "1H": "1HRS",
            "4H": "4HRS",
            "1D": "1DAY"
        }
        return mapping.get(timeframe, "1MIN")
    
    async def _websocket_handler(self):
        """Handle WebSocket connection and messages"""
        while self.running:
            try:
                self.connection_attempts += 1
                self.logger.info(f"Attempting WebSocket connection #{self.connection_attempts}")
                
                # Create SSL context
                ssl_context = ssl.create_default_context()
                
                # WebSocket URL with API key
                ws_url = f"{self.websocket_url}?apikey={self.api_key}"
                
                async with websockets.connect(
                    ws_url,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.logger.info("WebSocket connected successfully")
                    
                    # Subscribe to symbols
                    await self._subscribe_to_feeds()
                    
                    # Handle messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON received: {message[:100]}")
                        except Exception as e:
                            self.hft_logger.log_error("websocket_message", e)
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, attempting reconnect...")
                await asyncio.sleep(5)
            except Exception as e:
                self.hft_logger.log_error("websocket_connection", e)
                await asyncio.sleep(10)
    
    async def _subscribe_to_feeds(self):
        """Subscribe to market data feeds for all symbols"""
        symbols = self.config['trading']['symbols']
        
        for symbol in symbols:
            # Subscribe to trades (tick data)
            trade_message = {
                "type": "hello",
                "apikey": self.api_key,
                "heartbeat": True,
                "subscribe_data_type": ["trade"],
                "subscribe_filter_symbol_id": [symbol]
            }
            
            await self.websocket.send(json.dumps(trade_message))
            
            # Subscribe to order book
            book_message = {
                "type": "hello",
                "apikey": self.api_key,
                "heartbeat": True,
                "subscribe_data_type": ["book"],
                "subscribe_filter_symbol_id": [symbol]
            }
            
            await self.websocket.send(json.dumps(book_message))
            
            self.logger.info(f"Subscribed to feeds for {symbol}")
    
    async def _handle_websocket_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        try:
            self.message_count += 1
            
            if isinstance(data, list):
                # Handle array of messages
                for message in data:
                    await self._process_single_message(message)
            else:
                # Handle single message
                await self._process_single_message(data)
                
        except Exception as e:
            self.hft_logger.log_error("handle_websocket_message", e, {"data": str(data)[:200]})
    
    async def _process_single_message(self, message: Dict):
        """Process a single WebSocket message"""
        try:
            msg_type = message.get('type', '')
            
            if msg_type == 'trade':
                await self._handle_trade_update(message)
            elif msg_type == 'book':
                await self._handle_orderbook_update(message)
            elif msg_type == 'heartbeat':
                self.last_ping = time.time()
                self.logger.debug("Heartbeat received")
            elif msg_type == 'error':
                self.logger.error(f"WebSocket error: {message.get('message', 'Unknown error')}")
            
        except Exception as e:
            self.hft_logger.log_error("process_single_message", e, {"message": str(message)[:200]})
    
    async def _handle_trade_update(self, message: Dict):
        """Handle trade/tick updates"""
        try:
            symbol = message.get('symbol_id', '')
            
            tick_data = {
                'symbol': symbol,
                'price': float(message.get('price', 0)),
                'quantity': float(message.get('size', 0)),
                'timestamp': datetime.fromisoformat(message.get('time_exchange', '').replace('Z', '+00:00')),
                'trade_id': message.get('uuid', ''),
                'side': message.get('taker_side', 'unknown')
            }
            
            # Store tick data
            if symbol in self.tick_data:
                self.tick_data[symbol].append(tick_data)
                
                # Update latest price
                self.latest_prices[symbol] = tick_data['price']
            
            # Execute callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick_data)
                except Exception as e:
                    self.hft_logger.log_error("tick_callback", e)
                    
        except Exception as e:
            self.hft_logger.log_error("handle_trade_update", e, {"message": str(message)[:200]})
    
    async def _handle_orderbook_update(self, message: Dict):
        """Handle orderbook updates"""
        try:
            symbol = message.get('symbol_id', '')
            
            # Process bids and asks
            bids = []
            asks = []
            
            if 'bids' in message:
                bids = [[float(level.get('price', 0)), float(level.get('size', 0))] 
                       for level in message['bids']]
            
            if 'asks' in message:
                asks = [[float(level.get('price', 0)), float(level.get('size', 0))] 
                       for level in message['asks']]
            
            orderbook = {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.fromisoformat(message.get('time_exchange', '').replace('Z', '+00:00'))
            }
            
            # Store orderbook data
            if symbol in self.orderbook_data:
                self.orderbook_data[symbol] = orderbook
            
            # Execute callbacks
            for callback in self.orderbook_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    self.hft_logger.log_error("orderbook_callback", e)
                    
        except Exception as e:
            self.hft_logger.log_error("handle_orderbook_update", e, {"message": str(message)[:200]})
    
    def start_streams(self):
        """Start all data streams"""
        try:
            self.running = True
            
            # Start WebSocket in a separate thread to avoid event loop conflicts
            self.websocket_task = threading.Thread(target=self._start_websocket_thread, daemon=True)
            self.websocket_task.start()
            
            # Start background task for historical data loading in a separate thread
            historical_thread = threading.Thread(target=self._load_historical_data_sync, daemon=True)
            historical_thread.start()
            
            self.logger.info("CoinAPI.io streams started")
            
        except Exception as e:
            self.hft_logger.log_error("start_streams", e)
    
    def _start_websocket_thread(self):
        """Start WebSocket in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._websocket_handler())
    
    def stop_streams(self):
        """Stop all data streams"""
        self.running = False
        
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        
        if self.websocket_task and self.websocket_task.is_alive():
            self.websocket_task.join(timeout=5)
        
        self.logger.info("CoinAPI.io streams stopped")
    
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
            'websocket_connected': self.websocket is not None and not self.websocket.closed if self.websocket else False
        }
    
    def _load_historical_data_sync(self):
        """Load historical data synchronously in a separate thread"""
        try:
            symbols = self.config['trading']['symbols']
            timeframes = self.config['trading']['timeframes']
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def load_all_historical():
                tasks = []
                for symbol in symbols:
                    for timeframe in timeframes:
                        task = self._fetch_historical_data(symbol, timeframe)
                        tasks.append(task)
                await asyncio.gather(*tasks, return_exceptions=True)
            
            loop.run_until_complete(load_all_historical())
            loop.close()
            
        except Exception as e:
            self.hft_logger.log_error("load_historical_data_sync", e) 