#!/usr/bin/env python3
"""
Real Demo Order Manager - Uses Real Market Data for Realistic Demo Trading
Executes orders with real market prices and realistic execution but with demo money
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import requests
from utils.logger import get_hft_logger


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = "USD"
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    demo_order_id: Optional[str] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealDemoOrderManager:
    """Real demo order management system using live market data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("real_demo_order_manager")
        
        # Binance configuration for real market data
        self.api_key = config['binance']['api_key']
        self.api_secret = config['binance']['api_secret']
        self.testnet = config['binance'].get('testnet', False)
        
        # Initialize Binance client for market data fetching
        try:
            from binance.client import Client
            self.binance_client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
        except ImportError:
            self.binance_client = None
            self.logger.warning("Binance client not available - install python-binance package")
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        
        # Callbacks
        self.fill_callbacks: List[Callable] = []
        self.order_update_callbacks: List[Callable] = []
        
        # Performance metrics
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.total_commission = 0.0
        
        # Real execution settings
        self.commission_rate = 0.001  # 0.1% commission (realistic)
        self.max_slippage = 0.002  # 0.2% max slippage
        self.execution_delay_range = (0.05, 0.5)  # Realistic execution delay
        
        # Real market data cache
        self.market_prices = {}
        self.orderbooks = {}
        self.last_price_update = {}
        
        self.logger.info("Real demo order manager initialized with live market data")
    
    def set_market_prices(self, prices: Dict[str, float]):
        """Update market prices from real market data"""
        self.market_prices.update(prices)
        for symbol in prices:
            self.last_price_update[symbol] = time.time()
    
    def set_orderbook(self, symbol: str, orderbook: Dict):
        """Update orderbook data from real market feeds"""
        self.orderbooks[symbol] = orderbook
    
    def add_fill_callback(self, callback: Callable):
        """Add callback for order fills"""
        self.fill_callbacks.append(callback)
    
    def add_order_update_callback(self, callback: Callable):
        """Add callback for order updates"""
        self.order_update_callbacks.append(callback)
    
    async def submit_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                                 strategy: str = "", metadata: Dict = None) -> Order:
        """Submit a market order using real market data"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        return await self._execute_order(order)
    
    async def submit_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                                price: float, strategy: str = "", 
                                time_in_force: str = "GTC", metadata: Dict = None) -> Order:
        """Submit a limit order using real market data"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        return await self._execute_order(order)
    
    async def submit_stop_loss_order(self, symbol: str, side: OrderSide, quantity: float, 
                                    stop_price: float, limit_price: float,
                                    strategy: str = "", metadata: Dict = None) -> Order:
        """Submit a stop loss order using real market data"""
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        return await self._execute_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found for cancellation")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False
        
        # Execute cancellation
        order.status = OrderStatus.CANCELLED
        order.updated_time = datetime.now()
        
        # Remove from pending orders
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
        
        # Execute callbacks
        await self._execute_order_update_callbacks(order)
        
        self.logger.info(f"Order {order_id} cancelled successfully")
        return True
    
    async def _execute_order(self, order: Order) -> Order:
        """Execute order using real market data"""
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        self.orders_submitted += 1
        
        try:
            # Validate market data availability
            if not self._validate_market_data(order.symbol):
                order.status = OrderStatus.REJECTED
                order.updated_time = datetime.now()
                self.orders_rejected += 1
                
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                
                self.logger.warning(f"Order rejected - no market data for {order.symbol}")
                await self._execute_order_update_callbacks(order)
                return order
            
            # Submit order
            order.status = OrderStatus.SUBMITTED
            order.demo_order_id = f"DEMO_{int(time.time() * 1000)}"
            order.updated_time = datetime.now()
            
            self.logger.info(f"Demo order submitted: {order.order_id} {order.symbol} {order.side.value} {order.quantity}")
            
            # Execute callbacks
            await self._execute_order_update_callbacks(order)
            
            # Schedule real execution with live market data
            asyncio.create_task(self._execute_with_real_data(order))
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_time = datetime.now()
            self.orders_rejected += 1
            
            # Remove from pending
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            
            self.hft_logger.log_error("execute_order", e, {
                "order_id": order.order_id,
                "symbol": order.symbol
            })
            
            await self._execute_order_update_callbacks(order)
            return order
    
    def _validate_market_data(self, symbol: str) -> bool:
        """Validate that we have recent market data"""
        if symbol not in self.market_prices:
            return False
        
        # Check if price data is recent (within last 60 seconds)
        if symbol in self.last_price_update:
            time_since_update = time.time() - self.last_price_update[symbol]
            return time_since_update < 60
        
        return True
    
    async def _execute_with_real_data(self, order: Order):
        """Execute order using real market data with realistic timing"""
        try:
            # Realistic execution delay
            delay = random.uniform(*self.execution_delay_range)
            await asyncio.sleep(delay)
            
            # Check if order was cancelled
            if order.status == OrderStatus.CANCELLED:
                return
            
            # Get real market data for execution
            execution_data = await self._get_real_execution_data(order)
            
            if not execution_data:
                # No execution possible
                order.status = OrderStatus.REJECTED
                order.updated_time = datetime.now()
                self.orders_rejected += 1
                
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                
                self.logger.warning(f"Order rejected - no execution data: {order.order_id}")
                await self._execute_order_update_callbacks(order)
                return
            
            # Execute based on order type
            if order.order_type == OrderType.MARKET:
                await self._execute_market_order(order, execution_data)
            elif order.order_type == OrderType.LIMIT:
                await self._execute_limit_order(order, execution_data)
            else:
                await self._execute_stop_order(order, execution_data)
                
        except Exception as e:
            self.hft_logger.log_error("execute_with_real_data", e, {
                "order_id": order.order_id
            })
    
    async def _get_real_execution_data(self, order: Order) -> Optional[Dict]:
        """Get real market data for order execution"""
        try:
            symbol = order.symbol
            
            # Use cached market price
            market_price = self.market_prices.get(symbol)
            if not market_price:
                return None
            
            # Get real orderbook if available
            orderbook = self.orderbooks.get(symbol, {})
            
            # Get current market data from API if needed
            current_data = await self._fetch_current_market_data(symbol)
            
            return {
                'market_price': market_price,
                'orderbook': orderbook,
                'current_data': current_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.hft_logger.log_error("get_real_execution_data", e)
            return None
    
    async def _fetch_current_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current market data from Binance"""
        try:
            if self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return {
                    'rate': float(ticker['price']),
                    'time': ticker['time']
                }
            else:
                self.logger.warning("Binance client not available - cannot fetch current data")
                return None
                
        except Exception as e:
            self.hft_logger.log_error("fetch_current_market_data", e)
            return None
    
    async def _execute_market_order(self, order: Order, execution_data: Dict):
        """Execute market order with real market data"""
        try:
            market_price = execution_data['market_price']
            orderbook = execution_data.get('orderbook', {})
            
            # Calculate realistic execution price using orderbook if available
            execution_price = self._calculate_realistic_execution_price(order, market_price, orderbook)
            
            # Full fill for market orders (realistic behavior)
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            order.commission = order.quantity * execution_price * self.commission_rate
            order.updated_time = datetime.now()
            
            self.orders_filled += 1
            self.total_commission += order.commission
            
            # Move to filled orders
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            self.filled_orders[order.order_id] = order
            
            self.logger.info(f"Market order filled: {order.order_id} @ ${execution_price:.6f}")
            
            # Execute callbacks
            await self._execute_order_update_callbacks(order)
            await self._execute_fill_callbacks(order)
            
        except Exception as e:
            self.hft_logger.log_error("execute_market_order", e)
    
    async def _execute_limit_order(self, order: Order, execution_data: Dict):
        """Execute limit order with real market data"""
        try:
            market_price = execution_data['market_price']
            
            # Check if limit order can be filled
            can_fill = False
            if order.side == OrderSide.BUY and market_price <= order.price:
                can_fill = True
            elif order.side == OrderSide.SELL and market_price >= order.price:
                can_fill = True
            
            if can_fill:
                # Fill at limit price
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_price = order.price
                order.commission = order.quantity * order.price * self.commission_rate
                order.updated_time = datetime.now()
                
                self.orders_filled += 1
                self.total_commission += order.commission
                
                # Move to filled orders
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                self.filled_orders[order.order_id] = order
                
                self.logger.info(f"Limit order filled: {order.order_id} @ ${order.price:.6f}")
                
                # Execute callbacks
                await self._execute_order_update_callbacks(order)
                await self._execute_fill_callbacks(order)
            else:
                # Order remains pending
                self.logger.debug(f"Limit order pending: {order.order_id} - market ${market_price:.6f}, limit ${order.price:.6f}")
                
        except Exception as e:
            self.hft_logger.log_error("execute_limit_order", e)
    
    async def _execute_stop_order(self, order: Order, execution_data: Dict):
        """Execute stop loss order with real market data"""
        try:
            market_price = execution_data['market_price']
            
            # Check if stop is triggered
            triggered = False
            if order.side == OrderSide.SELL and market_price <= order.stop_price:
                triggered = True
            elif order.side == OrderSide.BUY and market_price >= order.stop_price:
                triggered = True
            
            if triggered:
                # Execute at limit price or market price
                execution_price = order.price or market_price
                
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_price = execution_price
                order.commission = order.quantity * execution_price * self.commission_rate
                order.updated_time = datetime.now()
                
                self.orders_filled += 1
                self.total_commission += order.commission
                
                # Move to filled orders
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                self.filled_orders[order.order_id] = order
                
                self.logger.info(f"Stop order triggered and filled: {order.order_id} @ ${execution_price:.6f}")
                
                # Execute callbacks
                await self._execute_order_update_callbacks(order)
                await self._execute_fill_callbacks(order)
            
        except Exception as e:
            self.hft_logger.log_error("execute_stop_order", e)
    
    def _calculate_realistic_execution_price(self, order: Order, market_price: float, orderbook: Dict) -> float:
        """Calculate realistic execution price using orderbook data"""
        try:
            # Use orderbook for more realistic pricing if available
            if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                if order.side == OrderSide.BUY:
                    # Buy at ask price (with potential slippage)
                    asks = orderbook['asks']
                    if asks:
                        best_ask = asks[0][0]  # Best ask price
                        # Add small amount of slippage
                        slippage = random.uniform(0, self.max_slippage)
                        return best_ask * (1 + slippage)
                else:
                    # Sell at bid price (with potential slippage)
                    bids = orderbook['bids']
                    if bids:
                        best_bid = bids[0][0]  # Best bid price
                        # Add small amount of slippage
                        slippage = random.uniform(0, self.max_slippage)
                        return best_bid * (1 - slippage)
            
            # Fallback to market price with realistic slippage
            slippage = random.uniform(0, self.max_slippage)
            if order.side == OrderSide.BUY:
                return market_price * (1 + slippage)
            else:
                return market_price * (1 - slippage)
                
        except Exception as e:
            self.hft_logger.log_error("calculate_realistic_execution_price", e)
            return market_price
    
    async def _execute_fill_callbacks(self, order: Order):
        """Execute fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                await callback(order)
            except Exception as e:
                self.hft_logger.log_error("fill_callback", e)
    
    async def _execute_order_update_callbacks(self, order: Order):
        """Execute order update callbacks"""
        for callback in self.order_update_callbacks:
            try:
                await callback(order)
            except Exception as e:
                self.hft_logger.log_error("order_update_callback", e)
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"DEMO_{uuid.uuid4().hex[:8]}"
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_strategy(self, strategy: str) -> List[Order]:
        """Get all orders for a strategy"""
        return [order for order in self.orders.values() if order.strategy == strategy]
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return list(self.pending_orders.values())
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return list(self.filled_orders.values())
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        total_orders = self.orders_submitted
        fill_rate = (self.orders_filled / total_orders) * 100 if total_orders > 0 else 0
        rejection_rate = (self.orders_rejected / total_orders) * 100 if total_orders > 0 else 0
        
        return {
            'total_orders_submitted': total_orders,
            'orders_filled': self.orders_filled,
            'orders_rejected': self.orders_rejected,
            'orders_pending': len(self.pending_orders),
            'fill_rate_percentage': fill_rate,
            'rejection_rate_percentage': rejection_rate,
            'total_commission': self.total_commission,
            'average_fill_time_seconds': self._calculate_average_fill_time()
        }
    
    def _calculate_average_fill_time(self) -> float:
        """Calculate average fill time for filled orders"""
        filled_orders = [order for order in self.filled_orders.values()]
        
        if not filled_orders:
            return 0.0
        
        total_time = 0.0
        for order in filled_orders:
            fill_time = (order.updated_time - order.created_time).total_seconds()
            total_time += fill_time
        
        return total_time / len(filled_orders) 