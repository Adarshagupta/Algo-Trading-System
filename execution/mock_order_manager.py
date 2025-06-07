#!/usr/bin/env python3
"""
Mock Order Manager - Simulates Order Execution for Demo Trading
Provides realistic order simulation without actual market execution
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
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
    mock_order_id: Optional[str] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockOrderManager:
    """Mock order management system for demo trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("mock_order_manager")
        
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
        
        # Mock execution settings
        self.fill_probability = 0.95  # 95% chance of fill
        self.partial_fill_probability = 0.1  # 10% chance of partial fill
        self.fill_delay_range = (0.1, 2.0)  # Fill delay between 0.1-2.0 seconds
        self.slippage_range = (0.0001, 0.003)  # 0.01% to 0.3% slippage
        self.commission_rate = 0.001  # 0.1% commission
        
        # Market data reference (will be set from feed handler)
        self.market_prices = {}
        
        self.logger.info("Mock order manager initialized for demo trading")
    
    def set_market_prices(self, prices: Dict[str, float]):
        """Update market prices for order simulation"""
        self.market_prices.update(prices)
    
    def add_fill_callback(self, callback: Callable):
        """Add callback for order fills"""
        self.fill_callbacks.append(callback)
    
    def add_order_update_callback(self, callback: Callable):
        """Add callback for order updates"""
        self.order_update_callbacks.append(callback)
    
    async def submit_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                                 strategy: str = "", metadata: Dict = None) -> Order:
        """Submit a market order"""
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
        """Submit a limit order"""
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
        """Submit a stop loss order"""
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
        
        # Simulate cancellation
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
        """Execute order with mock simulation"""
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        self.orders_submitted += 1
        
        try:
            # Simulate order submission
            order.status = OrderStatus.SUBMITTED
            order.mock_order_id = f"MOCK_{int(time.time() * 1000)}"
            order.updated_time = datetime.now()
            
            self.logger.info(f"Mock order submitted: {order.order_id} {order.symbol} {order.side.value} {order.quantity}")
            
            # Execute callbacks
            await self._execute_order_update_callbacks(order)
            
            # Schedule mock execution
            asyncio.create_task(self._simulate_order_execution(order))
            
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
    
    async def _simulate_order_execution(self, order: Order):
        """Simulate realistic order execution"""
        try:
            # Random delay for execution
            delay = random.uniform(*self.fill_delay_range)
            await asyncio.sleep(delay)
            
            # Check if order was cancelled
            if order.status == OrderStatus.CANCELLED:
                return
            
            # Simulate rejection (5% chance)
            if random.random() > self.fill_probability:
                order.status = OrderStatus.REJECTED
                order.updated_time = datetime.now()
                self.orders_rejected += 1
                
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                
                self.logger.warning(f"Mock order rejected: {order.order_id}")
                await self._execute_order_update_callbacks(order)
                return
            
            # Get execution price
            execution_price = self._calculate_execution_price(order)
            if execution_price is None:
                # No market price available, reject order
                order.status = OrderStatus.REJECTED
                order.updated_time = datetime.now()
                self.orders_rejected += 1
                
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
                
                self.logger.warning(f"Mock order rejected (no market price): {order.order_id}")
                await self._execute_order_update_callbacks(order)
                return
            
            # Simulate partial fill (10% chance for limit orders)
            if order.order_type == OrderType.LIMIT and random.random() < self.partial_fill_probability:
                # Partial fill
                fill_ratio = random.uniform(0.3, 0.8)
                filled_quantity = order.quantity * fill_ratio
                
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = filled_quantity
                order.average_price = execution_price
                order.commission = filled_quantity * execution_price * self.commission_rate
                order.updated_time = datetime.now()
                
                self.total_commission += order.commission
                
                self.logger.info(f"Mock order partially filled: {order.order_id} - {filled_quantity}/{order.quantity}")
                await self._execute_order_update_callbacks(order)
                
                # Schedule remaining fill
                asyncio.create_task(self._simulate_remaining_fill(order))
                
            else:
                # Full fill
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
                
                self.logger.info(f"Mock order filled: {order.order_id} @ ${execution_price:.2f}")
                
                # Execute callbacks
                await self._execute_order_update_callbacks(order)
                await self._execute_fill_callbacks(order)
                
        except Exception as e:
            self.hft_logger.log_error("simulate_order_execution", e, {
                "order_id": order.order_id
            })
    
    async def _simulate_remaining_fill(self, order: Order):
        """Simulate filling the remaining quantity"""
        try:
            # Wait a bit more
            delay = random.uniform(1.0, 5.0)
            await asyncio.sleep(delay)
            
            # Check if order was cancelled
            if order.status == OrderStatus.CANCELLED:
                return
            
            remaining_quantity = order.quantity - order.filled_quantity
            execution_price = self._calculate_execution_price(order)
            
            if execution_price is None:
                return
            
            # Fill remaining
            total_value = (order.filled_quantity * order.average_price) + (remaining_quantity * execution_price)
            order.filled_quantity = order.quantity
            order.average_price = total_value / order.quantity
            order.status = OrderStatus.FILLED
            order.commission += remaining_quantity * execution_price * self.commission_rate
            order.updated_time = datetime.now()
            
            self.orders_filled += 1
            
            # Move to filled orders
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            self.filled_orders[order.order_id] = order
            
            self.logger.info(f"Mock order fully filled: {order.order_id}")
            
            # Execute callbacks
            await self._execute_order_update_callbacks(order)
            await self._execute_fill_callbacks(order)
            
        except Exception as e:
            self.hft_logger.log_error("simulate_remaining_fill", e)
    
    def _calculate_execution_price(self, order: Order) -> Optional[float]:
        """Calculate realistic execution price with slippage"""
        market_price = self.market_prices.get(order.symbol)
        
        if market_price is None:
            return None
        
        if order.order_type == OrderType.MARKET:
            # Market order with slippage
            slippage = random.uniform(*self.slippage_range)
            if order.side == OrderSide.BUY:
                # Buy higher due to slippage
                execution_price = market_price * (1 + slippage)
            else:
                # Sell lower due to slippage
                execution_price = market_price * (1 - slippage)
            
            return execution_price
            
        elif order.order_type == OrderType.LIMIT:
            # Limit order at specified price (if market allows)
            if order.side == OrderSide.BUY and market_price <= order.price:
                return order.price
            elif order.side == OrderSide.SELL and market_price >= order.price:
                return order.price
            else:
                # Market hasn't reached limit price yet
                return None
                
        else:
            # Stop loss and other order types
            return order.price or market_price
    
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
        return f"MOCK_{uuid.uuid4().hex[:8]}"
    
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