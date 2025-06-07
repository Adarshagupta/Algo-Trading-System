import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
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
    commission_asset: str = ""
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    binance_order_id: Optional[str] = None
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManager:
    """High-performance order management system for Binance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("order_manager")  # Get BoundLogger for regular logging
        
        # Initialize Binance client
        self.client = Client(
            api_key=config['binance']['api_key'],
            api_secret=config['binance']['api_secret'],
            testnet=config['binance'].get('testnet', True)
        )
        
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
        
        # Order execution settings
        self.max_retries = 3
        self.retry_delay = 0.1  # seconds
        
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
        
        try:
            # Cancel order on Binance
            if order.binance_order_id:
                result = self.client.cancel_order(
                    symbol=order.symbol,
                    orderId=order.binance_order_id
                )
                
                # Update order status
                order.status = OrderStatus.CANCELLED
                order.updated_time = datetime.now()
                
                # Remove from pending orders
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                
                # Execute callbacks
                await self._execute_order_update_callbacks(order)
                
                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            
        except BinanceAPIException as e:
            self.hft_logger.log_error("cancel_order", e, {
                "order_id": order_id,
                "binance_order_id": order.binance_order_id
            })
            return False
    
    async def _execute_order(self, order: Order) -> Order:
        """Execute order on Binance with retry logic"""
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        
        for attempt in range(self.max_retries):
            try:
                # Prepare order parameters
                order_params = self._prepare_order_params(order)
                
                # Submit order to Binance
                order.status = OrderStatus.SUBMITTED
                order.updated_time = datetime.now()
                
                start_time = time.time()
                result = self.client.create_order(**order_params)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # Update order with Binance response
                self._update_order_from_binance_response(order, result)
                
                self.orders_submitted += 1
                
                # Log successful submission
                self.hft_logger.log_trade(
                    order.symbol, order.side.value, order.quantity, 
                    order.price or 0, order.strategy
                )
                
                self.logger.info(
                    f"Order {order.order_id} submitted successfully",
                    execution_time_ms=execution_time,
                    binance_order_id=order.binance_order_id
                )
                
                # Start monitoring order status
                asyncio.create_task(self._monitor_order_status(order))
                
                # Execute callbacks
                await self._execute_order_update_callbacks(order)
                
                return order
                
            except (BinanceAPIException, BinanceOrderException) as e:
                self.hft_logger.log_error("execute_order", e, {
                    "order_id": order.order_id,
                    "attempt": attempt + 1,
                    "max_retries": self.max_retries
                })
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Mark order as rejected after all retries
                    order.status = OrderStatus.REJECTED
                    order.updated_time = datetime.now()
                    self.orders_rejected += 1
                    
                    # Remove from pending orders
                    if order.order_id in self.pending_orders:
                        del self.pending_orders[order.order_id]
                    
                    # Execute callbacks
                    await self._execute_order_update_callbacks(order)
        
        return order
    
    def _prepare_order_params(self, order: Order) -> Dict[str, Any]:
        """Prepare order parameters for Binance API"""
        params = {
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': order.quantity,
            'timeInForce': order.time_in_force
        }
        
        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            if order.price is not None:
                params['price'] = str(order.price)
        
        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            if order.stop_price is not None:
                params['stopPrice'] = str(order.stop_price)
        
        # Add client order ID for tracking
        params['newClientOrderId'] = order.order_id
        
        return params
    
    def _update_order_from_binance_response(self, order: Order, response: Dict):
        """Update order object from Binance API response"""
        order.binance_order_id = str(response.get('orderId', ''))
        order.filled_quantity = float(response.get('executedQty', 0))
        order.status = self._map_binance_status(response.get('status', ''))
        
        # Update price information
        if 'price' in response and response['price'] != '0.00000000':
            order.price = float(response['price'])
        
        # Calculate average price for filled orders
        if order.filled_quantity > 0 and 'cummulativeQuoteQty' in response:
            cumulative_quote = float(response['cummulativeQuoteQty'])
            if cumulative_quote > 0:
                order.average_price = cumulative_quote / order.filled_quantity
    
    def _map_binance_status(self, binance_status: str) -> OrderStatus:
        """Map Binance order status to internal status"""
        status_mapping = {
            'NEW': OrderStatus.SUBMITTED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return status_mapping.get(binance_status, OrderStatus.PENDING)
    
    async def _monitor_order_status(self, order: Order):
        """Monitor order status until completion"""
        max_checks = 60  # Maximum status checks
        check_interval = 1.0  # seconds
        
        for _ in range(max_checks):
            try:
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                  OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    break
                
                await asyncio.sleep(check_interval)
                
                # Query order status from Binance
                if order.binance_order_id:
                    result = self.client.get_order(
                        symbol=order.symbol,
                        orderId=order.binance_order_id
                    )
                    
                    old_status = order.status
                    self._update_order_from_binance_response(order, result)
                    order.updated_time = datetime.now()
                    
                    # Check if status changed
                    if order.status != old_status:
                        await self._execute_order_update_callbacks(order)
                        
                        # Handle fills
                        if order.status == OrderStatus.FILLED:
                            await self._handle_order_fill(order)
                        elif order.status == OrderStatus.PARTIALLY_FILLED:
                            await self._handle_partial_fill(order)
                
            except Exception as e:
                self.hft_logger.log_error("monitor_order_status", e, {
                    "order_id": order.order_id
                })
    
    async def _handle_order_fill(self, order: Order):
        """Handle completely filled order"""
        self.orders_filled += 1
        
        # Move to filled orders
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        self.filled_orders[order.order_id] = order
        
        # Update commission tracking
        if hasattr(order, 'commission'):
            self.total_commission += order.commission
        
        # Execute fill callbacks
        await self._execute_fill_callbacks(order)
        
        self.logger.info(
            f"Order {order.order_id} completely filled",
            symbol=order.symbol,
            quantity=order.quantity,
            average_price=order.average_price,
            strategy=order.strategy
        )
    
    async def _handle_partial_fill(self, order: Order):
        """Handle partially filled order"""
        self.logger.info(
            f"Order {order.order_id} partially filled",
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.quantity - order.filled_quantity
        )
    
    async def _execute_fill_callbacks(self, order: Order):
        """Execute all fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                self.hft_logger.log_error("fill_callback", e, {
                    "order_id": order.order_id
                })
    
    async def _execute_order_update_callbacks(self, order: Order):
        """Execute all order update callbacks"""
        for callback in self.order_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                self.hft_logger.log_error("order_update_callback", e, {
                    "order_id": order.order_id
                })
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"HFT_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
    
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
        total_orders = len(self.orders)
        fill_rate = (self.orders_filled / max(total_orders, 1)) * 100
        rejection_rate = (self.orders_rejected / max(total_orders, 1)) * 100
        
        return {
            'total_orders': total_orders,
            'orders_submitted': self.orders_submitted,
            'orders_filled': self.orders_filled,
            'orders_rejected': self.orders_rejected,
            'pending_orders': len(self.pending_orders),
            'fill_rate_percentage': round(fill_rate, 2),
            'rejection_rate_percentage': round(rejection_rate, 2),
            'total_commission': self.total_commission,
            'average_fill_time_ms': self._calculate_average_fill_time()
        }
    
    def _calculate_average_fill_time(self) -> float:
        """Calculate average fill time for completed orders"""
        fill_times = []
        
        for order in self.filled_orders.values():
            if order.created_time and order.updated_time:
                fill_time = (order.updated_time - order.created_time).total_seconds() * 1000
                fill_times.append(fill_time)
        
        return sum(fill_times) / len(fill_times) if fill_times else 0.0 