#!/usr/bin/env python3
"""
High Frequency Trading System - Main Application
Orchestrates the complete HFT pipeline with real-time execution
"""

import asyncio
import signal
import sys
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import psutil
import os

# Import HFT components
from utils.logger import setup_logging, get_hft_logger
from market_data.coinapi_feed_handler import CoinAPIFeedHandler
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from execution.real_demo_order_manager import RealDemoOrderManager, OrderSide
from risk.risk_engine import RiskEngine
from portfolio.portfolio_tracker import PortfolioTracker


class HFTSystem:
    """Main HFT System orchestrator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(config_path)
        self.hft_logger = get_hft_logger()  # Get the HFTLogger instance
        self.logger = self.hft_logger.get_logger("hft_system")  # Get BoundLogger for regular logging
        
        # Initialize components
        self.feed_handler = None
        self.order_manager = None
        self.risk_engine = None
        self.strategies = {}
        self.portfolio_tracker = None
        
        # System state
        self.running = False
        self.start_time = None
        self.last_heartbeat = time.time()
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        
        # Initialize all components
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Error: Configuration file {config_path} not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Initialize all HFT system components"""
        try:
            # Initialize market data feed handler
            self.feed_handler = CoinAPIFeedHandler(self.config)
            
            # Initialize order manager
            self.order_manager = RealDemoOrderManager(self.config)
            
            # Initialize risk engine
            self.risk_engine = RiskEngine(self.config)
            
            # Initialize portfolio tracker
            initial_balance = self.config.get('trading', {}).get('initial_balance', 50000000.0)
            self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)
            
            # Initialize strategies
            if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', False):
                self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
                self.logger.info("Mean reversion strategy initialized")
            
            if self.config.get('strategies', {}).get('momentum', {}).get('enabled', False):
                self.strategies['momentum'] = MomentumStrategy(self.config)
                self.logger.info("Momentum strategy initialized")
            
            # Setup callbacks
            self._setup_callbacks()
            
            self.logger.info("All HFT components initialized successfully")
            
        except Exception as e:
            self.logger.log_error("initialize_components", e)
            sys.exit(1)
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        # Market data callbacks
        self.feed_handler.add_tick_callback(self._on_tick_data)
        self.feed_handler.add_kline_callback(self._on_kline_data)
        self.feed_handler.add_orderbook_callback(self._on_orderbook_data)
        
        # Order manager callbacks
        self.order_manager.add_fill_callback(self._on_order_fill)
        self.order_manager.add_order_update_callback(self._on_order_update)
    
    async def start(self):
        """Start the HFT system"""
        try:
            self.logger.info("Starting HFT System...")
            self.running = True
            self.start_time = datetime.now()
            
            # Start market data feeds
            self.feed_handler.start_streams()
            self.logger.info("Market data streams started")
            
            # Start main trading loop
            await asyncio.gather(
                self._trading_loop(),
                self._risk_monitoring_loop(),
                self._performance_monitoring_loop(),
                self._heartbeat_loop()
            )
            
        except Exception as e:
            self.logger.log_error("start_system", e)
            await self.stop()
    
    async def stop(self):
        """Stop the HFT system"""
        self.logger.info("Stopping HFT System...")
        self.running = False
        
        # Stop market data feeds
        if self.feed_handler:
            self.feed_handler.stop_streams()
        
        # Cancel pending orders
        if self.order_manager:
            pending_orders = self.order_manager.get_pending_orders()
            for order in pending_orders:
                await self.order_manager.cancel_order(order.order_id)
        
        # Print final statistics
        self._print_final_statistics()
        
        self.logger.info("HFT System stopped")
    
    async def _trading_loop(self):
        """Main trading loop - processes signals and executes trades"""
        self.logger.info("Trading loop started")
        
        while self.running:
            try:
                # Check each symbol for trading opportunities
                symbols = self.config['trading']['symbols']
                
                for symbol in symbols:
                    await self._process_symbol(symbol)
                
                # Sleep briefly to prevent excessive CPU usage
                await asyncio.sleep(0.1)  # 100ms
                
            except Exception as e:
                self.logger.log_error("trading_loop", e)
                await asyncio.sleep(1.0)  # Longer sleep on error
    
    async def _process_symbol(self, symbol: str):
        """Process trading signals for a specific symbol"""
        try:
            # Get latest market data
            ohlcv_data = self.feed_handler.get_ohlcv_data(symbol, "1m", 100)
            
            if ohlcv_data.empty:
                return
            
            current_price = self.feed_handler.get_latest_price(symbol)
            if not current_price:
                return
            
            # Update price history for strategies
            for strategy in self.strategies.values():
                if hasattr(strategy, 'update_price_history'):
                    strategy.update_price_history(symbol, current_price)
            
            # Update portfolio tracker with current prices
            if self.portfolio_tracker:
                self.portfolio_tracker.update_market_prices({symbol: current_price})
            
            # Generate signals from each strategy
            signals = []
            for strategy_name, strategy in self.strategies.items():
                signal = strategy.analyze_market_data(symbol, ohlcv_data)
                if signal and signal.signal_type != "HOLD":
                    signals.append((strategy_name, signal))
            
            # Process signals
            for strategy_name, signal in signals:
                await self._process_signal(strategy_name, signal)
                
        except Exception as e:
            self.logger.log_error("process_symbol", e, {"symbol": symbol})
    
    async def _process_signal(self, strategy_name: str, signal: Signal):
        """Process a trading signal through risk management and execution"""
        try:
            self.signals_generated += 1
            
            # Calculate position size
            available_capital = self.risk_engine.available_capital
            strategy = self.strategies[strategy_name]
            quantity = strategy.calculate_position_size(signal.symbol, signal, available_capital)
            
            if quantity <= 0:
                self.logger.warning(f"Invalid quantity calculated: {quantity}")
                return
            
            # Convert signal to order side
            order_side = OrderSide.BUY if signal.signal_type == "BUY" else OrderSide.SELL
            
            # Perform pre-trade risk check
            risk_check = self.risk_engine.perform_pre_trade_check(
                signal.symbol, signal.signal_type, quantity, signal.price, strategy_name
            )
            
            if not risk_check.passed:
                self.logger.warning(
                    f"Risk check failed for {signal.symbol}: {risk_check.message}",
                    risk_level=risk_check.risk_level
                )
                return
            
            # Execute trade
            await self._execute_trade(signal, order_side, quantity, strategy_name)
            
        except Exception as e:
            self.logger.log_error("process_signal", e, {
                "strategy": strategy_name,
                "signal": signal.__dict__ if signal else None
            })
    
    async def _execute_trade(self, signal: Signal, order_side: OrderSide, 
                           quantity: float, strategy_name: str):
        """Execute a trade based on signal"""
        try:
            # Submit market order for immediate execution
            order = await self.order_manager.submit_market_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=quantity,
                strategy=strategy_name,
                metadata={
                    'signal_strength': signal.strength,
                    'signal_metadata': signal.metadata
                }
            )
            
            self.trades_executed += 1
            
            self.logger.info(
                f"Trade executed: {order.symbol} {order.side.value} {order.quantity}",
                strategy=strategy_name,
                signal_strength=signal.strength,
                order_id=order.order_id
            )
            
        except Exception as e:
            self.logger.log_error("execute_trade", e, {
                "symbol": signal.symbol,
                "side": order_side.value,
                "quantity": quantity,
                "strategy": strategy_name
            })
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop"""
        self.logger.info("Risk monitoring loop started")
        
        while self.running:
            try:
                # Get current market prices
                symbols = self.config['trading']['symbols']
                market_data = {}
                
                for symbol in symbols:
                    price = self.feed_handler.get_latest_price(symbol)
                    if price:
                        market_data[symbol] = price
                
                # Monitor positions
                if market_data:
                    risk_checks = self.risk_engine.monitor_positions(market_data)
                    
                    for risk_check in risk_checks:
                        if not risk_check.passed and risk_check.risk_level in ["HIGH", "CRITICAL"]:
                            await self._handle_risk_violation(risk_check)
                
                # Check for daily loss limits
                if self.risk_engine.daily_pnl < 0:
                    daily_loss_ratio = abs(self.risk_engine.daily_pnl) / self.risk_engine.portfolio_value
                    if daily_loss_ratio > self.risk_engine.max_daily_loss:
                        self.risk_engine.halt_trading("Daily loss limit exceeded")
                        self.logger.warning("Trading halted due to daily loss limit")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.log_error("risk_monitoring_loop", e)
                await asyncio.sleep(10.0)
    
    async def _handle_risk_violation(self, risk_check):
        """Handle risk violations by closing positions or halting trading"""
        try:
            details = risk_check.details
            
            if details.get('action_required') == 'CLOSE_POSITION':
                symbol = details.get('symbol')
                if symbol and symbol in self.risk_engine.positions:
                    position = self.risk_engine.positions[symbol]
                    
                    # Close position with market order
                    close_side = OrderSide.SELL if position.side == "BUY" else OrderSide.BUY
                    
                    await self.order_manager.submit_market_order(
                        symbol=symbol,
                        side=close_side,
                        quantity=position.quantity,
                        strategy="risk_management",
                        metadata={'reason': risk_check.message}
                    )
                    
                    self.logger.warning(f"Position closed due to risk violation: {symbol}")
            
            elif details.get('action_required') == 'HALT_TRADING':
                self.risk_engine.halt_trading(risk_check.message)
                self.logger.critical(f"Trading halted: {risk_check.message}")
            
        except Exception as e:
            self.logger.log_error("handle_risk_violation", e)
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance metrics"""
        self.logger.info("Performance monitoring loop started")
        
        while self.running:
            try:
                # System metrics
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                # Feed handler performance
                feed_stats = self.feed_handler.get_performance_stats()
                
                # Log performance metrics
                self.logger.log_performance(
                    latency_ms=0,  # Would need actual latency measurement
                    memory_mb=psutil.virtual_memory().used / 1024 / 1024,
                    cpu_percent=cpu_usage
                )
                
                # Check performance thresholds
                monitoring_config = self.config.get('monitoring', {})
                if memory_usage > monitoring_config.get('memory_threshold_mb', 500):
                    self.logger.warning(f"High memory usage: {memory_usage}%")
                
                if cpu_usage > monitoring_config.get('cpu_threshold_percent', 80):
                    self.logger.warning(f"High CPU usage: {cpu_usage}%")
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.log_error("performance_monitoring_loop", e)
                await asyncio.sleep(60.0)
    
    async def _heartbeat_loop(self):
        """System heartbeat for monitoring system health"""
        while self.running:
            try:
                self.last_heartbeat = time.time()
                
                # Log system status every 5 minutes
                if int(self.last_heartbeat) % 300 == 0:
                    uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                    
                    self.logger.info(
                        "System heartbeat",
                        uptime_seconds=uptime.total_seconds(),
                        signals_generated=self.signals_generated,
                        trades_executed=self.trades_executed,
                        active_positions=len(self.risk_engine.positions),
                        portfolio_value=self.risk_engine.portfolio_value
                    )
                
                await asyncio.sleep(1.0)  # Heartbeat every second
                
            except Exception as e:
                self.logger.log_error("heartbeat_loop", e)
                await asyncio.sleep(5.0)
    
    def _on_tick_data(self, tick_data: Dict):
        """Callback for tick data updates"""
        # Update risk engine with latest prices
        symbol = tick_data['symbol']
        price = tick_data['price']
        
        # Update order manager with real market prices for execution
        if self.order_manager:
            self.order_manager.set_market_prices({symbol: price})
        
        # Update portfolio tracker with real prices
        if self.portfolio_tracker:
            self.portfolio_tracker.update_market_prices({symbol: price})
    
    def _on_kline_data(self, kline_data: Dict):
        """Callback for kline data updates"""
        # This is where real-time strategy calculations could be triggered
        pass
    
    def _on_orderbook_data(self, orderbook_data: Dict):
        """Callback for orderbook data updates"""
        symbol = orderbook_data['symbol']
        
        # Update order manager with real orderbook data for realistic execution
        if self.order_manager:
            self.order_manager.set_orderbook(symbol, orderbook_data)
    
    async def _on_order_fill(self, order):
        """Callback for order fills"""
        # Perform post-trade risk check
        risk_check = self.risk_engine.perform_post_trade_check(
            order.symbol, order.side.value, order.quantity, 
            order.average_price, order.order_id
        )
        
        # Record trade in portfolio tracker
        if self.portfolio_tracker:
            self.portfolio_tracker.add_trade(
                trade_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.average_price or order.price or 0,
                commission=order.commission,
                strategy=order.strategy,
                order_id=order.order_id
            )
        
        # Update total PnL tracking
        # This would need more sophisticated PnL calculation
        self.total_pnl += 0  # Placeholder
        
        self.logger.info(f"Order filled: {order.order_id}")
    
    async def _on_order_update(self, order):
        """Callback for order status updates"""
        pass
    
    def _print_final_statistics(self):
        """Print final system statistics"""
        if not self.start_time:
            return
        
        uptime = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("HFT SYSTEM - FINAL STATISTICS")
        print("="*60)
        print(f"Uptime: {uptime}")
        print(f"Signals Generated: {self.signals_generated}")
        print(f"Trades Executed: {self.trades_executed}")
        print(f"Total PnL: ${self.total_pnl:.2f}")
        
        if self.order_manager:
            order_stats = self.order_manager.get_order_statistics()
            print(f"Fill Rate: {order_stats['fill_rate_percentage']:.2f}%")
            print(f"Rejection Rate: {order_stats['rejection_rate_percentage']:.2f}%")
        
        if self.risk_engine:
            risk_stats = self.risk_engine.get_risk_statistics()
            print(f"Risk Checks Performed: {risk_stats['risk_checks_performed']}")
            print(f"Risk Check Success Rate: {risk_stats['risk_check_success_rate']:.2f}%")
            print(f"Final Portfolio Value: ${risk_stats['portfolio_value']:.2f}")
        
        if self.portfolio_tracker:
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            portfolio_metrics = portfolio_summary['overview']
            print(f"Portfolio Growth: {portfolio_metrics['portfolio_growth']:.2f}%")
            print(f"Total Unrealized P&L: ${portfolio_metrics['total_unrealized_pnl']:.2f}")
            print(f"Open Positions: {portfolio_metrics['position_count']}")
            print(f"Win Rate: {portfolio_metrics['win_rate']:.1f}%")
            
            # Save final portfolio snapshot
            snapshot_file = self.portfolio_tracker.save_snapshot()
            print(f"Portfolio snapshot saved: {snapshot_file}")
        
        print("="*60)


async def main():
    """Main application entry point"""
    # Setup signal handlers for graceful shutdown
    hft_system = None
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        if hft_system:
            asyncio.create_task(hft_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start HFT system
        hft_system = HFTSystem()
        print("HFT System initialized successfully")
        print("Starting trading operations...")
        print("Press Ctrl+C to stop")
        
        await hft_system.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        if hft_system and hft_system.logger:
            hft_system.logger.log_error("main", e)
    finally:
        if hft_system:
            await hft_system.stop()


if __name__ == "__main__":
    # Run the HFT system
    asyncio.run(main()) 