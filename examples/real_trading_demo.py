#!/usr/bin/env python3
"""
Real Trading Demo - $50M Portfolio with Real Market Data
Uses live CoinAPI.io market data with real-time order execution (demo money only)
"""

import asyncio
import os
import sys
import time
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data.coinapi_feed_handler import CoinAPIFeedHandler
from execution.real_demo_order_manager import RealDemoOrderManager, OrderSide
from portfolio.portfolio_tracker import PortfolioTracker
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from risk.risk_engine import RiskEngine
from utils.logger import setup_logging, get_hft_logger
import yaml


class RealTradingDemo:
    """Real market data trading demo with $50M portfolio"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(config_path)
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("real_trading_demo")
        
        # Initialize components
        self.feed_handler = None
        self.order_manager = None
        self.portfolio_tracker = None
        self.risk_engine = None
        self.strategies = {}
        
        # Demo state
        self.running = False
        self.real_trades_executed = 0
        self.start_time = None
        
        print("🚀 REAL TRADING DEMO")
        print("=" * 50)
        print("✅ Real market data from CoinAPI.io")
        print("✅ Real-time price feeds")
        print("✅ Live orderbook data")
        print("✅ Realistic order execution")
        print("💰 Demo money only - NO REAL TRADING")
        print(f"💼 Portfolio: ${self.config['trading']['initial_balance']:,.0f}")
        print("=" * 50)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️  Configuration file {config_path} not found - using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'coinapi': {
                'api_key': '3452f6ef-aef6-47f5-afda-31c5c1801a90',
                'base_url': 'https://rest.coinapi.io',
                'websocket_url': 'wss://ws.coinapi.io/v1/',
                'sandbox': False
            },
            'trading': {
                'symbols': ['COINBASE_SPOT_BTC_USD', 'COINBASE_SPOT_ETH_USD', 'COINBASE_SPOT_ADA_USD'],
                'timeframes': ['1MIN', '5MIN', '15MIN'],
                'initial_balance': 50000000.0,
                'max_positions': 8,
                'base_currency': 'USD'
            },
            'strategies': {
                'mean_reversion': {'enabled': True, 'position_size': 0.03},
                'momentum': {'enabled': True, 'position_size': 0.05}
            },
            'risk': {
                'max_portfolio_risk': 0.01,
                'max_daily_loss': 0.02,
                'stop_loss': 0.015,
                'take_profit': 0.03
            }
        }
    
    async def initialize_system(self):
        """Initialize all system components"""
        print("\n🔧 Initializing Real Trading System...")
        
        try:
            # Initialize CoinAPI.io feed handler
            print("📡 Connecting to CoinAPI.io...")
            self.feed_handler = CoinAPIFeedHandler(self.config)
            
            # Initialize real demo order manager
            print("💼 Initializing real demo order manager...")
            self.order_manager = RealDemoOrderManager(self.config)
            
            # Initialize portfolio tracker
            print("📊 Setting up $50M portfolio tracker...")
            initial_balance = self.config['trading']['initial_balance']
            self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)
            
            # Initialize risk engine
            print("⚠️  Configuring risk management...")
            self.risk_engine = RiskEngine(self.config)
            
            # Initialize trading strategies
            print("🎯 Loading trading strategies...")
            if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', False):
                self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
                print("  ✅ Mean reversion strategy loaded")
            
            if self.config.get('strategies', {}).get('momentum', {}).get('enabled', False):
                self.strategies['momentum'] = MomentumStrategy(self.config)
                print("  ✅ Momentum strategy loaded")
            
            # Setup real-time data callbacks
            self._setup_real_data_callbacks()
            
            print("✅ System initialization complete!")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            raise
    
    def _setup_real_data_callbacks(self):
        """Setup callbacks for real market data"""
        # Real market data callbacks
        self.feed_handler.add_tick_callback(self._on_real_tick_data)
        self.feed_handler.add_orderbook_callback(self._on_real_orderbook_data)
        self.feed_handler.add_kline_callback(self._on_real_kline_data)
        
        # Real order execution callbacks
        self.order_manager.add_fill_callback(self._on_real_order_fill)
        self.order_manager.add_order_update_callback(self._on_real_order_update)
    
    async def _on_real_tick_data(self, tick_data: dict):
        """Handle real tick data from CoinAPI.io"""
        symbol = tick_data['symbol']
        price = tick_data['price']
        
        # Update order manager with real market prices
        if self.order_manager:
            self.order_manager.set_market_prices({symbol: price})
        
        # Update portfolio with real prices
        if self.portfolio_tracker:
            self.portfolio_tracker.update_market_prices({symbol: price})
    
    async def _on_real_orderbook_data(self, orderbook_data: dict):
        """Handle real orderbook data from CoinAPI.io"""
        symbol = orderbook_data['symbol']
        
        # Update order manager with real orderbook for realistic execution
        if self.order_manager:
            self.order_manager.set_orderbook(symbol, orderbook_data)
    
    async def _on_real_kline_data(self, kline_data: dict):
        """Handle real OHLCV data from CoinAPI.io"""
        pass
    
    async def _on_real_order_fill(self, order):
        """Handle real order fills"""
        # Record trade in portfolio tracker
        if self.portfolio_tracker:
            self.portfolio_tracker.add_trade(
                trade_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                price=order.average_price or 0,
                commission=order.commission,
                strategy=order.strategy,
                order_id=order.order_id
            )
        
        self.real_trades_executed += 1
        
        print(f"💰 REAL EXECUTION: {order.symbol} {order.side.value} {order.quantity:.6f} @ ${order.average_price:.2f}")
        print(f"   Commission: ${order.commission:.2f} | Strategy: {order.strategy}")
    
    async def _on_real_order_update(self, order):
        """Handle real order updates"""
        if order.status.value in ['SUBMITTED', 'FILLED', 'REJECTED']:
            print(f"📋 Order {order.order_id}: {order.status.value}")
    
    async def start_real_data_feeds(self):
        """Start real market data feeds"""
        print("\n📡 Starting real-time market data feeds...")
        
        try:
            # Start CoinAPI.io streams
            self.feed_handler.start_streams()
            
            # Wait for data to start flowing
            print("⏳ Waiting for market data...")
            await asyncio.sleep(5)
            
            # Verify real data is flowing
            symbols = self.config['trading']['symbols']
            live_data_count = 0
            
            print("\n📊 Live Market Data:")
            for symbol in symbols:
                price = self.feed_handler.get_latest_price(symbol)
                if price:
                    live_data_count += 1
                    print(f"  {symbol}: ${price:,.2f}")
                else:
                    print(f"  {symbol}: Waiting for data...")
            
            if live_data_count > 0:
                print(f"✅ {live_data_count}/{len(symbols)} symbols streaming live data")
            else:
                print("⚠️  No live data yet - continuing to wait...")
            
        except Exception as e:
            print(f"❌ Failed to start market data: {e}")
            raise
    
    async def run_real_trading_demo(self, duration_minutes: int = 10):
        """Run real trading demo with live market data"""
        print(f"\n🎯 Starting REAL TRADING DEMO for {duration_minutes} minutes")
        print("Using live market data for all decisions!")
        
        self.running = True
        self.start_time = datetime.now()
        end_time = time.time() + (duration_minutes * 60)
        
        last_strategy_run = 0
        strategy_interval = 20  # Run strategies every 20 seconds
        
        last_status_update = 0
        status_interval = 30  # Status update every 30 seconds
        
        while self.running and time.time() < end_time:
            try:
                current_time = time.time()
                
                # Run trading strategies with real data
                if current_time - last_strategy_run >= strategy_interval:
                    await self._run_real_strategies()
                    last_strategy_run = current_time
                
                # Show real-time status
                if current_time - last_status_update >= status_interval:
                    await self._show_real_status()
                    last_status_update = current_time
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n🛑 Real trading demo stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in real trading: {e}")
                await asyncio.sleep(5)
        
        self.running = False
        print(f"\n✅ Real trading demo completed!")
        await self._show_final_real_results()
    
    async def _run_real_strategies(self):
        """Run strategies with real market data"""
        try:
            symbols = self.config['trading']['symbols']
            
            for symbol in symbols:
                # Get real market data
                current_price = self.feed_handler.get_latest_price(symbol)
                ohlcv_data = self.feed_handler.get_ohlcv_data(symbol, "1MIN", 50)
                
                if current_price is None or ohlcv_data.empty:
                    continue
                
                print(f"📈 Analyzing {symbol} @ ${current_price:.2f} with {len(ohlcv_data)} real candles")
                
                # Run each strategy with real data
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.analyze_market_data(symbol, ohlcv_data)
                        
                        if signal and signal.signal_type != "HOLD":
                            await self._execute_real_signal(strategy_name, signal, current_price)
                            
                    except Exception as e:
                        print(f"❌ Strategy error ({strategy_name}): {e}")
                        
        except Exception as e:
            print(f"❌ Error running real strategies: {e}")
    
    async def _execute_real_signal(self, strategy_name: str, signal, current_price: float):
        """Execute trading signal with real market data"""
        try:
            # Calculate position size using real portfolio value
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            portfolio_value = portfolio_summary['overview']['total_value']
            
            position_size_ratio = self.config['strategies'][strategy_name]['position_size']
            position_value = portfolio_value * position_size_ratio
            
            # Calculate quantity based on real current price
            quantity = position_value / current_price
            
            # Submit order using real market data
            side = OrderSide.BUY if signal.signal_type == "BUY" else OrderSide.SELL
            
            print(f"📤 REAL SIGNAL: {strategy_name} {signal.signal_type} {signal.symbol}")
            print(f"   Position Value: ${position_value:,.0f} | Quantity: {quantity:.6f}")
            
            order = await self.order_manager.submit_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                strategy=strategy_name,
                metadata={'signal_strength': signal.strength, 'current_price': current_price}
            )
            
        except Exception as e:
            print(f"❌ Error executing real signal: {e}")
    
    async def _show_real_status(self):
        """Show real-time status with live data"""
        try:
            if not self.portfolio_tracker:
                return
            
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n📊 REAL-TIME STATUS (Live Data):")
            print(f"  Portfolio Value: ${overview['total_value']:,.0f}")
            print(f"  Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"  Live P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"  Real Trades: {self.real_trades_executed}")
            print(f"  Open Positions: {overview['position_count']}")
            
            # Show live market prices
            symbols = self.config['trading']['symbols']
            print(f"  Live Prices:")
            for symbol in symbols:
                price = self.feed_handler.get_latest_price(symbol)
                if price:
                    print(f"    {symbol}: ${price:,.2f}")
            
        except Exception as e:
            print(f"❌ Error showing real status: {e}")
    
    async def _show_final_real_results(self):
        """Show final results with real data analysis"""
        try:
            if not self.portfolio_tracker:
                return
            
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n🎉 FINAL REAL TRADING RESULTS")
            print(f"=" * 60)
            print(f"✅ All data was REAL from CoinAPI.io")
            print(f"✅ All prices were LIVE market data")
            print(f"✅ All execution used REAL orderbooks")
            print(f"💰 Demo money only - NO REAL FUNDS USED")
            print(f"=" * 60)
            print(f"Initial Balance: ${self.config['trading']['initial_balance']:,.0f}")
            print(f"Final Value: ${overview['total_value']:,.0f}")
            print(f"Total Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"Real P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"Real Trades Executed: {self.real_trades_executed}")
            print(f"Win Rate: {overview['win_rate']:.1f}%")
            
            # Show final positions with real prices
            if summary['positions']:
                print(f"\n📈 Final Positions (Real Market Values):")
                for pos in summary['positions']:
                    current_price = self.feed_handler.get_latest_price(pos['symbol'])
                    pnl_symbol = "+" if pos['unrealized_pnl'] >= 0 else ""
                    print(f"  {pos['symbol']}: {pnl_symbol}${pos['unrealized_pnl']:,.0f} ({pos['unrealized_pnl_percent']:+.1f}%) @ ${current_price:.2f}")
            
            # Export real trading data
            json_file = self.portfolio_tracker.export_data('json')
            print(f"\n💾 Real trading data exported: {json_file}")
            
            # Show order execution statistics
            if self.order_manager:
                order_stats = self.order_manager.get_order_statistics()
                print(f"\n📋 Real Execution Statistics:")
                print(f"  Orders Submitted: {order_stats['total_orders_submitted']}")
                print(f"  Fill Rate: {order_stats['fill_rate_percentage']:.1f}%")
                print(f"  Total Commission: ${order_stats['total_commission']:.2f}")
                print(f"  Avg Fill Time: {order_stats['average_fill_time_seconds']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error showing final results: {e}")
    
    def stop(self):
        """Stop the real trading demo"""
        self.running = False
        if self.feed_handler:
            self.feed_handler.stop_streams()


async def main():
    """Main real trading demo"""
    parser = argparse.ArgumentParser(description='Real Trading Demo with Live Market Data')
    parser.add_argument('--duration', type=int, default=10, 
                       help='Demo duration in minutes (default: 10)')
    
    args = parser.parse_args()
    
    print("🚀 REAL TRADING DEMO")
    print("Using LIVE market data from CoinAPI.io")
    print("Real-time prices, orderbooks, and execution")
    print("Demo money only - completely safe!")
    
    demo = RealTradingDemo()
    
    try:
        # Initialize system
        await demo.initialize_system()
        
        # Start real data feeds
        await demo.start_real_data_feeds()
        
        # Run real trading demo
        await demo.run_real_trading_demo(args.duration)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        demo.stop()
        print("\n👋 Real trading demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 