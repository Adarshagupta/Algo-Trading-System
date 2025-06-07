#!/usr/bin/env python3
"""
CoinAPI.io Demo - Complete HFT System with Real-time Market Data
Demonstrates the full system with $50M demo portfolio using live CoinAPI.io data
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
from portfolio.portfolio_tracker import PortfolioTracker
from portfolio.portfolio_cli import PortfolioCLI
from portfolio.web_dashboard import PortfolioWebDashboard, create_html_template
from execution.real_demo_order_manager import RealDemoOrderManager
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from risk.risk_engine import RiskEngine
from utils.logger import setup_logging, get_hft_logger
import yaml


class CoinAPIDemoSystem:
    """Complete demo system with CoinAPI.io integration"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(config_path)
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("coinapi_demo")
        
        # Initialize components
        self.feed_handler = None
        self.order_manager = None
        self.portfolio_tracker = None
        self.risk_engine = None
        self.strategies = {}
        
        # Demo state
        self.running = False
        self.trades_executed = 0
        self.start_time = None
        
        print(f"🚀 CoinAPI.io Demo System Initialized")
        print(f"📊 Portfolio Balance: ${self.config['trading']['initial_balance']:,.0f}")
        print(f"🔗 Using CoinAPI.io for real-time market data")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Configuration file {config_path} not found!")
            # Return default configuration
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
                    'mean_reversion': {'enabled': True, 'position_size': 0.05},
                    'momentum': {'enabled': True, 'position_size': 0.08}
                },
                'risk': {
                    'max_portfolio_risk': 0.01,
                    'max_daily_loss': 0.02,
                    'stop_loss': 0.015,
                    'take_profit': 0.03
                }
            }
    
    async def initialize_components(self):
        """Initialize all system components"""
        print("\n🔧 Initializing Components...")
        
        try:
            # Initialize market data feed handler
            print("📡 Initializing CoinAPI.io feed handler...")
            self.feed_handler = CoinAPIFeedHandler(self.config)
            
            # Initialize portfolio tracker
            print("💼 Initializing portfolio tracker...")
            initial_balance = self.config['trading']['initial_balance']
            self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)
            
            # Initialize order manager
            print("📋 Initializing real demo order manager...")
            self.order_manager = RealDemoOrderManager(self.config)
            
            # Initialize risk engine
            print("⚠️  Initializing risk engine...")
            self.risk_engine = RiskEngine(self.config)
            
            # Initialize strategies
            print("🎯 Initializing trading strategies...")
            if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', False):
                self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
                print("  ✅ Mean reversion strategy")
            
            if self.config.get('strategies', {}).get('momentum', {}).get('enabled', False):
                self.strategies['momentum'] = MomentumStrategy(self.config)
                print("  ✅ Momentum strategy")
            
            # Setup callbacks
            self._setup_callbacks()
            
            print("✅ All components initialized successfully!")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            raise
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        # Market data callbacks
        self.feed_handler.add_tick_callback(self._on_tick_data)
        self.feed_handler.add_kline_callback(self._on_kline_data)
        self.feed_handler.add_orderbook_callback(self._on_orderbook_data)
        
        # Order manager callbacks
        self.order_manager.add_fill_callback(self._on_order_fill)
        self.order_manager.add_order_update_callback(self._on_order_update)
    
    async def _on_tick_data(self, tick_data: dict):
        """Handle tick data updates"""
        symbol = tick_data['symbol']
        price = tick_data['price']
        
        # Update order manager with market prices
        if self.order_manager:
            self.order_manager.set_market_prices({symbol: price})
        
        # Update portfolio tracker
        if self.portfolio_tracker:
            self.portfolio_tracker.update_market_prices({symbol: price})
    
    async def _on_kline_data(self, kline_data: dict):
        """Handle kline data updates"""
        pass
    
    async def _on_orderbook_data(self, orderbook_data: dict):
        """Handle orderbook data updates"""
        symbol = orderbook_data['symbol']
        
        # Update order manager with real orderbook data for realistic execution
        if self.order_manager:
            self.order_manager.set_orderbook(symbol, orderbook_data)
    
    async def _on_order_fill(self, order):
        """Handle order fills"""
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
        
        self.trades_executed += 1
        print(f"💰 Trade executed: {order.symbol} {order.side.value} {order.quantity:.6f} @ ${order.average_price:.2f}")
    
    async def _on_order_update(self, order):
        """Handle order updates"""
        pass
    
    async def start_market_data(self):
        """Start market data feeds"""
        print("\n📡 Starting CoinAPI.io market data feeds...")
        try:
            self.feed_handler.start_streams()
            
            # Wait a moment for data to start flowing
            await asyncio.sleep(3)
            
            # Check if we're receiving data
            symbols = self.config['trading']['symbols']
            prices_received = 0
            
            for symbol in symbols:
                price = self.feed_handler.get_latest_price(symbol)
                if price:
                    prices_received += 1
                    print(f"  📊 {symbol}: ${price:,.2f}")
            
            if prices_received > 0:
                print(f"✅ Market data streaming - {prices_received}/{len(symbols)} symbols active")
            else:
                print("⚠️  No market data received yet - this may take a moment...")
            
        except Exception as e:
            print(f"❌ Failed to start market data: {e}")
            raise
    
    async def run_demo_trading(self, duration_minutes: int = 5):
        """Run demo trading for specified duration"""
        print(f"\n🎯 Starting demo trading for {duration_minutes} minutes...")
        
        self.running = True
        self.start_time = datetime.now()
        end_time = time.time() + (duration_minutes * 60)
        
        last_trade_time = 0
        trade_interval = 30  # Try to trade every 30 seconds
        
        while self.running and time.time() < end_time:
            try:
                current_time = time.time()
                
                # Check if it's time for a trade
                if current_time - last_trade_time >= trade_interval:
                    await self._attempt_demo_trade()
                    last_trade_time = current_time
                
                # Show periodic updates
                elapsed = current_time - (end_time - duration_minutes * 60)
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    await self._show_demo_status()
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n🛑 Demo trading stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in demo trading: {e}")
                await asyncio.sleep(5)
        
        self.running = False
        print(f"\n✅ Demo trading completed!")
        await self._show_final_results()
    
    async def _attempt_demo_trade(self):
        """Attempt to execute a demo trade"""
        try:
            symbols = self.config['trading']['symbols']
            
            for symbol in symbols:
                # Get market data
                price = self.feed_handler.get_latest_price(symbol)
                ohlcv_data = self.feed_handler.get_ohlcv_data(symbol, "1MIN", 50)
                
                if price is None or ohlcv_data.empty:
                    continue
                
                # Try each strategy
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.analyze_market_data(symbol, ohlcv_data)
                        
                        if signal and signal.signal_type != "HOLD":
                            await self._execute_demo_signal(strategy_name, signal)
                            
                    except Exception as e:
                        self.logger.warning(f"Strategy error: {strategy_name} - {e}")
                        
        except Exception as e:
            print(f"❌ Error attempting demo trade: {e}")
    
    async def _execute_demo_signal(self, strategy_name: str, signal):
        """Execute a trading signal"""
        try:
            # Calculate position size (conservative for demo)
            portfolio_value = self.portfolio_tracker.cash_balance + sum(
                pos.market_value for pos in self.portfolio_tracker.positions.values()
            )
            
            position_size_ratio = self.config['strategies'][strategy_name]['position_size']
            position_value = portfolio_value * position_size_ratio
            
            # Calculate quantity based on current price
            current_price = self.feed_handler.get_latest_price(signal.symbol)
            if not current_price:
                return
            
            quantity = position_value / current_price
            
            # Submit order
            from execution.real_demo_order_manager import OrderSide
            side = OrderSide.BUY if signal.signal_type == "BUY" else OrderSide.SELL
            
            order = await self.order_manager.submit_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                strategy=strategy_name,
                metadata={'signal_strength': signal.strength}
            )
            
            print(f"📤 {strategy_name}: {signal.signal_type} {signal.symbol} - ${position_value:,.0f}")
            
        except Exception as e:
            print(f"❌ Error executing signal: {e}")
    
    async def _show_demo_status(self):
        """Show current demo status"""
        if not self.portfolio_tracker:
            return
        
        try:
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n📊 Demo Status:")
            print(f"  Portfolio Value: ${overview['total_value']:,.0f}")
            print(f"  Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"  Open Positions: {overview['position_count']}")
            print(f"  Trades Executed: {self.trades_executed}")
            
        except Exception as e:
            print(f"❌ Error showing status: {e}")
    
    async def _show_final_results(self):
        """Show final demo results"""
        if not self.portfolio_tracker:
            return
        
        try:
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n🎉 FINAL DEMO RESULTS")
            print(f"=" * 50)
            print(f"Initial Balance: ${self.config['trading']['initial_balance']:,.0f}")
            print(f"Final Value: ${overview['total_value']:,.0f}")
            print(f"Total Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"Total P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"Trades Executed: {self.trades_executed}")
            print(f"Open Positions: {overview['position_count']}")
            print(f"Win Rate: {overview['win_rate']:.1f}%")
            
            # Show positions
            if summary['positions']:
                print(f"\n📈 Open Positions:")
                for pos in summary['positions']:
                    pnl_symbol = "+" if pos['unrealized_pnl'] >= 0 else ""
                    print(f"  {pos['symbol']}: {pnl_symbol}${pos['unrealized_pnl']:,.0f} ({pos['unrealized_pnl_percent']:+.1f}%)")
            
            # Export data
            json_file = self.portfolio_tracker.export_data('json')
            print(f"\n💾 Data exported to: {json_file}")
            
        except Exception as e:
            print(f"❌ Error showing final results: {e}")
    
    def stop(self):
        """Stop the demo system"""
        self.running = False
        if self.feed_handler:
            self.feed_handler.stop_streams()


async def run_cli_demo(demo_system):
    """Run the CLI portfolio interface"""
    print("\n🖥️  Starting CLI Portfolio Interface...")
    print("Press Ctrl+C to exit the CLI")
    
    cli = PortfolioCLI(demo_system.portfolio_tracker)
    
    try:
        cli.run_live_display(update_interval=2.0)
    except KeyboardInterrupt:
        print("\n✅ CLI demo completed")


async def run_web_demo(demo_system, host='localhost', port=5000):
    """Run the web dashboard"""
    print(f"\n🌐 Starting Web Dashboard at http://{host}:{port}")
    print("Open your browser and navigate to the URL above")
    print("Press Ctrl+C to stop the web server")
    
    # Create HTML template if needed
    if not os.path.exists('portfolio/templates/dashboard.html'):
        create_html_template()
    
    dashboard = PortfolioWebDashboard(demo_system.portfolio_tracker, host=host, port=port)
    
    try:
        dashboard.start(debug=False)
    except KeyboardInterrupt:
        print("\n✅ Web dashboard demo completed")
        dashboard.stop()


async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='CoinAPI.io HFT Demo System')
    parser.add_argument('--mode', choices=['trading', 'cli', 'web', 'full'], 
                       default='trading', help='Demo mode to run')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Trading demo duration in minutes')
    parser.add_argument('--host', default='localhost', 
                       help='Web dashboard host')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Web dashboard port')
    
    args = parser.parse_args()
    
    print("🚀 CoinAPI.io HFT Demo System")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"API Key: 3452f6ef-aef6-47f5-afda-31c5c1801a90")
    print("Real-time market data from CoinAPI.io")
    print("$50,000,000 demo portfolio")
    
    # Initialize demo system
    demo_system = CoinAPIDemoSystem()
    
    try:
        # Initialize components
        await demo_system.initialize_components()
        
        # Start market data
        await demo_system.start_market_data()
        
        # Add some initial demo trades
        await demo_system._attempt_demo_trade()
        
        # Run selected mode
        if args.mode == 'trading':
            await demo_system.run_demo_trading(args.duration)
            
        elif args.mode == 'cli':
            await run_cli_demo(demo_system)
            
        elif args.mode == 'web':
            await run_web_demo(demo_system, args.host, args.port)
            
        elif args.mode == 'full':
            # Run trading for a few minutes, then show interfaces
            print("\n🎯 Running full demo: trading + interfaces")
            await demo_system.run_demo_trading(3)  # 3 minutes of trading
            
            print("\nChoose interface:")
            print("1. CLI Interface")
            print("2. Web Dashboard")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                await run_cli_demo(demo_system)
            else:
                await run_web_demo(demo_system, args.host, args.port)
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        demo_system.stop()
        print("\n👋 Demo completed. Thank you!")


if __name__ == "__main__":
    asyncio.run(main()) 