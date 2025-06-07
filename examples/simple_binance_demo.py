#!/usr/bin/env python3
"""
Simple Binance Real Trading Demo - REST API Only
Uses Binance REST API for real market data with demo trading
"""

import asyncio
import os
import sys
import time
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance.client import Client
from execution.real_demo_order_manager import RealDemoOrderManager, OrderSide
from portfolio.portfolio_tracker import PortfolioTracker
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from utils.logger import setup_logging, get_hft_logger
import yaml
import pandas as pd


class SimpleBinanceDemo:
    """Simple real trading demo using Binance REST API"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(config_path)
        self.hft_logger = get_hft_logger()
        self.logger = self.hft_logger.get_logger("simple_binance_demo")
        
        # Initialize Binance client
        self.binance_client = Client(
            api_key=self.config['binance']['api_key'],
            api_secret=self.config['binance']['api_secret'],
            testnet=False
        )
        
        # Initialize components
        self.order_manager = None
        self.portfolio_tracker = None
        self.strategies = {}
        
        # Demo state
        self.running = False
        self.trades_executed = 0
        self.start_time = None
        
        print("🚀 SIMPLE BINANCE REAL TRADING DEMO")
        print("=" * 60)
        print("✅ Using Binance REST API for real market data")
        print("✅ Live price updates every few seconds") 
        print("✅ Real order execution simulation")
        print("🔒 Read-only API (no trading risk)")
        print("💰 Demo money only - NO REAL TRADING")
        print(f"💼 Portfolio: ${self.config['trading']['initial_balance']:,.0f}")
        print("=" * 60)
        
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
            'binance': {
                'api_key': 'GKEYgU4j5FdiCx10Vj6fUNnrZZNpLKHM1QuYPhs9xkgOlvm9DNTcGiNjRfNMf8Xb',
                'api_secret': 'vt5H5Rd0DKKakiA2GGiQSmbF6rvD76Ju8ZIMitcUZQeZniTqBNHGiebsEd4MmBOR',
                'testnet': False
            },
            'trading': {
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
                'timeframes': ['1m', '5m', '15m'],
                'initial_balance': 50000000.0,
                'max_positions': 8,
                'base_currency': 'USDT'
            },
            'strategies': {
                'mean_reversion': {'enabled': True, 'position_size': 0.03},
                'momentum': {'enabled': True, 'position_size': 0.05}
            }
        }
    
    async def initialize_system(self):
        """Initialize system components"""
        print("\n🔧 Initializing System...")
        
        # Test connection
        print("📡 Testing Binance connection...")
        server_time = self.binance_client.get_server_time()
        print(f"✅ Connected! Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # Get initial prices
        print("📊 Getting initial market prices...")
        self.update_market_prices()
        
        # Initialize order manager
        print("💼 Initializing order manager...")
        self.order_manager = RealDemoOrderManager(self.config)
        
        # Initialize portfolio tracker
        print("📊 Setting up portfolio tracker...")
        initial_balance = self.config['trading']['initial_balance']
        self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)
        
        # Initialize strategies
        print("🎯 Loading strategies...")
        if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', False):
            self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
            print("  ✅ Mean reversion strategy")
        
        if self.config.get('strategies', {}).get('momentum', {}).get('enabled', False):
            self.strategies['momentum'] = MomentumStrategy(self.config)
            print("  ✅ Momentum strategy")
        
        # Setup callbacks
        self.order_manager.add_fill_callback(self._on_order_fill)
        
        print("✅ System initialized!")
    
    def update_market_prices(self):
        """Update market prices from Binance"""
        try:
            tickers = self.binance_client.get_all_tickers()
            symbols = self.config['trading']['symbols']
            prices = {}
            
            print("📊 Current Market Prices:")
            for ticker in tickers:
                if ticker['symbol'] in symbols:
                    price = float(ticker['price'])
                    prices[ticker['symbol']] = price
                    print(f"  {ticker['symbol']}: ${price:,.4f}")
            
            # Update order manager and portfolio
            if self.order_manager:
                self.order_manager.set_market_prices(prices)
            
            if self.portfolio_tracker:
                self.portfolio_tracker.update_market_prices(prices)
            
            return prices
            
        except Exception as e:
            print(f"❌ Error updating prices: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, interval: str = "1m", limit: int = 50):
        """Get historical candlestick data"""
        try:
            klines = self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _on_order_fill(self, order):
        """Handle order fills"""
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
        print(f"💰 TRADE: {order.symbol} {order.side.value} {order.quantity:.6f} @ ${order.average_price:.2f}")
    
    async def run_demo_trading(self, duration_minutes: int = 5):
        """Run trading demo"""
        print(f"\n🎯 Starting trading demo for {duration_minutes} minutes...")
        
        self.running = True
        self.start_time = datetime.now()
        end_time = time.time() + (duration_minutes * 60)
        
        last_price_update = 0
        price_update_interval = 10  # Update prices every 10 seconds
        
        last_strategy_run = 0
        strategy_interval = 30  # Run strategies every 30 seconds
        
        while self.running and time.time() < end_time:
            try:
                current_time = time.time()
                
                # Update market prices
                if current_time - last_price_update >= price_update_interval:
                    self.update_market_prices()
                    last_price_update = current_time
                
                # Run strategies
                if current_time - last_strategy_run >= strategy_interval:
                    await self._run_strategies()
                    last_strategy_run = current_time
                    
                    # Show status
                    await self._show_status()
                
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                print("\n🛑 Demo stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in demo: {e}")
                await asyncio.sleep(5)
        
        self.running = False
        print("\n✅ Demo completed!")
        await self._show_final_results()
    
    async def _run_strategies(self):
        """Run trading strategies"""
        try:
            symbols = self.config['trading']['symbols']
            
            for symbol in symbols:
                # Get historical data
                ohlcv_data = self.get_historical_data(symbol, "1m", 50)
                
                if ohlcv_data.empty:
                    continue
                
                print(f"📈 Analyzing {symbol} with {len(ohlcv_data)} candles")
                
                # Run each strategy
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.analyze_market_data(symbol, ohlcv_data)
                        
                        if signal and signal.signal_type != "HOLD":
                            await self._execute_signal(strategy_name, signal)
                            
                    except Exception as e:
                        print(f"❌ Strategy error ({strategy_name}): {e}")
                        
        except Exception as e:
            print(f"❌ Error running strategies: {e}")
    
    async def _execute_signal(self, strategy_name: str, signal):
        """Execute trading signal"""
        try:
            # Get current price
            tickers = self.binance_client.get_symbol_ticker(symbol=signal.symbol)
            current_price = float(tickers['price'])
            
            # Calculate position size
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            portfolio_value = portfolio_summary['overview']['total_value']
            
            position_size_ratio = self.config['strategies'][strategy_name]['position_size']
            position_value = portfolio_value * position_size_ratio
            quantity = position_value / current_price
            
            # Submit order
            side = OrderSide.BUY if signal.signal_type == "BUY" else OrderSide.SELL
            
            print(f"📤 SIGNAL: {strategy_name} {signal.signal_type} {signal.symbol}")
            print(f"   Position: ${position_value:,.0f} | Qty: {quantity:.6f}")
            
            order = await self.order_manager.submit_market_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                strategy=strategy_name,
                metadata={'signal_strength': signal.strength}
            )
            
        except Exception as e:
            print(f"❌ Error executing signal: {e}")
    
    async def _show_status(self):
        """Show current status"""
        try:
            if not self.portfolio_tracker:
                return
            
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n📊 STATUS:")
            print(f"  Portfolio: ${overview['total_value']:,.0f}")
            print(f"  Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"  P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"  Trades: {self.trades_executed}")
            print(f"  Positions: {overview['position_count']}")
            
        except Exception as e:
            print(f"❌ Error showing status: {e}")
    
    async def _show_final_results(self):
        """Show final results"""
        try:
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"\n🎉 FINAL RESULTS")
            print(f"=" * 50)
            print(f"Initial: ${self.config['trading']['initial_balance']:,.0f}")
            print(f"Final: ${overview['total_value']:,.0f}")
            print(f"Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"Trades: {self.trades_executed}")
            print(f"Win Rate: {overview['win_rate']:.1f}%")
            
            # Export data
            json_file = self.portfolio_tracker.export_data('json')
            print(f"\n💾 Data exported: {json_file}")
            
        except Exception as e:
            print(f"❌ Error showing results: {e}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Binance Real Trading Demo')
    parser.add_argument('--duration', type=int, default=3, 
                       help='Demo duration in minutes')
    
    args = parser.parse_args()
    
    demo = SimpleBinanceDemo()
    
    try:
        await demo.initialize_system()
        await demo.run_demo_trading(args.duration)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("\n👋 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 