#!/usr/bin/env python3
"""
Simple HFT Example - Educational Demonstration
This script shows how to use individual components of the HFT system
"""

import asyncio
import yaml
import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_data.feed_handler import BinanceFeedHandler
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from utils.logger import setup_logging, get_hft_logger


class SimpleHFTExample:
    """Simple example of HFT system usage"""
    
    def __init__(self):
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        setup_logging()
        self.logger = get_hft_logger().get_logger("simple_example")
        
        # Initialize components
        self.feed_handler = None
        self.strategies = {}
        
    def _load_config(self):
        """Load configuration"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print("Error: config.yaml not found. Please ensure it exists in the config/ directory.")
            sys.exit(1)
    
    def demo_market_data_feed(self):
        """Demonstrate market data feed functionality"""
        print("\n=== MARKET DATA FEED DEMO ===")
        
        # Initialize feed handler
        self.feed_handler = BinanceFeedHandler(self.config)
        
        # Add callbacks to print data
        def on_tick(tick_data):
            print(f"TICK: {tick_data['symbol']} - Price: ${tick_data['price']:.2f}, "
                  f"Qty: {tick_data['quantity']:.4f}, Time: {tick_data['timestamp']}")
        
        def on_kline(kline_data):
            if kline_data['is_closed']:  # Only print completed candles
                print(f"KLINE: {kline_data['symbol']} {kline_data['interval']} - "
                      f"O: ${kline_data['open']:.2f}, H: ${kline_data['high']:.2f}, "
                      f"L: ${kline_data['low']:.2f}, C: ${kline_data['close']:.2f}")
        
        self.feed_handler.add_tick_callback(on_tick)
        self.feed_handler.add_kline_callback(on_kline)
        
        try:
            print("Starting market data streams... (Press Ctrl+C to stop)")
            self.feed_handler.start_streams()
            
            # Let it run for a bit
            import time
            time.sleep(30)  # Run for 30 seconds
            
        except KeyboardInterrupt:
            print("\nStopping market data streams...")
        finally:
            self.feed_handler.stop_streams()
    
    def demo_strategy_signals(self):
        """Demonstrate strategy signal generation"""
        print("\n=== STRATEGY SIGNALS DEMO ===")
        
        # Initialize strategies
        mean_reversion = MeanReversionStrategy(self.config)
        momentum = MomentumStrategy(self.config)
        
        # Simulate some market data (in real scenario, this would come from feed handler)
        sample_data = self._generate_sample_data()
        
        print(f"Analyzing sample data with {len(sample_data)} candles...")
        
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            print(f"\n--- Analyzing {symbol} ---")
            
            # Get signals from strategies
            mr_signal = mean_reversion.analyze_market_data(symbol, sample_data)
            mom_signal = momentum.analyze_market_data(symbol, sample_data)
            
            # Print results
            if mr_signal and mr_signal.signal_type != "HOLD":
                print(f"Mean Reversion: {mr_signal.signal_type} signal with strength {mr_signal.strength:.2f}")
                print(f"  Metadata: {mr_signal.metadata}")
            else:
                print("Mean Reversion: No signal (HOLD)")
            
            if mom_signal and mom_signal.signal_type != "HOLD":
                print(f"Momentum: {mom_signal.signal_type} signal with strength {mom_signal.strength:.2f}")
                print(f"  Metadata: {mom_signal.metadata}")
            else:
                print("Momentum: No signal (HOLD)")
    
    def _generate_sample_data(self):
        """Generate sample OHLCV data for demonstration"""
        import numpy as np
        
        # Generate 100 candles of sample data
        np.random.seed(42)  # For reproducible results
        
        base_price = 45000  # Starting price
        data = []
        
        for i in range(100):
            # Random walk with some trend
            change = np.random.normal(0, 0.02)  # 2% volatility
            if i > 50:  # Add some trend in the second half
                change += 0.001  # Slight upward trend
            
            new_price = base_price * (1 + change)
            
            # Generate OHLC from the price
            high = new_price * (1 + abs(np.random.normal(0, 0.005)))
            low = new_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = base_price
            close_price = new_price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'timestamp': datetime.now()
            })
            
            base_price = new_price
        
        return pd.DataFrame(data)
    
    def demo_risk_checks(self):
        """Demonstrate risk management functionality"""
        print("\n=== RISK MANAGEMENT DEMO ===")
        
        from risk.risk_engine import RiskEngine
        
        # Initialize risk engine
        risk_engine = RiskEngine(self.config)
        
        print("Risk Engine Configuration:")
        print(f"  Max Portfolio Risk: {risk_engine.max_portfolio_risk:.1%}")
        print(f"  Max Daily Loss: {risk_engine.max_daily_loss:.1%}")
        print(f"  Stop Loss: {risk_engine.stop_loss:.1%}")
        print(f"  Take Profit: {risk_engine.take_profit:.1%}")
        print(f"  Max Open Positions: {risk_engine.max_open_positions}")
        
        # Test pre-trade risk checks
        print("\nTesting pre-trade risk checks:")
        
        # Normal trade
        risk_check = risk_engine.perform_pre_trade_check(
            "BTCUSDT", "BUY", 0.1, 45000, "test_strategy"
        )
        print(f"Normal trade: {'PASSED' if risk_check.passed else 'FAILED'}")
        
        # Large trade (should fail)
        risk_check = risk_engine.perform_pre_trade_check(
            "BTCUSDT", "BUY", 10.0, 45000, "test_strategy"
        )
        print(f"Large trade: {'PASSED' if risk_check.passed else 'FAILED'} - {risk_check.message}")
        
        # Show risk statistics
        stats = risk_engine.get_risk_statistics()
        print(f"\nRisk Statistics:")
        print(f"  Risk Checks Performed: {stats['risk_checks_performed']}")
        print(f"  Risk Checks Failed: {stats['risk_checks_failed']}")
        print(f"  Portfolio Value: ${stats['portfolio_value']:,.2f}")
    
    async def demo_order_simulation(self):
        """Demonstrate order management (simulation only)"""
        print("\n=== ORDER MANAGEMENT DEMO (SIMULATION) ===")
        
        from execution.order_manager import OrderManager, OrderSide
        
        # Note: This would normally connect to Binance testnet
        print("Note: This demo simulates order management without actual API calls")
        print("In a real scenario, this would connect to Binance testnet")
        
        # Show order types and workflow
        print("\nOrder Types Available:")
        print("  - Market Orders (immediate execution)")
        print("  - Limit Orders (execute at specific price)")
        print("  - Stop Loss Orders (risk management)")
        
        print("\nOrder Workflow:")
        print("  1. Create order with strategy metadata")
        print("  2. Submit to exchange via API")
        print("  3. Monitor order status")
        print("  4. Handle fills and updates")
        print("  5. Execute callbacks for risk management")
    
    def run_all_demos(self):
        """Run all demonstration functions"""
        print("=== HFT SYSTEM COMPONENT DEMONSTRATIONS ===")
        print("This demo shows the key components of the HFT system")
        print("All operations use dummy/test data - no real trading occurs")
        
        # Run demos
        self.demo_strategy_signals()
        self.demo_risk_checks()
        asyncio.run(self.demo_order_simulation())
        
        print("\n=== DEMO COMPLETE ===")
        print("To see live market data, run: python examples/simple_example.py --live-data")
        print("To run the full HFT system, run: python main.py")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HFT System Simple Example')
    parser.add_argument('--live-data', action='store_true', 
                       help='Demo live market data feed (requires API keys)')
    
    args = parser.parse_args()
    
    example = SimpleHFTExample()
    
    if args.live_data:
        print("Starting live market data demo...")
        print("Make sure your API keys are configured in config/config.yaml")
        example.demo_market_data_feed()
    else:
        example.run_all_demos()


if __name__ == "__main__":
    main() 