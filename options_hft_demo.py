#!/usr/bin/env python3
"""
Options Trading HFT Demo
Integrates options trading strategies with the existing HFT system
"""

import asyncio
import time
import sys
import os
from datetime import datetime
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.options_trading import OptionsStrategy, OptionsSignal
from binance.client import Client
from utils.logger import get_hft_logger
import yaml


class OptionsHFTSystem:
    """High-Frequency Options Trading System"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = get_hft_logger().get_logger("options_hft")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Binance client for price data
        self.binance_client = Client(
            api_key=self.config['binance']['api_key'],
            api_secret=self.config['binance']['api_secret'],
            testnet=False
        )
        
        # Initialize options strategy
        self.options_strategy = OptionsStrategy(self.config)
        
        # Trading state
        self.active_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        self.current_prices = {}
        self.options_signals = []
        self.running = True
        
        self.logger.info("Options HFT System initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration for options trading
            return {
                'binance': {
                    'api_key': 'GKEYgU4j5FdiCx10Vj6fUNnrZZNpLKHM1QuYPhs9xkgOlvm9DNTcGiNjRfNMf8Xb',
                    'api_secret': 'vt5H5Rd0DKKakiA2GGiQSmbF6rvD76Ju8ZIMitcUZQeZniTqBNHGiebsEd4MmBOR',
                    'testnet': False
                },
                'trading': {
                    'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                    'initial_balance': 1000000.0,
                    'base_currency': 'USDT'
                },
                'options': {
                    'risk_free_rate': 0.05,
                    'max_days_to_expiry': 45,
                    'min_days_to_expiry': 7,
                    'max_position_size': 10,
                    'max_delta_exposure': 5.0,
                    'enabled_strategies': ['momentum_breakout', 'volatility_expansion']
                }
            }
    
    async def start_trading(self):
        """Start the options trading loop"""
        print("\n🚀 STARTING OPTIONS HFT SYSTEM")
        print("=" * 60)
        print("⚡ Real-time options signal generation")
        print("📊 Greeks calculation and risk management")
        print("🎯 Multiple strategy execution")
        print("=" * 60)
        
        try:
            while self.running:
                # Update market prices
                await self._update_market_prices()
                
                # Generate options signals for each symbol
                for symbol in self.active_symbols:
                    if symbol in self.current_prices:
                        await self._analyze_options_for_symbol(symbol)
                
                # Display current status
                self._display_status()
                
                # Wait before next cycle (options strategies typically run slower)
                await asyncio.sleep(10)  # 10-second intervals for options
                
        except KeyboardInterrupt:
            print("\n✅ Options trading stopped by user")
            self.running = False
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
            print(f"❌ Error in trading loop: {e}")
    
    async def _update_market_prices(self):
        """Update current market prices from Binance"""
        try:
            tickers = self.binance_client.get_all_tickers()
            
            for ticker in tickers:
                if ticker['symbol'] in self.active_symbols:
                    self.current_prices[ticker['symbol']] = float(ticker['price'])
            
        except Exception as e:
            self.logger.error(f"Error updating prices: {e}")
    
    async def _analyze_options_for_symbol(self, symbol: str):
        """Analyze options opportunities for a specific symbol"""
        try:
            spot_price = self.current_prices[symbol]
            
            # Generate options signals
            signals = self.options_strategy.analyze_options_chain(symbol, spot_price)
            
            # Process and store signals
            for signal in signals:
                self._process_options_signal(signal)
                self.options_signals.append(signal)
            
            # Keep only recent signals (last 10 per symbol)
            symbol_signals = [s for s in self.options_signals if s.underlying == symbol]
            if len(symbol_signals) > 10:
                # Remove oldest signals for this symbol
                for old_signal in symbol_signals[:-10]:
                    self.options_signals.remove(old_signal)
            
        except Exception as e:
            self.logger.error(f"Error analyzing options for {symbol}: {e}")
    
    def _process_options_signal(self, signal: OptionsSignal):
        """Process and log an options signal"""
        self.logger.info(f"Options Signal Generated: {signal.signal_type} {signal.contract.symbol}")
        
        # In a real system, this would execute the trade
        # For demo, we just log the trade intention
        
        trade_info = {
            'timestamp': signal.timestamp.strftime('%H:%M:%S'),
            'underlying': signal.underlying,
            'strategy': signal.strategy,
            'action': signal.signal_type,
            'contract': signal.contract.symbol,
            'strike': signal.contract.strike_price,
            'expiry': signal.contract.expiry_date.strftime('%Y-%m-%d'),
            'premium': signal.contract.premium,
            'quantity': signal.quantity,
            'delta': signal.contract.delta,
            'gamma': signal.contract.gamma,
            'theta': signal.contract.theta,
            'vega': signal.contract.vega,
            'iv': signal.contract.implied_volatility,
            'max_risk': signal.max_risk,
            'expected_profit': signal.expected_profit,
            'probability': signal.probability,
            'rationale': signal.rationale
        }
        
        print(f"\n📈 OPTIONS SIGNAL: {signal.signal_type}")
        print(f"   Contract: {signal.contract.symbol}")
        print(f"   Strike: ${signal.contract.strike_price:,.2f}")
        print(f"   Premium: ${signal.contract.premium:.4f}")
        print(f"   Delta: {signal.contract.delta:.3f}")
        print(f"   IV: {signal.contract.implied_volatility:.1%}")
        print(f"   Strategy: {signal.strategy}")
        print(f"   Rationale: {signal.rationale}")
    
    def _display_status(self):
        """Display current system status"""
        current_time = datetime.now().strftime('%H:%M:%S')
        total_signals = len(self.options_signals)
        recent_signals = len([s for s in self.options_signals 
                            if (datetime.now() - s.timestamp).seconds < 60])
        
        print(f"\n🕐 {current_time} | Signals: {total_signals} total, {recent_signals} recent")
        
        # Display current prices
        print("💰 Current Prices:")
        for symbol, price in self.current_prices.items():
            print(f"   {symbol}: ${price:,.2f}")
        
        # Display portfolio summary
        portfolio_summary = self.options_strategy.get_portfolio_summary()
        print(f"📊 Options Portfolio: {portfolio_summary['total_positions']} positions")
        
        if portfolio_summary['portfolio_greeks']:
            greeks = portfolio_summary['portfolio_greeks']
            print(f"   Delta: {greeks['delta']:.2f} | Gamma: {greeks['gamma']:.4f}")
            print(f"   Theta: {greeks['theta']:.2f} | Vega: {greeks['vega']:.2f}")


def main():
    """Main function to run the options HFT demo"""
    print("🎯 OPTIONS HIGH-FREQUENCY TRADING DEMO")
    print("Integrating options strategies with real market data...")
    
    try:
        # Initialize system
        options_hft = OptionsHFTSystem()
        
        # Start trading loop
        asyncio.run(options_hft.start_trading())
        
    except Exception as e:
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    main() 