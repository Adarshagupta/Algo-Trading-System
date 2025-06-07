#!/usr/bin/env python3
"""
Launch Web Dashboard for Live Portfolio Monitoring
Shows real-time portfolio updates with Binance data
"""

import os
import sys
import asyncio
import time
import threading
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio.portfolio_tracker import PortfolioTracker
from portfolio.web_dashboard import PortfolioWebDashboard, create_html_template
from binance.client import Client
import yaml


class LivePortfolioDashboard:
    """Live portfolio dashboard with real Binance data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Binance client for live price updates
        self.binance_client = Client(
            api_key=self.config['binance']['api_key'],
            api_secret=self.config['binance']['api_secret'],
            testnet=False
        )
        
        # Initialize portfolio tracker
        initial_balance = self.config['trading']['initial_balance']
        self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)
        
        # Add some demo trades to show on dashboard
        self._add_demo_trades()
        
        print("🌐 LIVE PORTFOLIO WEB DASHBOARD")
        print("=" * 50)
        print("✅ Real-time Binance price updates")
        print("✅ Live portfolio monitoring")
        print("✅ Interactive charts and graphs")
        print("✅ Trade history visualization")
        print(f"💼 Portfolio: ${initial_balance:,.0f}")
        print("=" * 50)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {
                'binance': {
                    'api_key': 'GKEYgU4j5FdiCx10Vj6fUNnrZZNpLKHM1QuYPhs9xkgOlvm9DNTcGiNjRfNMf8Xb',
                    'api_secret': 'vt5H5Rd0DKKakiA2GGiQSmbF6rvD76Ju8ZIMitcUZQeZniTqBNHGiebsEd4MmBOR',
                    'testnet': False
                },
                'trading': {
                    'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
                    'initial_balance': 50000000.0,
                    'base_currency': 'USDT'
                }
            }
    
    def _add_demo_trades(self):
        """Add some demo trades to show portfolio activity"""
        try:
            # Get current prices
            tickers = self.binance_client.get_all_tickers()
            prices = {}
            for ticker in tickers:
                if ticker['symbol'] in self.config['trading']['symbols']:
                    prices[ticker['symbol']] = float(ticker['price'])
            
            # Add demo trades
            demo_trades = [
                ('BTCUSDT', 'BUY', 0.5, prices.get('BTCUSDT', 50000), 'momentum'),
                ('ETHUSDT', 'SELL', 1.2, prices.get('ETHUSDT', 3000), 'mean_reversion'),
                ('ADAUSDT', 'BUY', 1000, prices.get('ADAUSDT', 1), 'momentum'),
            ]
            
            for i, (symbol, side, quantity, price, strategy) in enumerate(demo_trades):
                if price:
                    self.portfolio_tracker.add_trade(
                        trade_id=f"DEMO_{i+1}",
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        commission=quantity * price * 0.001,
                        strategy=strategy,
                        order_id=f"ORDER_{i+1}"
                    )
            
            # Update with current market prices
            self.portfolio_tracker.update_market_prices(prices)
            
            print(f"✅ Added {len(demo_trades)} demo trades")
            
        except Exception as e:
            print(f"⚠️  Could not add demo trades: {e}")
    
    def start_price_updates(self):
        """Start background price updates"""
        def update_prices():
            while True:
                try:
                    # Get current prices from Binance
                    tickers = self.binance_client.get_all_tickers()
                    prices = {}
                    
                    for ticker in tickers:
                        if ticker['symbol'] in self.config['trading']['symbols']:
                            prices[ticker['symbol']] = float(ticker['price'])
                    
                    # Update portfolio with live prices
                    self.portfolio_tracker.update_market_prices(prices)
                    
                    # Sleep for 5 seconds before next update
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"❌ Price update error: {e}")
                    time.sleep(10)
        
        # Start price updates in background thread
        price_thread = threading.Thread(target=update_prices, daemon=True)
        price_thread.start()
        print("📊 Started live price updates (every 5 seconds)")
    
    def launch_dashboard(self, host='localhost', port=5000):
        """Launch the web dashboard"""
        try:
            # Create HTML template if needed
            template_path = 'portfolio/templates/dashboard.html'
            if not os.path.exists(template_path):
                os.makedirs('portfolio/templates', exist_ok=True)
                create_html_template()
            
            # Start price updates
            self.start_price_updates()
            
            # Create and start dashboard
            dashboard = PortfolioWebDashboard(
                self.portfolio_tracker, 
                host=host, 
                port=port
            )
            
            print(f"\n🚀 Starting Web Dashboard at http://{host}:{port}")
            print("🌐 Open your browser and navigate to the URL above")
            print("📊 You'll see real-time portfolio updates with live Binance prices")
            print("📈 Charts will update automatically with trade data")
            print("\nPress Ctrl+C to stop the dashboard")
            
            # Start the dashboard
            dashboard.start(debug=False)
            
        except KeyboardInterrupt:
            print("\n✅ Web dashboard stopped")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Live Portfolio Web Dashboard')
    parser.add_argument('--host', default='localhost', help='Dashboard host')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    
    args = parser.parse_args()
    
    dashboard = LivePortfolioDashboard()
    dashboard.launch_dashboard(args.host, args.port)


if __name__ == "__main__":
    main() 