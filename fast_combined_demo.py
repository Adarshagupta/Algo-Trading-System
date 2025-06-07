#!/usr/bin/env python3
"""
FAST Combined Real Trading Demo + Web Dashboard
Updates every SECOND with live Binance data for real-time monitoring
"""

import asyncio
import os
import sys
import time
import threading
from datetime import datetime
import webbrowser

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance.client import Client
from execution.real_demo_order_manager import RealDemoOrderManager, OrderSide
from portfolio.portfolio_tracker import PortfolioTracker
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from flask import Flask, render_template, jsonify
import yaml
import pandas as pd


class FastTradingDemo:
    """FAST trading demo with SECOND-by-SECOND updates"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
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
        
        print("🚀 FAST REAL-TIME TRADING DEMO")
        print("=" * 70)
        print("⚡ LIVE updates every SECOND")
        print("⚡ Real-time Binance market data") 
        print("⚡ Instant portfolio updates")
        print("⚡ Lightning-fast web dashboard")
        print("🔒 Read-only API (completely safe)")
        print("💰 Demo money only - NO REAL TRADING")
        print(f"💼 Portfolio: ${self.config['trading']['initial_balance']:,.0f}")
        print("=" * 70)
        
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
        print("\n🔧 Initializing FAST System...")
        
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
        
        print("✅ FAST System initialized!")
    
    def update_market_prices(self, verbose=False):
        """Update market prices from Binance FAST"""
        try:
            tickers = self.binance_client.get_all_tickers()
            symbols = self.config['trading']['symbols']
            prices = {}
            
            if verbose:
                print("📊 Current Market Prices:")
            
            for ticker in tickers:
                if ticker['symbol'] in symbols:
                    price = float(ticker['price'])
                    prices[ticker['symbol']] = price
                    if verbose:
                        print(f"  {ticker['symbol']}: ${price:,.4f}")
            
            # Update order manager and portfolio
            if self.order_manager:
                self.order_manager.set_market_prices(prices)
            
            if self.portfolio_tracker:
                self.portfolio_tracker.update_market_prices(prices)
            
            return prices
            
        except Exception as e:
            if verbose:
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
    
    async def run_fast_trading_loop(self):
        """Run FAST trading loop with SECOND updates"""
        print("\n⚡ Starting FAST trading loop...")
        print("⚡ Price updates: EVERY SECOND")
        print("⚡ Strategy runs: Every 10 seconds") 
        print("⚡ Portfolio updates: REAL-TIME")
        
        self.running = True
        self.start_time = datetime.now()
        
        last_price_update = 0
        price_update_interval = 1  # Update prices EVERY SECOND!
        
        last_strategy_run = 0
        strategy_interval = 10  # Run strategies every 10 seconds
        
        last_status_update = 0
        status_interval = 5  # Show status every 5 seconds
        
        update_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update market prices EVERY SECOND
                if current_time - last_price_update >= price_update_interval:
                    self.update_market_prices(verbose=False)  # Silent updates
                    last_price_update = current_time
                    update_count += 1
                    
                    # Show update indicator
                    if update_count % 5 == 0:  # Every 5 seconds
                        print(f"⚡ Live update #{update_count} - Portfolio tracking REAL-TIME")
                
                # Run strategies every 10 seconds
                if current_time - last_strategy_run >= strategy_interval:
                    await self._run_strategies()
                    last_strategy_run = current_time
                
                # Show status every 5 seconds
                if current_time - last_status_update >= status_interval:
                    self._show_fast_status()
                    last_status_update = current_time
                
                # VERY short sleep for maximum responsiveness
                await asyncio.sleep(0.5)  # 500ms sleep for ultra-fast updates
                
            except Exception as e:
                print(f"❌ Error in fast trading: {e}")
                await asyncio.sleep(1)
    
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
    
    def _show_fast_status(self):
        """Show FAST status with real-time data"""
        try:
            if not self.portfolio_tracker:
                return
            
            summary = self.portfolio_tracker.get_portfolio_summary()
            overview = summary['overview']
            
            print(f"⚡ FAST STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"  💼 Portfolio: ${overview['total_value']:,.0f}")
            print(f"  📈 Growth: {overview['portfolio_growth']:+.2f}%")
            print(f"  💰 P&L: ${overview['total_unrealized_pnl']:+,.0f}")
            print(f"  🔄 Trades: {self.trades_executed}")
            print(f"  📊 Positions: {overview['position_count']}")
            
        except Exception as e:
            print(f"❌ Error showing status: {e}")
    
    def create_fast_web_app(self):
        """Create FAST Flask web application"""
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            """FAST dashboard page"""
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>⚡ FAST Live Trading Dashboard - $50M Portfolio</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .header { background: #27ae60; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; animation: pulse 2s infinite; }
                    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
                    .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; transition: transform 0.3s; }
                    .stat-card:hover { transform: scale(1.05); }
                    .stat-value { font-size: 2em; font-weight: bold; color: #27ae60; }
                    .stat-label { color: #7f8c8d; margin-top: 5px; }
                    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    .live-indicator { color: #e74c3c; font-weight: bold; animation: blink 1s infinite; }
                    .trades-list { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-top: 20px; }
                    .trade-item { padding: 10px; border-bottom: 1px solid #ecf0f1; transition: background 0.3s; }
                    .trade-item:hover { background: #ecf0f1; }
                    .update-time { color: #e74c3c; font-weight: bold; }
                    
                    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
                    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>⚡ FAST Live Trading Dashboard</h1>
                    <h2>$50,000,000 Demo Portfolio</h2>
                    <p class="live-indicator">● LIVE - UPDATING EVERY SECOND</p>
                    <p>Last Update: <span id="update-time" class="update-time">Loading...</span></p>
                </div>
                
                <div class="stats" id="stats">
                    <!-- Stats will be populated by JavaScript -->
                </div>
                
                <div class="charts">
                    <div class="chart-container">
                        <h3>⚡ Portfolio Value (Real-Time)</h3>
                        <div id="portfolio-chart"></div>
                    </div>
                    <div class="chart-container">
                        <h3>📊 Position Allocation</h3>
                        <div id="allocation-chart"></div>
                    </div>
                </div>
                
                <div class="trades-list">
                    <h3>💰 Recent Trades</h3>
                    <div id="trades">
                        <!-- Trades will be populated by JavaScript -->
                    </div>
                </div>
                
                <script>
                    let updateCount = 0;
                    
                    function updateDashboard() {
                        updateCount++;
                        fetch('/api/portfolio')
                            .then(response => response.json())
                            .then(data => {
                                updateStats(data);
                                updateCharts(data);
                                updateTrades(data);
                                
                                // Update time indicator
                                const now = new Date();
                                document.getElementById('update-time').textContent = 
                                    now.toLocaleTimeString() + ` (Update #${updateCount})`;
                            })
                            .catch(error => {
                                console.error('Error:', error);
                                document.getElementById('update-time').textContent = 'ERROR - Retrying...';
                            });
                    }
                    
                    function updateStats(data) {
                        const overview = data.overview;
                        const statsDiv = document.getElementById('stats');
                        
                        statsDiv.innerHTML = `
                            <div class="stat-card">
                                <div class="stat-value">$${overview.total_value.toLocaleString()}</div>
                                <div class="stat-label">Portfolio Value</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${overview.portfolio_growth > 0 ? '+' : ''}${overview.portfolio_growth.toFixed(2)}%</div>
                                <div class="stat-label">Growth</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">$${overview.total_unrealized_pnl > 0 ? '+' : ''}${overview.total_unrealized_pnl.toLocaleString()}</div>
                                <div class="stat-label">Unrealized P&L</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${overview.position_count}</div>
                                <div class="stat-label">Open Positions</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${overview.win_rate.toFixed(1)}%</div>
                                <div class="stat-label">Win Rate</div>
                            </div>
                        `;
                    }
                    
                    function updateCharts(data) {
                        // Portfolio value chart
                        const portfolioTrace = {
                            x: data.performance.map(p => p.timestamp),
                            y: data.performance.map(p => p.portfolio_value),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Portfolio Value',
                            line: { color: '#27ae60', width: 3 },
                            marker: { size: 4 }
                        };
                        
                        Plotly.newPlot('portfolio-chart', [portfolioTrace], {
                            margin: { t: 0, r: 0, b: 40, l: 80 },
                            xaxis: { title: 'Time' },
                            yaxis: { title: 'Value ($)' }
                        });
                        
                        // Position allocation chart
                        if (data.positions && data.positions.length > 0) {
                            const allocationTrace = {
                                labels: data.positions.map(p => p.symbol),
                                values: data.positions.map(p => Math.abs(p.market_value)),
                                type: 'pie',
                                textinfo: 'label+percent',
                                textposition: 'auto',
                                marker: { colors: ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'] }
                            };
                            
                            Plotly.newPlot('allocation-chart', [allocationTrace], {
                                margin: { t: 0, r: 0, b: 0, l: 0 }
                            });
                        }
                    }
                    
                    function updateTrades(data) {
                        const tradesDiv = document.getElementById('trades');
                        
                        if (data.trades && data.trades.length > 0) {
                            tradesDiv.innerHTML = data.trades.slice(-10).reverse().map(trade => `
                                <div class="trade-item">
                                    <strong>${trade.symbol}</strong> ${trade.side} 
                                    ${trade.quantity.toFixed(6)} @ $${trade.price.toFixed(2)}
                                    <span style="color: #7f8c8d; float: right;">${trade.strategy}</span>
                                </div>
                            `).join('');
                        } else {
                            tradesDiv.innerHTML = '<div class="trade-item">⚡ No trades yet - waiting for signals...</div>';
                        }
                    }
                    
                    // Update EVERY SECOND for maximum speed!
                    updateDashboard();
                    setInterval(updateDashboard, 1000);
                </script>
            </body>
            </html>
            '''
        
        @app.route('/api/portfolio')
        def api_portfolio():
            """FAST API endpoint for portfolio data"""
            try:
                if self.portfolio_tracker:
                    return jsonify(self.portfolio_tracker.get_portfolio_summary())
                else:
                    return jsonify({'error': 'Portfolio tracker not initialized'})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        return app
    
    def start_fast_web_dashboard(self, port=5002):
        """Start FAST web dashboard"""
        def run_flask():
            app = self.create_fast_web_app()
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        web_thread = threading.Thread(target=run_flask, daemon=True)
        web_thread.start()
        
        print(f"\n⚡ FAST Web Dashboard started at http://localhost:{port}")
        print("⚡ Dashboard updates EVERY SECOND!")
        print("⚡ Portfolio tracking in REAL-TIME!")
        
        # Open browser automatically
        time.sleep(2)  # Wait for server to start
        webbrowser.open(f'http://localhost:{port}')
    
    async def run_fast_demo(self):
        """Run FAST combined trading and web demo"""
        try:
            # Initialize system
            await self.initialize_system()
            
            # Start FAST web dashboard
            self.start_fast_web_dashboard()
            
            # Add some initial demo trades for visualization
            self._add_demo_trades()
            
            print("\n⚡ Starting LIGHTNING-FAST trading simulation...")
            print("⚡ Web dashboard updating EVERY SECOND")
            print("⚡ Portfolio tracking in REAL-TIME")
            print("⚡ Market data refreshing CONTINUOUSLY")
            print("\nPress Ctrl+C to stop both systems")
            
            # Run FAST trading loop
            await self.run_fast_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 FAST demo stopped by user")
            self.running = False
        except Exception as e:
            print(f"❌ FAST demo failed: {e}")
    
    def _add_demo_trades(self):
        """Add demo trades for visualization"""
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
                ('ETHUSDT', 'BUY', 2.0, prices.get('ETHUSDT', 3000), 'mean_reversion'),
                ('ADAUSDT', 'BUY', 1500, prices.get('ADAUSDT', 1), 'momentum'),
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
            
            print(f"⚡ Added {len(demo_trades)} demo trades for FAST visualization")
            
        except Exception as e:
            print(f"⚠️  Could not add demo trades: {e}")


async def main():
    """Main function"""
    demo = FastTradingDemo()
    await demo.run_fast_demo()


if __name__ == "__main__":
    asyncio.run(main()) 