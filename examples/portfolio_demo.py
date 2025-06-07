#!/usr/bin/env python3
"""
Portfolio Tracker Demo - Showcase Portfolio Tracking Capabilities
Demonstrates the CLI and Web interfaces for portfolio monitoring
"""

import asyncio
import sys
import os
import time
import argparse
from datetime import datetime
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_tracker import PortfolioTracker
from portfolio.portfolio_cli import PortfolioCLI
from portfolio.web_dashboard import PortfolioWebDashboard, create_html_template


def simulate_trading_activity(portfolio: PortfolioTracker, duration_minutes: int = 5):
    """Simulate trading activity for demonstration"""
    print(f"🎬 Starting {duration_minutes}-minute trading simulation...")
    
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    strategies = ["mean_reversion", "momentum", "arbitrage"]
    
    # Starting prices
    prices = {
        "BTCUSDT": 45000,
        "ETHUSDT": 3000,
        "ADAUSDT": 0.5,
        "DOTUSDT": 25,
        "LINKUSDT": 15
    }
    
    trade_id = 1
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    print("Generating simulated trades...")
    
    while time.time() < end_time:
        # Random trade
        symbol = random.choice(symbols)
        side = random.choice(["BUY", "SELL"])
        quantity = random.uniform(0.1, 2.0)
        strategy = random.choice(strategies)
        
        # Price movement simulation
        price_change = random.uniform(-0.05, 0.05)  # ±5% price movement
        prices[symbol] *= (1 + price_change)
        
        # Execute trade
        commission = quantity * prices[symbol] * 0.001  # 0.1% commission
        portfolio.add_trade(
            trade_id=f"SIM_{trade_id:04d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=prices[symbol],
            commission=commission,
            strategy=strategy
        )
        
        # Update market prices
        portfolio.update_market_prices(prices)
        
        trade_id += 1
        
        # Random delay between trades
        time.sleep(random.uniform(1, 10))
    
    print(f"✅ Simulation complete! Generated {trade_id-1} trades")
    
    # Final portfolio summary
    summary = portfolio.get_portfolio_summary()
    metrics = summary['overview']
    
    print(f"\n📊 Final Portfolio Summary:")
    print(f"Portfolio Value: ${metrics['total_value']:,.2f}")
    print(f"Growth: {metrics['portfolio_growth']:.2f}%")
    print(f"Open Positions: {metrics['position_count']}")
    print(f"Total Trades: {len(summary['recent_trades'])}")


def demo_cli_interface(portfolio: PortfolioTracker):
    """Demonstrate the CLI interface"""
    print("\n🖥️  Starting CLI Demo...")
    print("The CLI provides a real-time terminal interface with:")
    print("- Live portfolio overview")
    print("- Position tracking with P&L")
    print("- Performance metrics")
    print("- Risk analysis")
    print("- Recent trades")
    
    cli = PortfolioCLI(portfolio)
    
    try:
        print("\nPress Ctrl+C to exit the CLI demo")
        cli.run_live_display(update_interval=1.0)
    except KeyboardInterrupt:
        print("\n✅ CLI demo completed")


def demo_web_interface(portfolio: PortfolioTracker, host='localhost', port=5000):
    """Demonstrate the web interface"""
    print(f"\n🌐 Starting Web Dashboard Demo...")
    print("The web dashboard provides:")
    print("- Beautiful browser-based interface")
    print("- Real-time updates via WebSocket")
    print("- Interactive charts")
    print("- Export functionality")
    print("- Responsive design")
    
    # Create HTML template if needed
    if not os.path.exists('portfolio/templates/dashboard.html'):
        create_html_template()
    
    dashboard = PortfolioWebDashboard(portfolio, host=host, port=port)
    
    print(f"\n🚀 Dashboard will start at http://{host}:{port}")
    print("Open your browser and navigate to the URL above")
    print("Press Ctrl+C to stop the web server")
    
    try:
        dashboard.start(debug=False)
    except KeyboardInterrupt:
        print("\n✅ Web dashboard demo completed")
        dashboard.stop()


def demo_portfolio_analytics(portfolio: PortfolioTracker):
    """Demonstrate portfolio analytics capabilities"""
    print("\n📊 Portfolio Analytics Demo")
    print("="*50)
    
    summary = portfolio.get_portfolio_summary()
    
    # Overview metrics
    print("\n💼 Portfolio Overview:")
    overview = summary['overview']
    print(f"  Portfolio Value: ${overview['total_value']:,.2f}")
    print(f"  Cash Balance: ${overview['cash_balance']:,.2f}")
    print(f"  Portfolio Growth: {overview['portfolio_growth']:.2f}%")
    print(f"  Total P&L: ${overview['total_unrealized_pnl']:,.2f}")
    print(f"  Win Rate: {overview['win_rate']:.1f}%")
    
    # Position details
    print(f"\n📈 Open Positions ({overview['position_count']}):")
    for pos in summary['positions']:
        pnl_symbol = "+" if pos['unrealized_pnl'] >= 0 else ""
        print(f"  {pos['symbol']}: {pos['side']} {pos['quantity']:.4f} @ ${pos['avg_entry_price']:.2f}")
        print(f"    Current: ${pos['current_price']:.2f} | P&L: {pnl_symbol}${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.2f}%)")
    
    # Performance analysis
    performance = summary.get('performance', {})
    if performance:
        print(f"\n⏱️  Performance Analysis:")
        time_returns = performance.get('time_returns', {})
        for period, return_val in time_returns.items():
            symbol = "▲" if return_val >= 0 else "▼"
            print(f"  {period.capitalize()}: {symbol} {return_val:.2f}%")
    
    # Risk analysis
    risk = summary.get('risk_analysis', {})
    if risk:
        print(f"\n⚠️  Risk Analysis:")
        concentration = risk.get('concentration', {})
        if concentration:
            print(f"  Largest Position: {concentration.get('largest_position_percent', 0):.1f}%")
        
        exposure = risk.get('exposure', {})
        if exposure:
            print(f"  Cash Ratio: {exposure.get('cash_ratio', 0):.1f}%")
            print(f"  Total Exposure: ${exposure.get('total_exposure', 0):,.2f}")
    
    # Strategy breakdown
    strategy_performance = performance.get('strategy_breakdown', {})
    if strategy_performance:
        print(f"\n🎯 Strategy Performance:")
        for strategy, metrics in strategy_performance.items():
            pnl = metrics.get('pnl', 0)
            trades = metrics.get('trades', 0)
            symbol = "+" if pnl >= 0 else ""
            print(f"  {strategy}: {symbol}${pnl:.2f} ({trades} trades)")
    
    # Export demonstration
    print(f"\n💾 Data Export Demo:")
    try:
        json_file = portfolio.export_data('json')
        print(f"  ✅ JSON export: {json_file}")
        
        csv_file = portfolio.export_data('csv')
        print(f"  ✅ CSV export: {csv_file}")
        
        snapshot_file = portfolio.save_snapshot()
        print(f"  ✅ Snapshot saved: {snapshot_file}")
        
    except Exception as e:
        print(f"  ❌ Export error: {e}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Portfolio Tracker Demo')
    parser.add_argument('--mode', choices=['cli', 'web', 'analytics', 'simulation'], 
                       default='analytics', help='Demo mode to run')
    parser.add_argument('--balance', type=float, default=10000.0, 
                       help='Initial portfolio balance')
    parser.add_argument('--duration', type=int, default=2, 
                       help='Simulation duration in minutes')
    parser.add_argument('--host', default='localhost', 
                       help='Web dashboard host')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Web dashboard port')
    parser.add_argument('--simulate', action='store_true', 
                       help='Run trading simulation before demo')
    
    args = parser.parse_args()
    
    print("🚀 HFT Portfolio Tracker Demo")
    print("="*40)
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Demo Mode: {args.mode}")
    
    # Initialize portfolio tracker
    config = {}
    portfolio = PortfolioTracker(config, initial_balance=args.balance)
    
    # Add some initial sample data
    portfolio.add_trade("DEMO_001", "BTCUSDT", "BUY", 0.1, 45000, 2.25, "initial")
    portfolio.add_trade("DEMO_002", "ETHUSDT", "BUY", 2.0, 3000, 6.0, "initial")
    portfolio.add_trade("DEMO_003", "ADAUSDT", "BUY", 1000, 0.5, 0.5, "initial")
    
    # Update with current "market" prices
    portfolio.update_market_prices({
        "BTCUSDT": 46000,
        "ETHUSDT": 3100,
        "ADAUSDT": 0.52
    })
    
    # Run simulation if requested
    if args.simulate:
        simulate_trading_activity(portfolio, args.duration)
    
    # Run selected demo mode
    if args.mode == 'analytics':
        demo_portfolio_analytics(portfolio)
        
    elif args.mode == 'cli':
        demo_cli_interface(portfolio)
        
    elif args.mode == 'web':
        demo_web_interface(portfolio, args.host, args.port)
        
    elif args.mode == 'simulation':
        simulate_trading_activity(portfolio, args.duration)
        demo_portfolio_analytics(portfolio)
    
    print(f"\n✅ Demo completed! Mode: {args.mode}")


if __name__ == "__main__":
    main() 