#!/usr/bin/env python3
"""
Portfolio CLI - Command Line Interface for Portfolio Tracking
Provides real-time display of positions, performance, and analytics
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_tracker import PortfolioTracker
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.align import Align
import argparse


class PortfolioCLI:
    """Command-line interface for portfolio monitoring"""
    
    def __init__(self, portfolio_tracker: PortfolioTracker):
        self.portfolio = portfolio_tracker
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup the layout for the CLI display"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        self.layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="portfolio_overview", ratio=1),
            Layout(name="positions", ratio=2)
        )
        
        self.layout["right"].split_column(
            Layout(name="performance", ratio=1),
            Layout(name="recent_trades", ratio=1),
            Layout(name="risk_metrics", ratio=1)
        )
    
    def _create_header(self) -> Panel:
        """Create header panel"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text("🚀 HFT PORTFOLIO TRACKER", style="bold blue")
        header_text.append(f" | {current_time}", style="dim")
        return Panel(Align.center(header_text), style="blue")
    
    def _create_portfolio_overview(self, summary: Dict[str, Any]) -> Panel:
        """Create portfolio overview panel"""
        overview = summary['overview']
        
        # Create overview table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Portfolio value and growth
        portfolio_value = overview['total_value']
        growth = overview['portfolio_growth']
        growth_color = "green" if growth >= 0 else "red"
        growth_symbol = "▲" if growth >= 0 else "▼"
        
        # Daily P&L
        daily_pnl = overview['daily_pnl']
        daily_color = "green" if daily_pnl >= 0 else "red"
        daily_symbol = "▲" if daily_pnl >= 0 else "▼"
        
        table.add_row(
            "Portfolio Value", f"${portfolio_value:,.2f}",
            "Cash Balance", f"${overview['cash_balance']:,.2f}"
        )
        table.add_row(
            "Total Growth", f"[{growth_color}]{growth_symbol} {growth:.2f}%[/{growth_color}]",
            "Daily P&L", f"[{daily_color}]{daily_symbol} ${daily_pnl:.2f}[/{daily_color}]"
        )
        table.add_row(
            "Positions", str(overview['position_count']),
            "Win Rate", f"{overview['win_rate']:.1f}%"
        )
        table.add_row(
            "Unrealized P&L", f"${overview['total_unrealized_pnl']:,.2f}",
            "Total Commission", f"${overview['total_commission']:.2f}"
        )
        
        return Panel(table, title="📊 Portfolio Overview", border_style="green")
    
    def _create_positions_table(self, summary: Dict[str, Any]) -> Panel:
        """Create positions table"""
        positions = summary['positions']
        
        if not positions:
            return Panel("No open positions", title="📈 Open Positions", border_style="yellow")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Side", width=6)
        table.add_column("Quantity", justify="right", width=10)
        table.add_column("Avg Price", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("Market Value", justify="right", width=12)
        table.add_column("P&L", justify="right", width=12)
        table.add_column("P&L %", justify="right", width=8)
        table.add_column("Strategy", width=12)
        
        for pos in positions:
            pnl = pos['unrealized_pnl']
            pnl_pct = pos['unrealized_pnl_percent']
            
            # Color coding for P&L
            if pnl > 0:
                pnl_color = "green"
                pnl_symbol = "+"
            elif pnl < 0:
                pnl_color = "red"
                pnl_symbol = ""
            else:
                pnl_color = "white"
                pnl_symbol = ""
            
            side_color = "green" if pos['side'] == "BUY" else "red"
            
            table.add_row(
                pos['symbol'],
                f"[{side_color}]{pos['side']}[/{side_color}]",
                f"{pos['quantity']:.4f}",
                f"${pos['avg_entry_price']:.2f}",
                f"${pos['current_price']:.2f}",
                f"${pos['market_value']:,.2f}",
                f"[{pnl_color}]{pnl_symbol}${abs(pnl):.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_symbol}{abs(pnl_pct):.2f}%[/{pnl_color}]",
                pos['strategy'][:10]
            )
        
        return Panel(table, title="📈 Open Positions", border_style="blue")
    
    def _create_performance_panel(self, summary: Dict[str, Any]) -> Panel:
        """Create performance metrics panel"""
        performance = summary.get('performance', {})
        
        if not performance:
            return Panel("No performance data", title="📊 Performance", border_style="yellow")
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Period", style="cyan")
        table.add_column("Return", style="white")
        
        time_returns = performance.get('time_returns', {})
        
        for period, return_val in time_returns.items():
            color = "green" if return_val >= 0 else "red"
            symbol = "▲" if return_val >= 0 else "▼"
            table.add_row(
                period.capitalize(),
                f"[{color}]{symbol} {return_val:.2f}%[/{color}]"
            )
        
        # Strategy breakdown
        strategy_breakdown = performance.get('strategy_breakdown', {})
        if strategy_breakdown:
            table.add_row("", "")  # Spacer
            table.add_row("[bold]Strategy Performance[/bold]", "")
            for strategy, metrics in strategy_breakdown.items():
                pnl = metrics.get('pnl', 0)
                trades = metrics.get('trades', 0)
                color = "green" if pnl >= 0 else "red"
                table.add_row(
                    strategy[:10],
                    f"[{color}]${pnl:.2f}[/{color}] ({trades} trades)"
                )
        
        return Panel(table, title="📊 Performance", border_style="magenta")
    
    def _create_recent_trades_panel(self, summary: Dict[str, Any]) -> Panel:
        """Create recent trades panel"""
        trades = summary.get('recent_trades', [])
        
        if not trades:
            return Panel("No recent trades", title="📋 Recent Trades", border_style="yellow")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Time", width=8)
        table.add_column("Symbol", width=8)
        table.add_column("Side", width=6)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("Price", justify="right", width=8)
        table.add_column("Strategy", width=10)
        
        for trade in trades[-5:]:  # Show last 5 trades
            timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            time_str = timestamp.strftime("%H:%M:%S")
            
            side_color = "green" if trade['side'] == "BUY" else "red"
            
            table.add_row(
                time_str,
                trade['symbol'],
                f"[{side_color}]{trade['side']}[/{side_color}]",
                f"{trade['quantity']:.4f}",
                f"${trade['price']:.2f}",
                trade['strategy'][:8]
            )
        
        return Panel(table, title="📋 Recent Trades", border_style="cyan")
    
    def _create_risk_metrics_panel(self, summary: Dict[str, Any]) -> Panel:
        """Create risk metrics panel"""
        risk = summary.get('risk_analysis', {})
        overview = summary['overview']
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Portfolio level metrics
        table.add_row("Max Drawdown", f"{overview.get('max_drawdown', 0):.2f}%")
        table.add_row("Sharpe Ratio", f"{overview.get('sharpe_ratio', 0):.2f}")
        
        if risk:
            concentration = risk.get('concentration', {})
            exposure = risk.get('exposure', {})
            
            table.add_row("Largest Position", f"{concentration.get('largest_position_percent', 0):.1f}%")
            table.add_row("Cash Ratio", f"{exposure.get('cash_ratio', 0):.1f}%")
            table.add_row("Total Exposure", f"${exposure.get('total_exposure', 0):,.2f}")
        
        return Panel(table, title="⚠️ Risk Metrics", border_style="red")
    
    def _create_footer(self) -> Panel:
        """Create footer panel"""
        footer_text = Text("Commands: ", style="dim")
        footer_text.append("[q]uit | [r]efresh | [e]xport | [h]elp", style="bold")
        return Panel(Align.center(footer_text), style="dim")
    
    def update_display(self) -> Layout:
        """Update the display with current data"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            
            self.layout["header"].update(self._create_header())
            self.layout["portfolio_overview"].update(self._create_portfolio_overview(summary))
            self.layout["positions"].update(self._create_positions_table(summary))
            self.layout["performance"].update(self._create_performance_panel(summary))
            self.layout["recent_trades"].update(self._create_recent_trades_panel(summary))
            self.layout["risk_metrics"].update(self._create_risk_metrics_panel(summary))
            self.layout["footer"].update(self._create_footer())
            
        except Exception as e:
            error_panel = Panel(f"Error updating display: {str(e)}", 
                              title="Error", border_style="red")
            self.layout["body"].update(error_panel)
        
        return self.layout
    
    def run_live_display(self, update_interval: float = 1.0):
        """Run live updating display"""
        with Live(self.update_display(), refresh_per_second=1) as live:
            try:
                while True:
                    time.sleep(update_interval)
                    live.update(self.update_display())
            except KeyboardInterrupt:
                self.console.print("\n👋 Portfolio tracking stopped", style="yellow")
    
    def show_position_details(self, symbol: str):
        """Show detailed information for a specific position"""
        details = self.portfolio.get_position_details(symbol)
        
        if not details:
            self.console.print(f"No position found for {symbol}", style="red")
            return
        
        position = details['position']
        trades = details['trades']
        stats = details['statistics']
        
        # Position summary
        self.console.print(f"\n📈 Position Details: {symbol}", style="bold blue")
        
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Symbol", position['symbol'])
        overview_table.add_row("Side", position['side'])
        overview_table.add_row("Quantity", f"{position['quantity']:.4f}")
        overview_table.add_row("Avg Entry Price", f"${position['avg_entry_price']:.2f}")
        overview_table.add_row("Current Price", f"${position['current_price']:.2f}")
        overview_table.add_row("Market Value", f"${position['market_value']:,.2f}")
        overview_table.add_row("Unrealized P&L", f"${position['unrealized_pnl']:.2f}")
        overview_table.add_row("P&L Percentage", f"{position['unrealized_pnl_percent']:.2f}%")
        overview_table.add_row("Duration", f"{position['duration_hours']:.1f} hours")
        overview_table.add_row("Max Profit", f"${position['max_profit']:.2f}")
        overview_table.add_row("Max Loss", f"${position['max_loss']:.2f}")
        
        self.console.print(Panel(overview_table, title="Position Summary"))
        
        # Trade history
        if trades:
            self.console.print(f"\n📋 Trade History ({len(trades)} trades)")
            
            trades_table = Table(show_header=True)
            trades_table.add_column("Timestamp")
            trades_table.add_column("Side")
            trades_table.add_column("Quantity", justify="right")
            trades_table.add_column("Price", justify="right")
            trades_table.add_column("Commission", justify="right")
            
            for trade in trades[-10:]:  # Show last 10 trades
                timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                trades_table.add_row(
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    trade['side'],
                    f"{trade['quantity']:.4f}",
                    f"${trade['price']:.2f}",
                    f"${trade['commission']:.2f}"
                )
            
            self.console.print(trades_table)
    
    def export_portfolio_data(self, format: str = 'json'):
        """Export portfolio data"""
        try:
            filename = self.portfolio.export_data(format)
            self.console.print(f"✅ Portfolio data exported to: {filename}", style="green")
        except Exception as e:
            self.console.print(f"❌ Export failed: {str(e)}", style="red")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🚀 HFT Portfolio Tracker - Help

Commands:
  q, quit     - Exit the application
  r, refresh  - Manually refresh the display
  e, export   - Export portfolio data to JSON
  h, help     - Show this help message
  
Position Details:
  position <SYMBOL> - Show detailed information for a specific position
  
Navigation:
  The display updates automatically every second
  Press Ctrl+C to exit gracefully
  
Metrics Explained:
  - Portfolio Value: Total cash + market value of positions
  - Portfolio Growth: Percentage change from initial balance
  - Daily P&L: Profit/Loss for current day
  - Win Rate: Percentage of profitable positions
  - Unrealized P&L: Current profit/loss on open positions
  - Max Drawdown: Largest peak-to-trough decline
  - Sharpe Ratio: Risk-adjusted return measure
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='HFT Portfolio Tracker CLI')
    parser.add_argument('--balance', type=float, default=10000.0,
                       help='Initial portfolio balance (default: 10000)')
    parser.add_argument('--update-interval', type=float, default=1.0,
                       help='Display update interval in seconds (default: 1.0)')
    parser.add_argument('--export', type=str, choices=['json', 'csv'],
                       help='Export portfolio data and exit')
    
    args = parser.parse_args()
    
    # Initialize portfolio tracker with dummy config
    config = {}
    portfolio = PortfolioTracker(config, initial_balance=args.balance)
    
    # Add some sample data for demonstration
    portfolio.add_trade("DEMO_001", "BTCUSDT", "BUY", 0.1, 45000, 2.25, "demo")
    portfolio.add_trade("DEMO_002", "ETHUSDT", "BUY", 2.0, 3000, 6.0, "demo")
    portfolio.update_market_prices({"BTCUSDT": 46000, "ETHUSDT": 3100})
    
    cli = PortfolioCLI(portfolio)
    
    if args.export:
        cli.export_portfolio_data(args.export)
        return
    
    try:
        cli.run_live_display(args.update_interval)
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main() 