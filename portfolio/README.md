# Portfolio Tracking System

A comprehensive, real-time portfolio tracking system for the HFT project that monitors positions, calculates growth, and provides detailed investment analytics through both CLI and web interfaces.

## 🚀 Features

### Core Portfolio Tracking
- **Real-time Position Monitoring**: Track open positions with live P&L calculations
- **Trade History**: Complete record of all trades with timestamps and metadata
- **Growth Analytics**: Portfolio growth tracking from initial investment
- **Commission Tracking**: Monitor trading costs and their impact on returns
- **Multi-Strategy Support**: Track performance by trading strategy

### Advanced Analytics
- **Performance Metrics**: Time-based returns (hourly, daily, weekly, total)
- **Risk Analysis**: Value at Risk (VaR), position concentration, exposure ratios
- **Win/Loss Statistics**: Win rate, largest winner/loser tracking
- **Sharpe Ratio**: Risk-adjusted return calculations
- **Maximum Drawdown**: Peak-to-trough portfolio decline measurement

### User Interfaces

#### 1. Command Line Interface (CLI)
- **Rich Terminal UI**: Beautiful, colorful interface using the Rich library
- **Real-time Updates**: Live updating display with 1-second refresh
- **Comprehensive Dashboard**: Portfolio overview, positions, performance, trades, risk metrics
- **Interactive Commands**: Refresh, export, help functions

#### 2. Web Dashboard
- **Modern Browser Interface**: Responsive web UI with real-time updates
- **WebSocket Integration**: Live data streaming for instant updates
- **Interactive Charts**: Portfolio value visualization over time
- **Export Functionality**: JSON/CSV export directly from the browser
- **Beautiful Design**: Glass-morphism design with animations

### Data Management
- **Automatic Snapshots**: Portfolio state preservation
- **Export Capabilities**: JSON and CSV export formats
- **Historical Data**: Maintains price history and position snapshots
- **Persistence**: Save/load portfolio state to/from files

## 📁 File Structure

```
portfolio/
├── portfolio_tracker.py    # Core portfolio tracking engine
├── portfolio_cli.py        # Command-line interface
├── web_dashboard.py        # Web-based dashboard
├── templates/              # HTML templates for web interface
│   └── dashboard.html      # Main dashboard template
└── README.md              # This file
```

## 🛠️ Installation

The portfolio tracker is integrated into the main HFT system. Dependencies are included in the main `requirements.txt`:

```bash
# Install dependencies
pip install -r requirements.txt

# Additional web dependencies (included in requirements.txt)
# flask==3.0.0
# flask-socketio==5.3.6
# rich==13.7.0
```

## 🚀 Usage

### 1. Integrated with HFT System

The portfolio tracker is automatically integrated into the main HFT system (`main.py`). It tracks all trades and positions automatically:

```python
# Portfolio tracker is initialized automatically
# It records all trades from the order manager
# Updates positions with real-time market data
```

### 2. Command Line Interface

Run the CLI interface for real-time monitoring:

```bash
# Basic CLI
python portfolio/portfolio_cli.py

# Custom balance and update interval
python portfolio/portfolio_cli.py --balance 50000 --update-interval 0.5

# Export data only
python portfolio/portfolio_cli.py --export json
```

### 3. Web Dashboard

Start the web dashboard for browser-based monitoring:

```bash
# Default settings (localhost:5000)
python portfolio/web_dashboard.py

# Custom host and port
python portfolio/web_dashboard.py --host 0.0.0.0 --port 8080

# Create template only
python portfolio/web_dashboard.py --create-template
```

Access the dashboard at: `http://localhost:5000`

### 4. Demonstration Mode

Run the comprehensive demo to see all features:

```bash
# Basic analytics demo
python examples/portfolio_demo.py

# CLI interface demo
python examples/portfolio_demo.py --mode cli

# Web dashboard demo
python examples/portfolio_demo.py --mode web

# Trading simulation demo
python examples/portfolio_demo.py --mode simulation --duration 5

# Full demo with simulation
python examples/portfolio_demo.py --mode analytics --simulate --duration 3
```

## 📊 Dashboard Features

### Portfolio Overview
- **Portfolio Value**: Total cash + market value of positions
- **Portfolio Growth**: Percentage change from initial balance
- **Cash Balance**: Available cash for trading
- **Unrealized P&L**: Current profit/loss on open positions
- **Win Rate**: Percentage of profitable positions

### Position Tracking
- **Open Positions Table**: Real-time position details
- **Symbol**: Trading pair (e.g., BTCUSDT)
- **Side**: BUY or SELL position
- **Quantity**: Amount held
- **Average Entry Price**: Cost basis
- **Current Price**: Live market price
- **Market Value**: Current position value
- **P&L**: Profit/Loss in dollars and percentage
- **Strategy**: Trading strategy used

### Performance Analytics
- **Time-based Returns**:
  - Hourly return
  - Daily return
  - Weekly return
  - Total return since inception
- **Strategy Breakdown**: Performance by trading strategy
- **Trade Statistics**: Total trades, average trade size, commissions

### Risk Metrics
- **Maximum Drawdown**: Largest portfolio decline
- **Sharpe Ratio**: Risk-adjusted performance
- **Position Concentration**: Largest position percentage
- **Cash Ratio**: Percentage of portfolio in cash
- **Value at Risk (VaR)**: Potential loss estimation

### Recent Activity
- **Trade History**: Last 5-10 trades with timestamps
- **Real-time Updates**: Live data streaming
- **Export Functions**: Download portfolio data

## 🎛️ API Reference

### PortfolioTracker Class

```python
from portfolio.portfolio_tracker import PortfolioTracker

# Initialize
portfolio = PortfolioTracker(config, initial_balance=10000.0)

# Add a trade
portfolio.add_trade(
    trade_id="TRADE_001",
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.1,
    price=45000,
    commission=2.25,
    strategy="momentum"
)

# Update market prices
portfolio.update_market_prices({"BTCUSDT": 46000})

# Get portfolio summary
summary = portfolio.get_portfolio_summary()

# Get position details
details = portfolio.get_position_details("BTCUSDT")

# Export data
filename = portfolio.export_data("json")

# Save snapshot
snapshot_file = portfolio.save_snapshot()
```

### Key Methods

#### `add_trade(trade_id, symbol, side, quantity, price, commission, strategy, order_id)`
Records a new trade and updates positions.

#### `update_market_prices(market_data)`
Updates current market prices for all positions.

#### `get_portfolio_summary()`
Returns comprehensive portfolio data including:
- Overview metrics
- Position details
- Recent trades
- Performance analysis
- Risk analysis

#### `get_position_details(symbol)`
Returns detailed information for a specific position:
- Position summary
- Price history
- Related trades
- Statistics (volatility, time in position)

#### `export_data(format)`
Exports portfolio data to file:
- `'json'`: Complete portfolio data
- `'csv'`: Positions table

#### `save_snapshot()`
Saves current portfolio state to timestamped file.

## 📈 Metrics Explained

### Portfolio Metrics
- **Portfolio Value**: Cash + market value of all positions
- **Portfolio Growth**: `((current_value - initial_balance) / initial_balance) × 100`
- **Unrealized P&L**: Current profit/loss on open positions
- **Win Rate**: `(profitable_positions / total_positions) × 100`

### Position Metrics
- **Market Value**: `quantity × current_price`
- **Cost Basis**: `quantity × average_entry_price`
- **Unrealized P&L**: 
  - BUY: `market_value - cost_basis`
  - SELL: `cost_basis - market_value`
- **P&L Percentage**: `(unrealized_pnl / cost_basis) × 100`

### Risk Metrics
- **Sharpe Ratio**: `(mean_return / std_return) × √252`
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (95%)**: 5th percentile of return distribution
- **Position Concentration**: Largest position as % of portfolio

## 🎨 Interface Customization

### CLI Customization
The CLI interface uses Rich library styling. Colors and layouts can be customized in `portfolio_cli.py`:

```python
# Color schemes
positive_color = "green"
negative_color = "red"
neutral_color = "cyan"

# Update intervals
update_interval = 1.0  # seconds
```

### Web Dashboard Customization
The web interface can be customized by modifying the HTML template and CSS:

```bash
# Generate custom template
python portfolio/web_dashboard.py --create-template

# Edit the template
nano portfolio/templates/dashboard.html
```

## 🔧 Integration Examples

### With HFT Main System
```python
# In main.py, portfolio tracker is automatically integrated:
from portfolio.portfolio_tracker import PortfolioTracker

# Initialize in HFTSystem.__init__()
self.portfolio_tracker = PortfolioTracker(self.config, initial_balance)

# Record trades in _on_order_fill()
self.portfolio_tracker.add_trade(...)

# Update prices in trading loop
self.portfolio_tracker.update_market_prices(market_data)
```

### Custom Integration
```python
from portfolio.portfolio_tracker import PortfolioTracker

# Create tracker
portfolio = PortfolioTracker({}, initial_balance=50000)

# Your trading logic
def on_trade_executed(symbol, side, quantity, price, strategy):
    portfolio.add_trade(
        trade_id=generate_trade_id(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        commission=calculate_commission(quantity, price),
        strategy=strategy
    )

# Your price update logic
def on_price_update(market_data):
    portfolio.update_market_prices(market_data)

# Monitor performance
summary = portfolio.get_portfolio_summary()
print(f"Portfolio Growth: {summary['overview']['portfolio_growth']:.2f}%")
```

## 📝 Data Formats

### Portfolio Summary Structure
```json
{
  "overview": {
    "timestamp": "2024-01-01T12:00:00",
    "total_value": 10500.00,
    "cash_balance": 5000.00,
    "invested_amount": 5000.00,
    "total_unrealized_pnl": 500.00,
    "total_unrealized_pnl_percent": 10.0,
    "portfolio_growth": 5.0,
    "position_count": 3,
    "win_rate": 66.7,
    "sharpe_ratio": 1.5,
    "max_drawdown": 2.5
  },
  "positions": [...],
  "recent_trades": [...],
  "performance": {...},
  "risk_analysis": {...}
}
```

### Position Structure
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.1,
  "avg_entry_price": 45000.0,
  "current_price": 46000.0,
  "market_value": 4600.0,
  "cost_basis": 4500.0,
  "unrealized_pnl": 100.0,
  "unrealized_pnl_percent": 2.22,
  "total_commission": 2.25,
  "strategy": "momentum",
  "duration_hours": 24.5
}
```

## 🚨 Important Notes

### Data Persistence
- Portfolio snapshots are automatically saved to `data/portfolio/`
- Exported files include timestamps for version control
- Historical data is maintained in memory with configurable limits

### Performance Considerations
- Position snapshots are limited to 1000 per symbol
- Portfolio snapshots are limited to 10000 total
- Update frequencies should be balanced with system performance

### Risk Management Integration
- Portfolio tracker works with the risk engine
- Real-time position monitoring for risk compliance
- Automatic stop-loss and take-profit tracking

## 🤝 Contributing

When extending the portfolio tracker:

1. **Add new metrics** in `_calculate_portfolio_metrics()`
2. **Extend analytics** in `_get_performance_analysis()` or `_get_risk_analysis()`
3. **Update interfaces** in both CLI and web dashboard
4. **Maintain data structure** compatibility for exports
5. **Add tests** for new functionality

## 📞 Support

For issues with the portfolio tracker:

1. Check the logs in `logs/` directory
2. Verify data structure integrity
3. Test with the demo scripts
4. Review export files for data validation

The portfolio tracking system provides comprehensive monitoring and analytics for your HFT trading activities with beautiful, real-time interfaces for both terminal and web environments. 