# High Frequency Trading (HFT) System for Cryptocurrency

A comprehensive, modular High Frequency Trading system built in Python for cryptocurrency markets using the Binance API. This system is designed for **educational purposes and testing with dummy money** on Binance testnet.

## 🚀 Features

- **Real-time Market Data**: WebSocket-based live data feeds from Binance
- **Multiple Trading Strategies**: Mean reversion and momentum strategies with extensible architecture
- **Comprehensive Risk Management**: Multi-layered risk checks and position monitoring
- **Order Management**: High-performance order execution with retry logic and status monitoring
- **Structured Logging**: Detailed logging for performance monitoring and debugging
- **Configurable Parameters**: YAML-based configuration for easy customization
- **Async Architecture**: Built with asyncio for high-performance concurrent operations

## 📁 Project Structure

```
hft_project/
│
├── config/              # Configuration files
│   └── config.yaml      # Main configuration (API keys, trading parameters)
│
├── data/                # Market data storage
│   ├── raw/            # Raw market data
│   ├── processed/      # Processed data
│   └── tick_data/      # Real-time tick data
│
├── research/            # Strategy development and analysis
│   ├── notebooks/      # Jupyter notebooks for research
│   └── backtests/      # Backtesting results
│
├── strategies/          # Trading strategies
│   ├── mean_reversion.py   # Mean reversion strategy
│   └── momentum.py         # Momentum strategy
│
├── execution/           # Order execution engine
│   └── order_manager.py    # Order management system
│
├── market_data/         # Market data components
│   └── feed_handler.py     # Real-time data feed handler
│
├── risk/                # Risk management
│   └── risk_engine.py      # Comprehensive risk engine
│
├── logs/                # System logs
│
├── utils/               # Shared utilities
│   └── logger.py           # Structured logging system
│
├── tests/               # Unit and integration tests
│
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Binance account with API access (testnet recommended)
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd hft_project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv hft_env
   source hft_env/bin/activate  # On Windows: hft_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TA-Lib** (technical analysis library):
   
   **On macOS**:
   ```bash
   brew install ta-lib
   pip install TA-Lib
   ```
   
   **On Ubuntu/Debian**:
   ```bash
   sudo apt-get install build-essential
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   pip install TA-Lib
   ```

5. **Configure API keys**:
   
   Edit `config/config.yaml` and add your Binance testnet API credentials:
   ```yaml
   binance:
     api_key: "YOUR_BINANCE_TESTNET_API_KEY"
     api_secret: "YOUR_BINANCE_TESTNET_SECRET_KEY"
     testnet: true
   ```

   **⚠️ Important**: Always use testnet for development and testing!

## 🔧 Configuration

### Getting Binance Testnet API Keys

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Create an account or log in
3. Generate API keys in the API Management section
4. Add the keys to your `config/config.yaml` file

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `trading.symbols` | Trading pairs to monitor | `["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]` |
| `trading.initial_balance` | Starting capital (testnet money) | `10000.0` |
| `risk.max_portfolio_risk` | Maximum portfolio risk per trade | `0.02` (2%) |
| `risk.max_daily_loss` | Maximum daily loss limit | `0.05` (5%) |
| `risk.stop_loss` | Stop loss percentage | `0.02` (2%) |
| `risk.take_profit` | Take profit percentage | `0.04` (4%) |

## 🚀 Usage

### Basic Usage

1. **Start the HFT system**:
   ```bash
   python main.py
   ```

2. **Monitor the logs**:
   ```bash
   tail -f logs/hft_$(date +%Y%m%d).log
   ```

3. **Stop the system**:
   Press `Ctrl+C` for graceful shutdown

### Running Individual Components

**Test market data feed**:
```python
from market_data.feed_handler import BinanceFeedHandler
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

feed = BinanceFeedHandler(config)
feed.start_streams()
```

**Test strategy signals**:
```python
from strategies.mean_reversion import MeanReversionStrategy
import pandas as pd

strategy = MeanReversionStrategy(config)
# Provide OHLCV data to get signals
signal = strategy.analyze_market_data('BTCUSDT', ohlcv_data)
```

## 📊 Trading Strategies

### Mean Reversion Strategy

- **Logic**: Identifies when prices deviate significantly from their moving average
- **Indicators**: Moving average, standard deviation, RSI, Bollinger Bands
- **Signal**: Buy when price is below mean - 2 standard deviations and RSI < 30

### Momentum Strategy

- **Logic**: Follows trending movements and breakouts
- **Indicators**: Fast/slow moving averages, MACD, RSI, support/resistance levels
- **Signal**: Buy when fast MA > slow MA, MACD bullish, and strong volume

### Adding Custom Strategies

1. Create a new file in `strategies/` directory
2. Implement the strategy class with required methods:
   ```python
   class CustomStrategy:
       def __init__(self, config):
           # Initialize strategy
           pass
       
       def analyze_market_data(self, symbol: str, ohlcv_data: pd.DataFrame) -> Optional[Signal]:
           # Return trading signal or None
           pass
   ```
3. Register the strategy in `main.py`

## 🛡️ Risk Management

The system includes comprehensive risk management:

### Pre-trade Risk Checks
- Position size limits
- Portfolio exposure limits
- Daily loss limits
- Capital adequacy checks
- Trading halt checks

### Real-time Monitoring
- Stop loss/take profit monitoring
- Position size monitoring
- Portfolio risk assessment
- Circuit breaker functionality

### Risk Limits (Configurable)
- Maximum 3 open positions
- 2% stop loss per trade
- 5% maximum daily loss
- 20% maximum position size
- No leverage (for safety)

## 📈 Performance Monitoring

### System Metrics
- Latency monitoring
- Memory and CPU usage
- Message processing rates
- Order execution statistics

### Trading Metrics
- Signal generation rate
- Trade execution success rate
- Win/loss ratios
- Portfolio performance

### Logging
All events are logged with structured data:
- Trade executions
- Signal generations
- Risk check results
- System performance metrics
- Error tracking

## 🔍 Monitoring and Debugging

### Log Files
- `logs/hft_YYYYMMDD.log`: Main system log
- Structured JSON logging for easy parsing
- Automatic log rotation

### Key Metrics to Monitor
- Fill rate percentage
- Risk check success rate
- Latency measurements
- Memory/CPU usage
- Daily PnL

## ⚠️ Important Warnings

1. **This is for educational purposes only**
2. **Always use testnet for development**
3. **Never use real money without thorough testing**
4. **Cryptocurrency trading involves significant risk**
5. **High-frequency trading requires substantial technical expertise**
6. **Regulatory compliance varies by jurisdiction**

## 🧪 Testing

### Running Tests
```bash
python -m pytest tests/
```

### Manual Testing
1. Start with small position sizes
2. Monitor all risk checks carefully
3. Test strategy performance in different market conditions
4. Verify order execution accuracy

## 📚 Additional Resources

### Documentation
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [TA-Lib Documentation](https://ta-lib.org/doc_index.html)
- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

### Recommended Reading
- "Algorithmic Trading" by Ernie Chan
- "High-Frequency Trading" by Irene Aldridge
- "Python for Finance" by Yves Hilpisch

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🆘 Support and Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your API keys and network connection
2. **Import Errors**: Ensure all dependencies are installed
3. **Permission Errors**: Verify file permissions for log directory
4. **Strategy Not Trading**: Check if strategies are enabled in config

### Getting Help

1. Check the logs for error messages
2. Verify configuration settings
3. Test individual components separately
4. Review the strategy logic and parameters

---

**Remember**: This system is designed for learning and experimentation with cryptocurrency trading concepts. Always practice responsible trading and never risk more than you can afford to lose. # Algo-Trading-System
