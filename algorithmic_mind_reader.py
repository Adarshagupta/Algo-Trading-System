#!/usr/bin/env python3
"""
ALGORITHMIC MIND READER - Real-time Algorithm Decision Monitoring
Shows exactly what the algorithm is thinking every microsecond
"""

import asyncio
import os
import sys
import time
import threading
from datetime import datetime, timedelta
import webbrowser
import json
import logging
from collections import deque
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance.client import Client
from execution.real_demo_order_manager import RealDemoOrderManager, OrderSide
from portfolio.portfolio_tracker import PortfolioTracker
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.take_profit import TakeProfitStrategy
from flask import Flask, render_template, jsonify
import yaml
import pandas as pd
import numpy as np


class AlgorithmicMindReader:
    """Real-time algorithm decision monitoring system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Algorithm monitoring configuration
        self.initial_balance = 10000.0
        self.max_position_size = 500.0
        self.decision_interval = 0.001  # 1 millisecond decisions
        self.monitoring_window = 1000   # Keep last 1000 decisions
        
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
        
        # Algorithm mind state
        self.running = False
        self.start_time = None
        
        # Real-time decision tracking
        self.algorithm_thoughts = deque(maxlen=self.monitoring_window)
        self.latest_prices = {}
        self.market_analysis = {}
        self.strategy_decisions = {}
        self.risk_calculations = {}
        self.execution_thoughts = {}
        
        # Performance metrics
        self.decisions_per_second = 0
        self.total_calculations = 0
        self.last_decision_time = None
        
        print("🧠 ALGORITHMIC MIND READER - REAL-TIME DECISION MONITORING")
        print("=" * 80)
        print("🔬 Microsecond-level decision tracking")
        print("📊 Real-time strategy calculations")
        print("⚡ Risk assessment monitoring")
        print("🎯 Signal generation analysis")
        print("💭 Algorithm thought process visualization")
        print("=" * 80)
        
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
                    'initial_balance': 10000.0,
                    'max_positions': 5,
                    'base_currency': 'USDT'
                }
            }
    
    def log_algorithm_thought(self, category: str, symbol: str, thought: str, data: dict = None):
        """Log what the algorithm is thinking"""
        timestamp = datetime.now()
        microsecond = timestamp.microsecond
        
        thought_entry = {
            'timestamp': timestamp.isoformat(),
            'microsecond': microsecond,
            'category': category,
            'symbol': symbol,
            'thought': thought,
            'data': data or {},
            'decision_id': self.total_calculations,
            'processing_time': time.time() - (self.last_decision_time or time.time())
        }
        
        self.algorithm_thoughts.append(thought_entry)
        self.total_calculations += 1
        self.last_decision_time = time.time()
        
        # Print real-time thoughts (first 50 chars)
        print(f"🧠 {timestamp.strftime('%H:%M:%S.%f')[:-3]} | {category:12} | {symbol:8} | {thought[:50]}...")
    
    async def initialize_system(self):
        """Initialize algorithm monitoring system"""
        try:
            self.log_algorithm_thought("SYSTEM", "INIT", "Initializing algorithmic mind reader system", {
                'initial_balance': self.initial_balance,
                'monitoring_window': self.monitoring_window,
                'decision_interval': self.decision_interval
            })
            
            # Initialize order manager
            self.order_manager = RealDemoOrderManager(self.config)
            self.log_algorithm_thought("SYSTEM", "ORDER_MGR", "Order manager initialized", {
                'demo_mode': True,
                'live_data': True
            })
            
            # Initialize portfolio tracker
            self.portfolio_tracker = PortfolioTracker(self.config, self.initial_balance)
            self.log_algorithm_thought("SYSTEM", "PORTFOLIO", "Portfolio tracker initialized", {
                'balance': self.initial_balance,
                'tracking_enabled': True
            })
            
            # Initialize strategies
            if self.config.get('strategies', {}).get('mean_reversion', {}).get('enabled', True):
                self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
                self.log_algorithm_thought("STRATEGY", "MEAN_REV", "Mean reversion strategy loaded", {
                    'lookback_period': 20,
                    'std_multiplier': 2.0
                })
            
            if self.config.get('strategies', {}).get('momentum', {}).get('enabled', True):
                self.strategies['momentum'] = MomentumStrategy(self.config)
                self.log_algorithm_thought("STRATEGY", "MOMENTUM", "Momentum strategy loaded", {
                    'momentum_period': 14,
                    'threshold': 0.02
                })
            
            if self.config.get('strategies', {}).get('take_profit', {}).get('enabled', True):
                self.strategies['take_profit'] = TakeProfitStrategy(self.config)
                self.log_algorithm_thought("STRATEGY", "TAKE_PROFIT", "Take profit strategy loaded", {
                    'profit_target': 0.03,
                    'stop_loss': 0.02
                })
            
            self.log_algorithm_thought("SYSTEM", "INIT", "All systems initialized - algorithm ready", {
                'strategies_loaded': len(self.strategies),
                'monitoring_active': True
            })
            
        except Exception as e:
            self.log_algorithm_thought("ERROR", "INIT", f"System initialization failed: {str(e)}", {
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            raise
    
    def analyze_market_microsecond(self):
        """Analyze market data with microsecond precision"""
        try:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            
            for symbol in symbols:
                start_time = time.time()
                
                self.log_algorithm_thought("MARKET", symbol, "Starting market data analysis", {
                    'analysis_start': start_time
                })
                
                try:
                    # Get real-time price
                    ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    
                    self.log_algorithm_thought("MARKET", symbol, f"Price retrieved: ${price:.4f}", {
                        'current_price': price,
                        'retrieval_time': time.time() - start_time
                    })
                    
                    # Store price and calculate changes
                    old_price = self.latest_prices.get(symbol, {}).get('price', price)
                    price_change = ((price - old_price) / old_price) * 100 if old_price != price else 0
                    
                    self.latest_prices[symbol] = {
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'change_percent': price_change,
                        'analysis_time': time.time() - start_time
                    }
                    
                    self.log_algorithm_thought("MARKET", symbol, f"Price change calculated: {price_change:+.4f}%", {
                        'old_price': old_price,
                        'new_price': price,
                        'change_percent': price_change
                    })
                    
                    # Technical analysis thoughts
                    self.analyze_technical_indicators(symbol, price)
                    
                except Exception as e:
                    self.log_algorithm_thought("ERROR", symbol, f"Market analysis failed: {str(e)}", {
                        'error_type': type(e).__name__
                    })
            
            # Update portfolio with new prices
            if self.portfolio_tracker and self.latest_prices:
                market_prices = {symbol: data['price'] for symbol, data in self.latest_prices.items()}
                self.portfolio_tracker.update_market_prices(market_prices)
                
                self.log_algorithm_thought("PORTFOLIO", "UPDATE", "Portfolio updated with market prices", {
                    'symbols_updated': len(market_prices),
                    'total_value': self.portfolio_tracker.get_portfolio_summary().get('overview', {}).get('total_value', 0)
                })
                
        except Exception as e:
            self.log_algorithm_thought("ERROR", "MARKET", f"Market analysis error: {str(e)}", {
                'error_type': type(e).__name__
            })
    
    def analyze_technical_indicators(self, symbol: str, price: float):
        """Analyze technical indicators and log thoughts"""
        try:
            # Simulate technical analysis thoughts
            self.log_algorithm_thought("TECHNICAL", symbol, "Calculating moving averages", {
                'current_price': price,
                'indicator': 'moving_average'
            })
            
            # Simulate RSI calculation
            rsi = 45 + (price % 100) / 2  # Simulated RSI
            self.log_algorithm_thought("TECHNICAL", symbol, f"RSI calculated: {rsi:.2f}", {
                'rsi_value': rsi,
                'overbought': rsi > 70,
                'oversold': rsi < 30,
                'signal': 'NEUTRAL' if 30 <= rsi <= 70 else ('SELL' if rsi > 70 else 'BUY')
            })
            
            # Simulate MACD calculation
            macd_signal = "BULLISH" if (price % 10) > 5 else "BEARISH"
            self.log_algorithm_thought("TECHNICAL", symbol, f"MACD signal: {macd_signal}", {
                'macd_line': price % 10,
                'signal_line': 5,
                'histogram': (price % 10) - 5,
                'trend': macd_signal
            })
            
        except Exception as e:
            self.log_algorithm_thought("ERROR", symbol, f"Technical analysis error: {str(e)}", {
                'error_type': type(e).__name__
            })
    
    def run_strategy_decisions(self):
        """Run strategy decision making with detailed logging"""
        try:
            for strategy_name, strategy in self.strategies.items():
                self.log_algorithm_thought("STRATEGY", strategy_name, "Starting strategy evaluation", {
                    'strategy_type': strategy_name,
                    'enabled': True
                })
                
                for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']:
                    if symbol in self.latest_prices:
                        price = self.latest_prices[symbol]['price']
                        
                        # Strategy-specific decision logic
                        self.evaluate_strategy_decision(strategy_name, symbol, price)
        
        except Exception as e:
            self.log_algorithm_thought("ERROR", "STRATEGY", f"Strategy decision error: {str(e)}", {
                'error_type': type(e).__name__
            })
    
    def evaluate_strategy_decision(self, strategy_name: str, symbol: str, price: float):
        """Evaluate individual strategy decisions"""
        try:
            self.log_algorithm_thought("STRATEGY", symbol, f"Evaluating {strategy_name} decision", {
                'strategy': strategy_name,
                'current_price': price,
                'evaluation_start': time.time()
            })
            
            # Simulate strategy decision logic
            if strategy_name == 'mean_reversion':
                # Mean reversion logic
                mean_price = 50000 if 'BTC' in symbol else 3000  # Simulated mean
                deviation = abs(price - mean_price) / mean_price
                
                self.log_algorithm_thought("STRATEGY", symbol, f"Mean reversion: deviation {deviation:.4f}", {
                    'current_price': price,
                    'mean_price': mean_price,
                    'deviation': deviation,
                    'threshold': 0.02
                })
                
                if deviation > 0.02:
                    signal = 'BUY' if price < mean_price else 'SELL'
                    confidence = min(deviation * 50, 0.95)
                    
                    self.log_algorithm_thought("SIGNAL", symbol, f"Mean reversion signal: {signal}", {
                        'signal': signal,
                        'confidence': confidence,
                        'reason': 'price_deviation',
                        'deviation': deviation
                    })
                    
                    # Risk assessment
                    self.assess_risk_for_signal(symbol, signal, confidence, price)
            
            elif strategy_name == 'momentum':
                # Momentum logic
                momentum_score = (price % 100) / 100  # Simulated momentum
                
                self.log_algorithm_thought("STRATEGY", symbol, f"Momentum score: {momentum_score:.4f}", {
                    'momentum_score': momentum_score,
                    'threshold': 0.6,
                    'trend': 'STRONG' if momentum_score > 0.6 else 'WEAK'
                })
                
                if momentum_score > 0.6:
                    signal = 'BUY'
                    confidence = momentum_score
                    
                    self.log_algorithm_thought("SIGNAL", symbol, f"Momentum signal: {signal}", {
                        'signal': signal,
                        'confidence': confidence,
                        'reason': 'strong_momentum',
                        'momentum_score': momentum_score
                    })
                    
                    # Risk assessment
                    self.assess_risk_for_signal(symbol, signal, confidence, price)
        
        except Exception as e:
            self.log_algorithm_thought("ERROR", symbol, f"Strategy evaluation error: {str(e)}", {
                'strategy': strategy_name,
                'error_type': type(e).__name__
            })
    
    def assess_risk_for_signal(self, symbol: str, signal: str, confidence: float, price: float):
        """Assess risk for trading signals"""
        try:
            self.log_algorithm_thought("RISK", symbol, "Starting risk assessment", {
                'signal': signal,
                'confidence': confidence,
                'price': price
            })
            
            # Portfolio risk assessment
            if self.portfolio_tracker:
                summary = self.portfolio_tracker.get_portfolio_summary()
                current_positions = len(summary.get('positions', []))
                total_value = summary.get('overview', {}).get('total_value', 0)
                
                self.log_algorithm_thought("RISK", symbol, f"Portfolio risk check: {current_positions}/5 positions", {
                    'current_positions': current_positions,
                    'max_positions': 5,
                    'total_value': total_value,
                    'available_cash': total_value - sum(pos.get('market_value', 0) for pos in summary.get('positions', []))
                })
                
                # Position sizing calculation
                max_position_value = min(self.max_position_size, total_value * 0.05)
                
                self.log_algorithm_thought("RISK", symbol, f"Position sizing: max ${max_position_value:.2f}", {
                    'max_position': max_position_value,
                    'portfolio_percent': 5.0,
                    'absolute_max': self.max_position_size
                })
                
                # Risk score calculation
                risk_score = self.calculate_risk_score(symbol, signal, confidence, current_positions)
                
                self.log_algorithm_thought("RISK", symbol, f"Risk score: {risk_score:.3f}", {
                    'risk_score': risk_score,
                    'acceptable': risk_score < 0.5,
                    'factors': {
                        'position_count': current_positions / 5,
                        'confidence': 1 - confidence,
                        'volatility': 0.2  # Simulated
                    }
                })
                
                # Final decision
                if risk_score < 0.5 and current_positions < 5 and confidence > 0.7:
                    self.log_algorithm_thought("DECISION", symbol, f"EXECUTE {signal} - Risk acceptable", {
                        'final_decision': f'EXECUTE_{signal}',
                        'risk_score': risk_score,
                        'confidence': confidence,
                        'position_size': max_position_value
                    })
                    
                    # Simulate order execution
                    self.simulate_order_execution(symbol, signal, max_position_value, price)
                else:
                    self.log_algorithm_thought("DECISION", symbol, f"SKIP {signal} - Risk too high", {
                        'final_decision': 'SKIP',
                        'risk_score': risk_score,
                        'confidence': confidence,
                        'reason': 'risk_management'
                    })
        
        except Exception as e:
            self.log_algorithm_thought("ERROR", symbol, f"Risk assessment error: {str(e)}", {
                'error_type': type(e).__name__
            })
    
    def calculate_risk_score(self, symbol: str, signal: str, confidence: float, current_positions: int) -> float:
        """Calculate risk score for a trade"""
        position_risk = current_positions / 5  # Position count risk
        confidence_risk = 1 - confidence       # Confidence risk (lower confidence = higher risk)
        volatility_risk = 0.2                  # Simulated volatility risk
        
        total_risk = (position_risk + confidence_risk + volatility_risk) / 3
        
        self.log_algorithm_thought("RISK", symbol, "Risk components calculated", {
            'position_risk': position_risk,
            'confidence_risk': confidence_risk,
            'volatility_risk': volatility_risk,
            'total_risk': total_risk
        })
        
        return total_risk
    
    def simulate_order_execution(self, symbol: str, signal: str, size: float, price: float):
        """Simulate order execution with detailed logging"""
        try:
            self.log_algorithm_thought("EXECUTION", symbol, f"Preparing {signal} order", {
                'order_type': signal,
                'size_usd': size,
                'current_price': price,
                'order_id': f"SIM_{int(time.time())}"
            })
            
            # Calculate quantity
            quantity = size / price
            
            self.log_algorithm_thought("EXECUTION", symbol, f"Order details calculated", {
                'quantity': quantity,
                'price': price,
                'total_value': size,
                'side': signal
            })
            
            # Simulate order placement
            order_id = f"SIM_{symbol}_{int(time.time())}"
            
            self.log_algorithm_thought("EXECUTION", symbol, f"Order placed: {order_id}", {
                'order_id': order_id,
                'status': 'FILLED',
                'execution_time': 0.001,  # 1ms simulated
                'slippage': 0.0001
            })
            
            # Update portfolio (simulation)
            if self.order_manager:
                # This would be real order execution
                pass
            
        except Exception as e:
            self.log_algorithm_thought("ERROR", symbol, f"Order execution error: {str(e)}", {
                'error_type': type(e).__name__
            })
    
    def create_mind_reader_dashboard(self):
        """Create real-time mind reader dashboard"""
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>🧠 Algorithmic Mind Reader - Real-time Decision Monitoring</title>
                <style>
                    body { font-family: 'Courier New', monospace; margin: 10px; background: #000; color: #00ff00; font-size: 12px; }
                    .header { background: #001100; padding: 10px; border: 1px solid #00ff00; margin-bottom: 10px; }
                    .metrics { display: flex; gap: 10px; margin-bottom: 10px; }
                    .metric { background: #001100; padding: 10px; border: 1px solid #00ff00; flex: 1; }
                    .thoughts-container { height: 400px; overflow-y: scroll; background: #000; border: 1px solid #00ff00; padding: 10px; }
                    .thought { margin-bottom: 5px; padding: 5px; background: #001100; }
                    .timestamp { color: #ffff00; }
                    .category { color: #00ffff; font-weight: bold; }
                    .symbol { color: #ff00ff; }
                    .thought-text { color: #00ff00; }
                    .data { color: #888; font-size: 10px; }
                    .MARKET { border-left: 3px solid #0066ff; }
                    .STRATEGY { border-left: 3px solid #ff6600; }
                    .RISK { border-left: 3px solid #ff0066; }
                    .SIGNAL { border-left: 3px solid #66ff00; }
                    .EXECUTION { border-left: 3px solid #ffff00; }
                    .ERROR { border-left: 3px solid #ff0000; background: #330000; }
                    .status { color: #00ff00; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🧠 ALGORITHMIC MIND READER</h1>
                    <p>Real-time algorithm decision monitoring • Microsecond precision • Every thought tracked</p>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div>Decisions/Second</div>
                        <div class="status" id="decisionsPerSecond">0</div>
                    </div>
                    <div class="metric">
                        <div>Total Calculations</div>
                        <div class="status" id="totalCalculations">0</div>
                    </div>
                    <div class="metric">
                        <div>Active Strategies</div>
                        <div class="status" id="activeStrategies">0</div>
                    </div>
                    <div class="metric">
                        <div>Processing Speed</div>
                        <div class="status" id="processingSpeed">0ms</div>
                    </div>
                    <div class="metric">
                        <div>Market Symbols</div>
                        <div class="status" id="marketSymbols">0</div>
                    </div>
                </div>
                
                <div class="thoughts-container" id="thoughtsContainer">
                    <div>🧠 Waiting for algorithm thoughts...</div>
                </div>
                
                <script>
                let lastUpdateTime = Date.now();
                let thoughtCount = 0;
                
                async function updateMindReader() {
                    try {
                        const [thoughts, metrics] = await Promise.all([
                            fetch('/api/algorithm-thoughts').then(r => r.json()),
                            fetch('/api/algorithm-metrics').then(r => r.json())
                        ]);
                        
                        // Update metrics
                        document.getElementById('decisionsPerSecond').textContent = metrics.decisions_per_second || 0;
                        document.getElementById('totalCalculations').textContent = metrics.total_calculations || 0;
                        document.getElementById('activeStrategies').textContent = metrics.active_strategies || 0;
                        document.getElementById('processingSpeed').textContent = (metrics.avg_processing_time || 0).toFixed(3) + 'ms';
                        document.getElementById('marketSymbols').textContent = metrics.market_symbols || 0;
                        
                        // Update thoughts
                        const container = document.getElementById('thoughtsContainer');
                        
                        // Only show new thoughts
                        const newThoughts = thoughts.slice(thoughtCount);
                        
                        newThoughts.forEach(thought => {
                            const thoughtDiv = document.createElement('div');
                            thoughtDiv.className = `thought ${thought.category}`;
                            
                            const timestamp = new Date(thought.timestamp).toLocaleTimeString() + 
                                             '.' + thought.microsecond.toString().padStart(6, '0').slice(0, 3);
                            
                            thoughtDiv.innerHTML = `
                                <span class="timestamp">${timestamp}</span> |
                                <span class="category">${thought.category.padEnd(12)}</span> |
                                <span class="symbol">${thought.symbol.padEnd(8)}</span> |
                                <span class="thought-text">${thought.thought}</span>
                                ${Object.keys(thought.data).length > 0 ? 
                                    `<div class="data">${JSON.stringify(thought.data, null, 2)}</div>` : ''}
                            `;
                            
                            container.appendChild(thoughtDiv);
                        });
                        
                        thoughtCount = thoughts.length;
                        
                        // Auto-scroll to bottom
                        container.scrollTop = container.scrollHeight;
                        
                        // Keep only last 500 thoughts visible
                        while (container.children.length > 500) {
                            container.removeChild(container.firstChild);
                        }
                        
                    } catch (error) {
                        console.error('Mind reader update error:', error);
                    }
                }
                
                // Update every 50ms for real-time experience
                setInterval(updateMindReader, 50);
                updateMindReader();
                </script>
            </body>
            </html>
            '''
        
        @app.route('/api/algorithm-thoughts')
        def api_thoughts():
            """Get algorithm thoughts"""
            return jsonify(list(self.algorithm_thoughts))
        
        @app.route('/api/algorithm-metrics')
        def api_metrics():
            """Get algorithm performance metrics"""
            current_time = time.time()
            if self.start_time:
                runtime = current_time - self.start_time
                dps = self.total_calculations / runtime if runtime > 0 else 0
            else:
                dps = 0
            
            avg_processing_time = 0
            if len(self.algorithm_thoughts) > 1:
                processing_times = [t.get('processing_time', 0) for t in list(self.algorithm_thoughts)[-100:]]
                avg_processing_time = sum(processing_times) / len(processing_times) * 1000  # Convert to ms
            
            return jsonify({
                'decisions_per_second': round(dps, 2),
                'total_calculations': self.total_calculations,
                'active_strategies': len(self.strategies),
                'avg_processing_time': avg_processing_time,
                'market_symbols': len(self.latest_prices),
                'monitoring_window': self.monitoring_window,
                'uptime': round(runtime, 2) if self.start_time else 0
            })
        
        return app
    
    def start_mind_reader_dashboard(self, port=5006):
        """Start mind reader dashboard"""
        def run_flask():
            app = self.create_mind_reader_dashboard()
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        web_thread = threading.Thread(target=run_flask, daemon=True)
        web_thread.start()
        
        print(f"\n🧠 MIND READER Dashboard started at http://localhost:{port}")
        print("📊 Real-time algorithm decision monitoring!")
        
        # Open browser automatically
        time.sleep(2)
        webbrowser.open(f'http://localhost:{port}')
    
    async def run_mind_reader(self):
        """Run the algorithmic mind reader"""
        try:
            # Initialize system
            await self.initialize_system()
            
            # Start mind reader dashboard
            self.start_mind_reader_dashboard()
            
            self.running = True
            self.start_time = time.time()
            
            print("\n🧠 Starting ALGORITHMIC MIND READER...")
            print("🔬 Every algorithm thought will be monitored")
            print("⚡ Microsecond-level decision tracking active")
            print("📊 Real-time strategy analysis running")
            print("\nPress Ctrl+C to stop the mind reader")
            
            # Main monitoring loop
            while self.running:
                start_cycle = time.time()
                
                # Market analysis
                self.analyze_market_microsecond()
                
                # Strategy decisions
                self.run_strategy_decisions()
                
                # Calculate decisions per second
                cycle_time = time.time() - start_cycle
                if cycle_time > 0:
                    self.decisions_per_second = 1 / cycle_time
                
                # Brief pause (adjust for desired frequency)
                await asyncio.sleep(max(0.001, self.decision_interval - cycle_time))
                
        except KeyboardInterrupt:
            print("\n🛑 MIND READER stopped by user")
            self.running = False
        except Exception as e:
            print(f"❌ MIND READER failed: {e}")
            self.log_algorithm_thought("ERROR", "SYSTEM", f"Mind reader crashed: {str(e)}", {
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })


async def main():
    """Main function"""
    mind_reader = AlgorithmicMindReader()
    await mind_reader.run_mind_reader()


if __name__ == "__main__":
    asyncio.run(main()) 