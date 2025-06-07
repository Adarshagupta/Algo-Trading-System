#!/usr/bin/env python3
"""
Web Dashboard for Portfolio Tracking
Simple Flask-based web interface for real-time portfolio monitoring
"""

import json
import os
import sys
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import asyncio
import threading
import time


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_tracker import PortfolioTracker


class PortfolioWebDashboard:
    """Web-based portfolio dashboard"""
    
    def __init__(self, portfolio_tracker: PortfolioTracker, host='localhost', port=5000):
        self.portfolio = portfolio_tracker
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'hft_portfolio_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        
        # Background thread for real-time updates
        self.update_thread = None
        self.running = False
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/portfolio/summary')
        def get_portfolio_summary():
            """API endpoint for portfolio summary"""
            try:
                summary = self.portfolio.get_portfolio_summary()
                return jsonify(summary)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/portfolio/position/<symbol>')
        def get_position_details(symbol):
            """API endpoint for position details"""
            try:
                details = self.portfolio.get_position_details(symbol)
                if details:
                    return jsonify(details)
                else:
                    return jsonify({'error': 'Position not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/portfolio/export/<format>')
        def export_portfolio(format):
            """API endpoint for portfolio export"""
            try:
                if format.lower() not in ['json', 'csv']:
                    return jsonify({'error': 'Invalid format'}), 400
                
                filename = self.portfolio.export_data(format.lower())
                return jsonify({'filename': filename, 'message': 'Export successful'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            """Handle client connection"""
            print(f"Client connected: {request.sid}")
            # Send initial portfolio data
            summary = self.portfolio.get_portfolio_summary()
            # Convert datetime objects to strings for JSON serialization
            summary_json = json.loads(json.dumps(summary, cls=DateTimeEncoder))
            emit('portfolio_update', summary_json)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request"""
            summary = self.portfolio.get_portfolio_summary()
            # Convert datetime objects to strings for JSON serialization
            summary_json = json.loads(json.dumps(summary, cls=DateTimeEncoder))
            emit('portfolio_update', summary_json)
    
    def _background_update_thread(self):
        """Background thread for real-time updates"""
        while self.running:
            try:
                summary = self.portfolio.get_portfolio_summary()
                # Convert datetime objects to strings for JSON serialization
                summary_json = json.loads(json.dumps(summary, cls=DateTimeEncoder))
                self.socketio.emit('portfolio_update', summary_json)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Error in background update: {e}")
                time.sleep(5)
    
    def start(self, debug=False):
        """Start the web dashboard"""
        self.running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._background_update_thread, daemon=True)
        self.update_thread.start()
        
        print(f"🌐 Portfolio Dashboard starting at http://{self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
    
    def stop(self):
        """Stop the web dashboard"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)


def create_html_template():
    """Create the HTML template for the dashboard"""
    template_dir = "portfolio/templates"
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFT Portfolio Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            font-weight: 300;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .card h2 {
            margin-bottom: 1rem;
            color: #00ff88;
            font-weight: 300;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.8rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #ccc;
        }
        
        .metric-value {
            font-weight: 600;
        }
        
        .positive {
            color: #00ff88;
        }
        
        .negative {
            color: #ff4757;
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .positions-table th,
        .positions-table td {
            padding: 0.8rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .positions-table th {
            background: rgba(0,0,0,0.2);
            color: #00ff88;
            font-weight: 600;
        }
        
        .buy {
            color: #00ff88;
        }
        
        .sell {
            color: #ff4757;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .btn {
            padding: 0.8rem 1.5rem;
            border: none;
            border-radius: 8px;
            background: #00ff88;
            color: #1e3c72;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background: #00cc6a;
            transform: translateY(-2px);
        }
        
        .chart-container {
            grid-column: 1 / -1;
            height: 400px;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #ccc;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 HFT Portfolio Dashboard</h1>
        <div class="status">
            <div class="status-dot"></div>
            <span id="lastUpdate">Connecting...</span>
        </div>
    </div>
    
    <div class="container">
        <!-- Portfolio Overview -->
        <div class="card">
            <h2>📊 Portfolio Overview</h2>
            <div id="portfolioOverview" class="loading">Loading...</div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="card">
            <h2>📈 Performance</h2>
            <div id="performanceMetrics" class="loading">Loading...</div>
        </div>
        
        <!-- Open Positions -->
        <div class="card">
            <h2>💼 Open Positions</h2>
            <div id="openPositions" class="loading">Loading...</div>
        </div>
        
        <!-- Recent Trades -->
        <div class="card">
            <h2>📋 Recent Trades</h2>
            <div id="recentTrades" class="loading">Loading...</div>
        </div>
        
        <!-- Portfolio Chart -->
        <div class="card chart-container">
            <h2>📊 Portfolio Value Over Time</h2>
            <canvas id="portfolioChart"></canvas>
        </div>
        
        <!-- Controls -->
        <div class="card">
            <h2>🎛️ Controls</h2>
            <div class="controls">
                <button class="btn" onclick="refreshData()">🔄 Refresh</button>
                <button class="btn" onclick="exportData('json')">📊 Export JSON</button>
                <button class="btn" onclick="exportData('csv')">📄 Export CSV</button>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO connection
        const socket = io();
        
        // Chart instance
        let portfolioChart;
        let portfolioData = [];
        
        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            portfolioChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: 'white',
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    }
                }
            });
        }
        
        // Update portfolio overview
        function updatePortfolioOverview(overview) {
            const container = document.getElementById('portfolioOverview');
            const growth = overview.portfolio_growth;
            const growthClass = growth >= 0 ? 'positive' : 'negative';
            const growthSymbol = growth >= 0 ? '▲' : '▼';
            
            container.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Portfolio Value</span>
                    <span class="metric-value">$${overview.total_value.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Growth</span>
                    <span class="metric-value ${growthClass}">${growthSymbol} ${Math.abs(growth).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cash Balance</span>
                    <span class="metric-value">$${overview.cash_balance.toLocaleString()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Unrealized P&L</span>
                    <span class="metric-value ${overview.total_unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                        $${overview.total_unrealized_pnl.toFixed(2)}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate</span>
                    <span class="metric-value">${overview.win_rate.toFixed(1)}%</span>
                </div>
            `;
        }
        
        // Update positions table
        function updatePositions(positions) {
            const container = document.getElementById('openPositions');
            
            if (positions.length === 0) {
                container.innerHTML = '<p>No open positions</p>';
                return;
            }
            
            let html = `
                <table class="positions-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Avg Price</th>
                            <th>Current</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            positions.forEach(pos => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
                const sideClass = pos.side === 'BUY' ? 'buy' : 'sell';
                
                html += `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td><span class="${sideClass}">${pos.side}</span></td>
                        <td>${pos.quantity.toFixed(4)}</td>
                        <td>$${pos.avg_entry_price.toFixed(2)}</td>
                        <td>$${pos.current_price.toFixed(2)}</td>
                        <td><span class="${pnlClass}">$${pos.unrealized_pnl.toFixed(2)}</span></td>
                        <td><span class="${pnlClass}">${pos.unrealized_pnl_percent.toFixed(2)}%</span></td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        
        // Update recent trades
        function updateRecentTrades(trades) {
            const container = document.getElementById('recentTrades');
            
            if (trades.length === 0) {
                container.innerHTML = '<p>No recent trades</p>';
                return;
            }
            
            let html = '<div>';
            trades.slice(-5).forEach(trade => {
                const timestamp = new Date(trade.timestamp).toLocaleTimeString();
                const sideClass = trade.side === 'BUY' ? 'buy' : 'sell';
                
                html += `
                    <div class="metric">
                        <span class="metric-label">${timestamp} - ${trade.symbol}</span>
                        <span class="metric-value">
                            <span class="${sideClass}">${trade.side}</span> 
                            ${trade.quantity.toFixed(4)} @ $${trade.price.toFixed(2)}
                        </span>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }
        
        // Update performance metrics
        function updatePerformance(performance) {
            const container = document.getElementById('performanceMetrics');
            
            if (!performance.time_returns) {
                container.innerHTML = '<p>No performance data</p>';
                return;
            }
            
            const returns = performance.time_returns;
            let html = '<div>';
            
            Object.entries(returns).forEach(([period, value]) => {
                const returnClass = value >= 0 ? 'positive' : 'negative';
                const symbol = value >= 0 ? '▲' : '▼';
                
                html += `
                    <div class="metric">
                        <span class="metric-label">${period.charAt(0).toUpperCase() + period.slice(1)}</span>
                        <span class="metric-value ${returnClass}">${symbol} ${Math.abs(value).toFixed(2)}%</span>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        // Update chart
        function updateChart(overview) {
            const now = new Date().toLocaleTimeString();
            portfolioData.push({
                time: now,
                value: overview.total_value
            });
            
            // Keep only last 50 data points
            if (portfolioData.length > 50) {
                portfolioData.shift();
            }
            
            portfolioChart.data.labels = portfolioData.map(d => d.time);
            portfolioChart.data.datasets[0].data = portfolioData.map(d => d.value);
            portfolioChart.update('none');
        }
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('lastUpdate').textContent = 'Connected';
        });
        
        socket.on('portfolio_update', function(data) {
            console.log('Portfolio update received');
            document.getElementById('lastUpdate').textContent = 
                'Last update: ' + new Date().toLocaleTimeString();
            
            updatePortfolioOverview(data.overview);
            updatePositions(data.positions);
            updateRecentTrades(data.recent_trades);
            updatePerformance(data.performance);
            updateChart(data.overview);
        });
        
        // Control functions
        function refreshData() {
            socket.emit('request_update');
        }
        
        function exportData(format) {
            fetch(`/api/portfolio/export/${format}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('Export failed: ' + data.error);
                    } else {
                        alert('Export successful: ' + data.filename);
                    }
                })
                .catch(error => {
                    alert('Export failed: ' + error);
                });
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
        });
    </script>
</body>
</html>'''
    
    with open(f"{template_dir}/dashboard.html", 'w') as f:
        f.write(html_content)
    
    print(f"HTML template created at {template_dir}/dashboard.html")


def main():
    """Main function to run the web dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HFT Portfolio Web Dashboard')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--create-template', action='store_true', 
                       help='Create HTML template and exit')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_html_template()
        return
    
    # Create HTML template if it doesn't exist
    if not os.path.exists('portfolio/templates/dashboard.html'):
        create_html_template()
    
    # Initialize portfolio tracker with sample data
    config = {}
    portfolio = PortfolioTracker(config, initial_balance=args.balance)
    
    # Add sample data
    portfolio.add_trade("WEB_001", "BTCUSDT", "BUY", 0.1, 45000, 2.25, "demo")
    portfolio.add_trade("WEB_002", "ETHUSDT", "BUY", 2.0, 3000, 6.0, "demo")
    portfolio.update_market_prices({"BTCUSDT": 46000, "ETHUSDT": 3100})
    
    # Start dashboard
    dashboard = PortfolioWebDashboard(portfolio, host=args.host, port=args.port)
    
    try:
        dashboard.start(debug=False)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        dashboard.stop()


if __name__ == "__main__":
    main() 