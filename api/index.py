#!/usr/bin/env python3
"""
Vercel API Entry Point for HFT Dashboard
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from launch_web_dashboard import LivePortfolioDashboard

# Create Flask app
app = Flask(__name__)

try:
    # Initialize the dashboard
    dashboard_instance = LivePortfolioDashboard()
    
    # Get the Flask app from the dashboard
    dashboard_app = dashboard_instance.portfolio_tracker
    
    @app.route('/')
    def index():
        return '''
        <h1>🚀 HFT Algorithmic Trading System</h1>
        <p>✅ Successfully deployed to Vercel!</p>
        <p>📊 Portfolio Dashboard: $100M Demo Capital</p>
        <p>🔄 Real-time Trading Algorithms Active</p>
        <p>📈 58 Cryptocurrency Pairs</p>
        <p><a href="https://github.com/Adarshagupta/Algo-Trading-System">GitHub Repository</a></p>
        '''
    
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'HFT Trading System'}
    
    @app.route('/api/status')
    def api_status():
        return {
            'status': 'active',
            'system': 'HFT Algorithmic Trading',
            'portfolio': '$100,000,000',
            'pairs': 58,
            'strategies': ['Mean Reversion', 'Momentum', 'Options Trading'],
            'deployed': 'Vercel'
        }

except Exception as e:
    print(f"Dashboard initialization error: {e}")
    
    @app.route('/')
    def index():
        return f'''
        <h1>🚀 HFT Algorithmic Trading System</h1>
        <p>⚠️ Dashboard initialization in progress...</p>
        <p>Error: {str(e)}</p>
        <p><a href="https://github.com/Adarshagupta/Algo-Trading-System">GitHub Repository</a></p>
        '''

# Export for Vercel
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    app.run(debug=True) 