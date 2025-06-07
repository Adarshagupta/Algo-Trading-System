#!/usr/bin/env python3
"""
Test Options Trading Integration in Full Details Demo
"""

import requests
import json
import time

def test_options_endpoints():
    """Test all options-related API endpoints"""
    base_url = "http://localhost:5003"
    
    endpoints = [
        "/api/options-signals",
        "/api/options-positions", 
        "/api/portfolio-greeks",
        "/api/options-summary"
    ]
    
    print("🧪 Testing Options Trading Integration")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint}: OK")
                
                if endpoint == "/api/options-signals":
                    print(f"   📈 Options Signals: {len(data)} signals")
                    if data:
                        latest = data[-1]
                        print(f"   📊 Latest: {latest.get('underlying')} {latest.get('signal_type')} {latest.get('contract', {}).get('option_type')}")
                
                elif endpoint == "/api/options-positions":
                    print(f"   💼 Options Positions: {len(data)} positions")
                    if data:
                        for key, pos in list(data.items())[:3]:  # Show first 3
                            print(f"   📋 {key}: {pos.get('side')} {pos.get('quantity')} contracts")
                
                elif endpoint == "/api/portfolio-greeks":
                    print(f"   🔢 Portfolio Greeks:")
                    print(f"      Delta: {data.get('total_delta', 0):.3f}")
                    print(f"      Gamma: {data.get('total_gamma', 0):.3f}")
                    print(f"      Theta: {data.get('total_theta', 0):.3f}")
                    print(f"      Vega: {data.get('total_vega', 0):.3f}")
                
                elif endpoint == "/api/options-summary":
                    print(f"   📊 Summary:")
                    print(f"      Total Signals: {data.get('total_signals', 0)}")
                    print(f"      Active Positions: {data.get('active_positions', 0)}")
                    print(f"      Symbols Analyzed: {data.get('symbols_analyzed', 0)}")
                
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: Connection error - {e}")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
        
        print()

def test_options_chain():
    """Test options chain endpoint for a specific symbol"""
    base_url = "http://localhost:5003"
    symbol = "BTCUSDT"
    
    try:
        response = requests.get(f"{base_url}/api/options-chain/{symbol}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Options Chain for {symbol}: {len(data)} contracts")
            
            if data:
                print("   📈 Sample Contracts:")
                for contract in data[:3]:  # Show first 3 contracts
                    print(f"      {contract.get('option_type')} Strike: ${contract.get('strike_price')} "
                          f"Premium: ${contract.get('premium', 0):.4f} "
                          f"Delta: {contract.get('delta', 0):.3f}")
        else:
            print(f"❌ Options Chain {symbol}: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Options Chain {symbol}: Error - {e}")

def main():
    """Main test function"""
    print("🚀 Starting Options Trading Integration Test")
    print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test basic endpoints
    test_options_endpoints()
    
    # Test options chain
    test_options_chain()
    
    print("🏁 Options Trading Integration Test Complete!")

if __name__ == "__main__":
    main() 