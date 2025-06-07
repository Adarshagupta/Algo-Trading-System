#!/usr/bin/env python3
"""
Analytics Fix - Diagnose and correct portfolio calculation mismatches
Identifies and fixes calculation errors in the HFT system
"""

import json
import requests
import yaml
from datetime import datetime
from typing import Dict, Any

class AnalyticsFix:
    """Fix portfolio calculation mismatches"""
    
    def __init__(self):
        self.ports = [5000, 5001, 5002, 5003]
        self.results = {}
        
    def load_config(self):
        """Load configuration"""
        try:
            with open('config/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except:
            return None
    
    def check_running_instances(self):
        """Check all running instances and their data"""
        print("🔍 ANALYTICS CALCULATION AUDIT")
        print("="*60)
        
        config = self.load_config()
        expected_balance = config['trading']['initial_balance'] if config else 100000000.0
        print(f"📋 Expected Initial Balance: ${expected_balance:,.0f}")
        
        for port in self.ports:
            try:
                url = f"http://localhost:{port}/api/portfolio"
                response = requests.get(url, timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    overview = data.get('overview', {})
                    
                    print(f"\n🚨 PORT {port} - CALCULATION ANALYSIS:")
                    print(f"  Portfolio Value: ${overview.get('total_value', 0):,.2f}")
                    print(f"  Cash Balance: ${overview.get('cash_balance', 0):,.2f}")
                    print(f"  Growth: {overview.get('portfolio_growth', 0):+.2f}%")
                    print(f"  P&L: ${overview.get('total_unrealized_pnl', 0):+,.2f}")
                    print(f"  Positions: {overview.get('position_count', 0)}")
                    print(f"  Win Rate: {overview.get('win_rate', 0):.1f}%")
                    
                    # Validate calculations
                    self.validate_calculations(port, data, expected_balance)
                    
                    self.results[port] = {
                        'status': 'running',
                        'data': overview,
                        'validation': self.validate_calculations(port, data, expected_balance)
                    }
                    
            except Exception as e:
                print(f"\n❌ PORT {port}: Not accessible ({e})")
                self.results[port] = {'status': 'error', 'error': str(e)}
    
    def validate_calculations(self, port: int, data: Dict[str, Any], expected_balance: float) -> Dict[str, bool]:
        """Validate portfolio calculations"""
        overview = data.get('overview', {})
        positions = data.get('positions', [])
        
        # Basic validation
        total_value = overview.get('total_value', 0)
        cash_balance = overview.get('cash_balance', 0)
        portfolio_growth = overview.get('portfolio_growth', 0)
        unrealized_pnl = overview.get('total_unrealized_pnl', 0)
        
        validation = {
            'initial_balance_correct': False,
            'growth_calculation_correct': False,
            'position_math_correct': False,
            'reasonable_values': False,
            'data_consistency': False
        }
        
        # Check initial balance alignment
        growth_decimal = portfolio_growth / 100
        calculated_initial = total_value / (1 + growth_decimal) if growth_decimal != -1 else expected_balance
        validation['initial_balance_correct'] = abs(calculated_initial - expected_balance) < 1000
        
        # Check growth calculation
        if expected_balance > 0:
            calculated_growth = ((total_value - expected_balance) / expected_balance) * 100
            validation['growth_calculation_correct'] = abs(calculated_growth - portfolio_growth) < 0.01
        
        # Check position math
        if positions:
            total_market_value = sum(pos.get('market_value', 0) for pos in positions)
            calculated_total = cash_balance + total_market_value
            validation['position_math_correct'] = abs(calculated_total - total_value) < 100
        
        # Check for reasonable values
        validation['reasonable_values'] = (
            total_value >= 0 and 
            cash_balance >= 0 and 
            abs(portfolio_growth) < 1000 and  # Growth shouldn't be > 1000%
            len(positions) <= 20  # Reasonable position count
        )
        
        # Data consistency check
        validation['data_consistency'] = all([
            'total_value' in overview,
            'cash_balance' in overview,
            'portfolio_growth' in overview,
            isinstance(positions, list)
        ])
        
        print(f"    ✅ Initial Balance Check: {'PASS' if validation['initial_balance_correct'] else 'FAIL'}")
        print(f"    ✅ Growth Calculation: {'PASS' if validation['growth_calculation_correct'] else 'FAIL'}")
        print(f"    ✅ Position Math: {'PASS' if validation['position_math_correct'] else 'FAIL'}")
        print(f"    ✅ Reasonable Values: {'PASS' if validation['reasonable_values'] else 'FAIL'}")
        print(f"    ✅ Data Consistency: {'PASS' if validation['data_consistency'] else 'FAIL'}")
        
        return validation
    
    def identify_problems(self):
        """Identify specific calculation problems"""
        print(f"\n🔧 PROBLEM IDENTIFICATION:")
        print("="*60)
        
        running_instances = [port for port, result in self.results.items() if result.get('status') == 'running']
        
        if len(running_instances) > 1:
            print(f"⚠️ Multiple instances running on ports: {running_instances}")
            print("  This can cause data inconsistencies!")
            
            # Compare values across instances
            values = {}
            for port in running_instances:
                data = self.results[port]['data']
                values[port] = {
                    'portfolio_value': data.get('total_value', 0),
                    'growth': data.get('portfolio_growth', 0),
                    'positions': data.get('position_count', 0)
                }
            
            print(f"\n📊 VALUE COMPARISON:")
            for metric in ['portfolio_value', 'growth', 'positions']:
                print(f"  {metric.title()}:")
                for port, data in values.items():
                    print(f"    Port {port}: {data[metric]:,.2f}")
                
                # Check if values are consistent
                values_list = [data[metric] for data in values.values()]
                is_consistent = all(abs(v - values_list[0]) < 100 for v in values_list)
                print(f"    Consistency: {'✅ CONSISTENT' if is_consistent else '❌ INCONSISTENT'}")
        
        # Identify specific issues
        issues = []
        for port, result in self.results.items():
            if result.get('status') == 'running':
                validation = result.get('validation', {})
                data = result.get('data', {})
                
                if not validation.get('reasonable_values', False):
                    if data.get('total_value', 0) > 500000000:  # > $500M
                        issues.append(f"Port {port}: Unrealistic portfolio value (${data.get('total_value', 0):,.0f})")
                    
                    if abs(data.get('portfolio_growth', 0)) > 100:  # > 100% growth
                        issues.append(f"Port {port}: Extreme growth rate ({data.get('portfolio_growth', 0):+.1f}%)")
                
                if not validation.get('growth_calculation_correct', False):
                    issues.append(f"Port {port}: Growth calculation error")
                
                if not validation.get('position_math_correct', False):
                    issues.append(f"Port {port}: Position math mismatch")
        
        if issues:
            print(f"\n🚨 IDENTIFIED ISSUES:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print(f"\n✅ No major calculation issues found")
    
    def generate_fix_recommendations(self):
        """Generate recommendations to fix the issues"""
        print(f"\n🔧 FIX RECOMMENDATIONS:")
        print("="*60)
        
        print("1. ⚡ IMMEDIATE ACTIONS:")
        print("   • Stop all running instances except one")
        print("   • Verify configuration consistency")
        print("   • Clear any cached/stale data")
        
        print("\n2. 🔄 CONFIGURATION FIXES:")
        print("   • Standardize initial_balance across all demos")
        print("   • Use consistent position sizing")
        print("   • Validate calculation formulas")
        
        print("\n3. 📊 CALCULATION VALIDATION:")
        print("   • Add real-time validation checks")
        print("   • Implement calculation cross-checks")
        print("   • Add portfolio value limits")
        
        print("\n4. 🐛 DEBUG RECOMMENDATIONS:")
        print("   • Add detailed logging for calculations")
        print("   • Implement calculation audit trail")
        print("   • Add unit tests for portfolio math")
        
        # Specific fixes based on findings
        config = self.load_config()
        if config:
            print(f"\n5. 🎯 SPECIFIC FIXES FOR YOUR SYSTEM:")
            print(f"   • Set consistent initial_balance: ${config['trading']['initial_balance']:,.0f}")
            print(f"   • Limit position size to reasonable amounts")
            print(f"   • Add validation for portfolio growth > 50%")
            print(f"   • Implement emergency stop for calculation errors")
    
    def run_full_audit(self):
        """Run complete analytics audit"""
        print(f"🕐 Analytics Audit Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.check_running_instances()
        self.identify_problems()
        self.generate_fix_recommendations()
        
        # Save audit results
        audit_file = f"analytics_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(audit_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'audit_summary': 'Analytics calculation audit completed'
            }, f, indent=2)
        
        print(f"\n💾 Audit results saved to: {audit_file}")
        print(f"\n🏁 Analytics Audit Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function"""
    auditor = AnalyticsFix()
    auditor.run_full_audit()


if __name__ == "__main__":
    main() 