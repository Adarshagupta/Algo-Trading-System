#!/usr/bin/env python3
"""
AI Market Analysis Model (Simplified)
Advanced machine learning for cryptocurrency market prediction using scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import asyncio
import time

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical analysis indicators calculation"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }


class MarketFeatureEngineer:
    """Advanced feature engineering for market data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical analysis features"""
        features_df = df.copy()
        
        # Price-based features
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_volatility'] = df['close'].rolling(20).std()
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = TechnicalIndicators.sma(df['close'], period)
            features_df[f'ema_{period}'] = TechnicalIndicators.ema(df['close'], period)
            features_df[f'price_sma_{period}_ratio'] = df['close'] / features_df[f'sma_{period}']
        
        # Technical indicators
        features_df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        features_df['rsi_7'] = TechnicalIndicators.rsi(df['close'], 7)
        features_df['rsi_21'] = TechnicalIndicators.rsi(df['close'], 21)
        
        # MACD
        macd_data = TechnicalIndicators.macd(df['close'])
        features_df['macd'] = macd_data['macd']
        features_df['macd_signal'] = macd_data['signal']
        features_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df['close'])
        features_df['bb_upper'] = bb_data['upper']
        features_df['bb_middle'] = bb_data['middle']
        features_df['bb_lower'] = bb_data['lower']
        features_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        features_df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Stochastic
        stoch_data = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        features_df['stoch_k'] = stoch_data['k']
        features_df['stoch_d'] = stoch_data['d']
        
        # Volume indicators
        features_df['volume_sma'] = df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
        
        # Price patterns
        features_df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
        features_df['hammer'] = ((df['close'] > df['open']) & 
                                (df['close'] - df['open'] > 2 * (df['open'] - df['low']))).astype(int)
        
        # Time-based features
        if 'timestamp' in df.columns:
            dt_index = pd.to_datetime(df['timestamp'])
            features_df['hour'] = dt_index.dt.hour
            features_df['day_of_week'] = dt_index.dt.dayofweek
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            features_df[f'rsi_lag_{lag}'] = features_df['rsi_14'].shift(lag)
        
        return features_df


class AIMarketModel:
    """Advanced AI model for market analysis and prediction"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.feature_engineer = MarketFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.model_path = f"models/{symbol}"
        
        # Create models directory
        os.makedirs(self.model_path, exist_ok=True)
        
        # Model performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {
            'rf_accuracy': 0.0,
            'gb_accuracy': 0.0,
            'ensemble_accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
        
        print(f"🤖 AI Market Model initialized for {symbol}")
    
    def build_ensemble_models(self) -> Dict:
        """Build ensemble of ML models"""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Logistic Regression
        models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Support Vector Machine
        models['svm'] = SVC(
            probability=True,
            random_state=42
        )
        
        # Voting Classifier (ensemble)
        models['ensemble'] = VotingClassifier(
            estimators=[
                ('rf', models['random_forest']),
                ('gb', models['gradient_boost']),
                ('lr', models['logistic'])
            ],
            voting='soft'
        )
        
        return models
    
    async def train_models(self, df: pd.DataFrame) -> Dict:
        """Train all AI models on historical data"""
        print(f"🔬 Training AI models for {self.symbol}...")
        
        # Feature engineering
        features_df = self.feature_engineer.create_technical_features(df)
        
        # Prepare features
        feature_columns = [col for col in features_df.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        X_features = features_df[feature_columns].fillna(method='forward').fillna(0)
        
        # Create target variable (next period direction: 1 for up, 0 for down)
        future_returns = features_df['close'].shift(-1) / features_df['close'] - 1
        y_direction = (future_returns > 0).astype(int).shift(1).fillna(0)  # 1 for up, 0 for down
        
        # Remove rows with NaN target
        valid_indices = ~y_direction.isna()
        X_features = X_features[valid_indices]
        y_direction = y_direction[valid_indices]
        
        training_results = {}
        
        if len(X_features) > 100:  # Minimum data requirement
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_direction, test_size=0.2, random_state=42, stratify=y_direction
            )
            
            # Scale features
            feature_scaler = StandardScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_test_scaled = feature_scaler.transform(X_test)
            
            # Build and train models
            ml_models = self.build_ensemble_models()
            
            for name, model in ml_models.items():
                print(f"🔬 Training {name}...")
                
                # Train model
                if name == 'ensemble':
                    # Train ensemble on scaled data
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)
                
                training_results[f'{name}_train_accuracy'] = train_accuracy
                training_results[f'{name}_test_accuracy'] = test_accuracy
                
                # Save model
                joblib.dump(model, f"{self.model_path}/{name}_model.pkl")
                
                self.models[name] = model
                
                print(f"✅ {name} trained - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}")
            
            # Save feature scaler
            joblib.dump(feature_scaler, f"{self.model_path}/feature_scaler.pkl")
            self.scalers['features'] = feature_scaler
        
        # Save training results
        with open(f"{self.model_path}/training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"🎯 AI model training completed for {self.symbol}")
        return training_results
    
    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            # Load ensemble models
            for model_name in ['random_forest', 'gradient_boost', 'logistic', 'svm', 'ensemble']:
                model_file = f"{self.model_path}/{model_name}_model.pkl"
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} model for {self.symbol}")
            
            # Load feature scaler
            if os.path.exists(f"{self.model_path}/feature_scaler.pkl"):
                self.scalers['features'] = joblib.load(f"{self.model_path}/feature_scaler.pkl")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    async def predict_market_movement(self, df: pd.DataFrame) -> Dict:
        """Make predictions using ensemble of AI models"""
        try:
            predictions = {
                'rf_prediction': 0.5,
                'gb_prediction': 0.5,
                'lr_prediction': 0.5,
                'svm_prediction': 0.5,
                'ensemble_prediction': 0.5,
                'confidence': 0.0,
                'signal': 'HOLD',
                'strength': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            if len(df) < 50:  # Need minimum data
                return predictions
            
            # Feature engineering
            features_df = self.feature_engineer.create_technical_features(df)
            
            # Traditional ML predictions
            if 'features' in self.scalers and len(self.models) > 0:
                try:
                    # Prepare feature data
                    feature_columns = [col for col in features_df.columns 
                                     if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    latest_features = features_df[feature_columns].tail(1).fillna(method='forward').fillna(0)
                    scaled_features = self.scalers['features'].transform(latest_features)
                    
                    # Get predictions from each model
                    model_predictions = {}
                    
                    for model_name, model in self.models.items():
                        try:
                            pred_proba = model.predict_proba(scaled_features)[0]
                            # pred_proba[1] is probability of up movement
                            model_predictions[model_name] = pred_proba[1]
                            predictions[f'{model_name[:2]}_prediction'] = float(pred_proba[1])
                        except Exception as e:
                            print(f"⚠️ {model_name} prediction error: {e}")
                    
                    # Ensemble prediction (weighted average)
                    if model_predictions:
                        weights = {
                            'ensemble': 0.4,
                            'random_forest': 0.25,
                            'gradient_boost': 0.25,
                            'logistic': 0.05,
                            'svm': 0.05
                        }
                        
                        ensemble_score = sum(
                            weights.get(name, 0.1) * pred 
                            for name, pred in model_predictions.items()
                        ) / sum(weights.get(name, 0.1) for name in model_predictions.keys())
                        
                        predictions['ensemble_prediction'] = ensemble_score
                        
                        # Generate trading signal
                        if ensemble_score > 0.65:
                            predictions['signal'] = 'BUY'
                            predictions['strength'] = min(ensemble_score, 1.0)
                            predictions['confidence'] = (ensemble_score - 0.5) * 2
                        elif ensemble_score < 0.35:
                            predictions['signal'] = 'SELL'
                            predictions['strength'] = min(1.0 - ensemble_score, 1.0)
                            predictions['confidence'] = (0.5 - ensemble_score) * 2
                        else:
                            predictions['signal'] = 'HOLD'
                            predictions['strength'] = 0.5
                            predictions['confidence'] = 1.0 - abs(ensemble_score - 0.5) * 2
                        
                except Exception as e:
                    print(f"⚠️ ML prediction error: {e}")
            
            # Update accuracy tracking
            self.prediction_history.append(predictions)
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return predictions
    
    def get_model_performance(self) -> Dict:
        """Get current model performance metrics"""
        try:
            # Load training results if available
            results_file = f"{self.model_path}/training_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    training_results = json.load(f)
            else:
                training_results = {}
            
            # Calculate live accuracy from recent predictions
            recent_predictions = self.prediction_history[-50:] if len(self.prediction_history) > 50 else self.prediction_history
            
            performance = {
                'symbol': self.symbol,
                'models_loaded': list(self.models.keys()),
                'training_results': training_results,
                'recent_predictions': len(recent_predictions),
                'accuracy_metrics': self.accuracy_metrics,
                'last_update': datetime.now().isoformat()
            }
            
            return performance
            
        except Exception as e:
            print(f"❌ Performance calculation error: {e}")
            return {'error': str(e)}


class AIMarketAnalyzer:
    """Main AI market analysis system"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.models = {}
        self.predictions = {}
        self.analysis_results = {}
        
        # Initialize models for each symbol
        for symbol in symbols:
            self.models[symbol] = AIMarketModel(symbol)
        
        print(f"🤖 AI Market Analyzer initialized for {len(symbols)} symbols")
    
    async def initialize_models(self, binance_client) -> Dict:
        """Initialize and train models for all symbols"""
        print("🔬 Initializing AI models...")
        
        initialization_results = {}
        
        for symbol in self.symbols[:5]:  # Limit to first 5 symbols to avoid overwhelming
            try:
                print(f"📊 Processing {symbol}...")
                
                # Try to load existing models first
                if self.models[symbol].load_models():
                    print(f"✅ Loaded existing models for {symbol}")
                    initialization_results[symbol] = {'status': 'loaded_existing'}
                    continue
                
                # Get historical data for training
                klines = binance_client.get_historical_klines(
                    symbol=symbol,
                    interval="1h",  # 1-hour intervals
                    limit=500  # Last 500 hours (~20 days)
                )
                
                if not klines:
                    print(f"⚠️ No historical data for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert to proper types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Train models
                training_results = await self.models[symbol].train_models(df)
                initialization_results[symbol] = {
                    'status': 'trained_new',
                    'training_results': training_results
                }
                
                print(f"✅ AI models trained for {symbol}")
                
            except Exception as e:
                print(f"❌ Error initializing {symbol}: {e}")
                initialization_results[symbol] = {'status': 'error', 'error': str(e)}
        
        print("🎯 AI model initialization completed")
        return initialization_results
    
    async def analyze_market(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Perform AI analysis on market data"""
        try:
            if symbol not in self.models:
                return {'error': f'No model for {symbol}'}
            
            # Get AI predictions
            predictions = await self.models[symbol].predict_market_movement(df)
            
            # Store results
            self.predictions[symbol] = predictions
            
            # Enhanced analysis
            analysis = {
                'symbol': symbol,
                'ai_predictions': predictions,
                'model_performance': self.models[symbol].get_model_performance(),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'data_points': len(df),
                    'latest_price': float(df['close'].iloc[-1]) if not df.empty else 0,
                    'price_change_24h': float(df['close'].pct_change().iloc[-1]) if len(df) > 1 else 0
                }
            }
            
            self.analysis_results[symbol] = analysis
            return analysis
            
        except Exception as e:
            error_analysis = {
                'symbol': symbol,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
            self.analysis_results[symbol] = error_analysis
            return error_analysis
    
    def get_ai_signals(self) -> List[Dict]:
        """Get trading signals from AI models"""
        signals = []
        
        for symbol, prediction in self.predictions.items():
            if prediction.get('signal') in ['BUY', 'SELL']:
                signal = {
                    'symbol': symbol,
                    'signal_type': prediction['signal'],
                    'strength': prediction['strength'],
                    'confidence': prediction['confidence'],
                    'ai_source': 'ml_ensemble',
                    'metadata': {
                        'rf_prediction': prediction.get('rf_prediction', 0),
                        'gb_prediction': prediction.get('gb_prediction', 0),
                        'lr_prediction': prediction.get('lr_prediction', 0),
                        'ensemble_prediction': prediction.get('ensemble_prediction', 0),
                        'timestamp': prediction.get('timestamp')
                    },
                    'price': 0  # Will be set by the trading system
                }
                signals.append(signal)
        
        return signals
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of AI analysis results"""
        summary = {
            'total_symbols': len(self.symbols),
            'models_active': len([s for s in self.symbols if s in self.models and len(self.models[s].models) > 0]),
            'recent_predictions': len(self.predictions),
            'signals_generated': len(self.get_ai_signals()),
            'last_analysis': max([r.get('analysis_timestamp', '') for r in self.analysis_results.values()]) if self.analysis_results else '',
            'model_types_active': [],
            'performance_summary': {}
        }
        
        # Collect model types and performance
        for symbol in self.symbols:
            if symbol in self.models:
                model_performance = self.models[symbol].get_model_performance()
                summary['model_types_active'].extend(model_performance.get('models_loaded', []))
                summary['performance_summary'][symbol] = model_performance.get('training_results', {})
        
        summary['model_types_active'] = list(set(summary['model_types_active']))
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    async def test_ai_model():
        # Test the AI model with sample data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        analyzer = AIMarketAnalyzer(symbols)
        
        print("🧪 Testing AI Market Model...")
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.rand(500) * 100 + 45000,
            'high': np.random.rand(500) * 100 + 45050,
            'low': np.random.rand(500) * 100 + 44950,
            'close': np.random.rand(500) * 100 + 45000,
            'volume': np.random.rand(500) * 1000000
        })
        
        # Test model training
        btc_model = AIMarketModel('BTCUSDT')
        training_results = await btc_model.train_models(sample_data)
        print(f"Training results: {training_results}")
        
        # Test prediction
        predictions = await btc_model.predict_market_movement(sample_data.tail(100))
        print(f"Predictions: {predictions}")
        
        print("✅ AI Model test completed")
    
    # Run test
    asyncio.run(test_ai_model()) 