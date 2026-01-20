Stock analyzer fixed · PY
Copy

"""
Multi-Stock Real-Time Analysis Engine
Combines technical indicators, statistical modeling, and machine learning
to predict probability of reaching custom target prices
Updates second-by-second for real-time probability tracking
Supports multiple stocks simultaneously

FIXED VERSION: Added rate limiting, error handling, and retry logic
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, ticker, target_price):
        self.ticker = ticker.upper()
        self.target_price = target_price
        self.current_price = None
        self.historical_data = None
        self.scaler = StandardScaler()
        
    def fetch_real_time_data(self, max_retries=3, base_delay=5):
        """Fetch real-time and historical data with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                # Add delay to respect rate limits
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retry {attempt}/{max_retries} - waiting {delay}s...")
                    time.sleep(delay)
                else:
                    time.sleep(2)  # Always wait 2 seconds before first request
                
                stock = yf.Ticker(self.ticker)
                
                # Get current price with error checking
                print(f"Fetching current data for {self.ticker}...")
                current_data = stock.history(period='1d', interval='1m')
                
                if current_data is None or current_data.empty:
                    print(f"Warning: No current data returned for {self.ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                self.current_price = current_data['Close'].iloc[-1]
                print(f"Current price: ${self.current_price:.2f}")
                
                # Wait between requests
                time.sleep(3)
                
                # Get historical data for analysis
                print(f"Fetching historical data for {self.ticker}...")
                self.historical_data = stock.history(period='6mo', interval='1d')
                
                if self.historical_data is None or self.historical_data.empty:
                    print(f"Warning: No historical data returned for {self.ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                print(f"Retrieved {len(self.historical_data)} days of historical data")
                
                # Wait between requests
                time.sleep(3)
                
                # Get intraday data for short-term analysis
                print(f"Fetching intraday data for {self.ticker}...")
                self.intraday_data = stock.history(period='5d', interval='1m')
                
                if self.intraday_data is None or self.intraday_data.empty:
                    print(f"Warning: No intraday data available, using historical only")
                    self.intraday_data = self.historical_data.tail(50)  # Fallback
                
                print("Data fetch successful!")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error fetching data (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check if it's a rate limit error
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** (attempt + 1))
                        print(f"Rate limited! Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Rate limit exceeded after {max_retries} attempts")
                        return False
                
                # For other errors, retry with shorter delay
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                else:
                    print(f"Failed after {max_retries} attempts")
                    return False
        
        return False
    
    def calculate_technical_indicators(self):
        """Calculate comprehensive technical indicators"""
        if self.historical_data is None or self.historical_data.empty:
            print("Error: No historical data available for technical analysis")
            return None
        
        df = self.historical_data.copy()
        indicators = {}
        
        try:
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Stochastic Oscillator
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['ATR'] = ranges.max(axis=1).rolling(14).mean()
            
            # OBV (On-Balance Volume)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # ADX (Average Directional Index)
            df['TR'] = ranges.max(axis=1)
            df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']), 
                                  np.maximum(df['High'] - df['High'].shift(), 0), 0)
            df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()), 
                                  np.maximum(df['Low'].shift() - df['Low'], 0), 0)
            df['+DI'] = 100 * (df['+DM'].rolling(14).mean() / df['ATR'])
            df['-DI'] = 100 * (df['-DM'].rolling(14).mean() / df['ATR'])
            df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
            df['ADX'] = df['DX'].rolling(14).mean()
            
            # CCI (Commodity Channel Index)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            
            # Money Flow Index
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']
            pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
            df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))
            
            # Williams %R
            df['Williams_R'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
            
            # Momentum indicators
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Store latest values with null checks
            latest = df.iloc[-1]
            indicators['current_price'] = self.current_price
            indicators['sma_20'] = latest['SMA_20'] if pd.notna(latest['SMA_20']) else self.current_price
            indicators['sma_50'] = latest['SMA_50'] if pd.notna(latest['SMA_50']) else self.current_price
            indicators['sma_200'] = latest['SMA_200'] if pd.notna(latest['SMA_200']) else self.current_price
            indicators['macd'] = latest['MACD'] if pd.notna(latest['MACD']) else 0
            indicators['macd_signal'] = latest['MACD_Signal'] if pd.notna(latest['MACD_Signal']) else 0
            indicators['macd_histogram'] = latest['MACD_Hist'] if pd.notna(latest['MACD_Hist']) else 0
            indicators['rsi'] = latest['RSI'] if pd.notna(latest['RSI']) else 50
            indicators['bb_position'] = (self.current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) if pd.notna(latest['BB_Lower']) and pd.notna(latest['BB_Upper']) else 0.5
            indicators['bb_width'] = latest['BB_Width'] if pd.notna(latest['BB_Width']) else 0
            indicators['stoch_k'] = latest['Stoch_K'] if pd.notna(latest['Stoch_K']) else 50
            indicators['stoch_d'] = latest['Stoch_D'] if pd.notna(latest['Stoch_D']) else 50
            indicators['atr'] = latest['ATR'] if pd.notna(latest['ATR']) else 0
            indicators['adx'] = latest['ADX'] if pd.notna(latest['ADX']) else 0
            indicators['cci'] = latest['CCI'] if pd.notna(latest['CCI']) else 0
            indicators['mfi'] = latest['MFI'] if pd.notna(latest['MFI']) else 50
            indicators['williams_r'] = latest['Williams_R'] if pd.notna(latest['Williams_R']) else -50
            indicators['roc'] = latest['ROC'] if pd.notna(latest['ROC']) else 0
            indicators['volume_ratio'] = latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 1
            
            # Trend signals
            indicators['trend_signals'] = {
                'price_above_sma20': bool(self.current_price > indicators['sma_20']),
                'price_above_sma50': bool(self.current_price > indicators['sma_50']),
                'price_above_sma200': bool(self.current_price > indicators['sma_200']),
                'golden_cross': bool(indicators['sma_50'] > indicators['sma_200']),
                'macd_bullish': bool(indicators['macd'] > indicators['macd_signal']),
                'rsi_neutral': bool(30 < indicators['rsi'] < 70),
                'strong_trend': bool(indicators['adx'] > 25)
            }
            
            self.technical_df = df
            return indicators
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None
    
    def calculate_momentum_score(self, indicators):
        """Calculate momentum score from technical indicators"""
        if indicators is None:
            return 50.0  # Neutral score if no indicators
        
        score = 0
        max_score = 0
        
        try:
            # RSI scoring (0-100 scale)
            rsi = indicators['rsi']
            if 40 < rsi < 60:
                score += 5
            elif 60 <= rsi < 70:
                score += 8
            elif 70 <= rsi < 80:
                score += 6
            elif rsi >= 80:
                score += 3
            max_score += 10
            
            # MACD scoring
            if indicators['macd'] > indicators['macd_signal']:
                score += 10
            if indicators['macd_histogram'] > 0:
                score += 5
            max_score += 15
            
            # Moving average alignment
            if indicators['trend_signals']['price_above_sma20']:
                score += 8
            if indicators['trend_signals']['price_above_sma50']:
                score += 7
            if indicators['trend_signals']['price_above_sma200']:
                score += 5
            if indicators['trend_signals']['golden_cross']:
                score += 5
            max_score += 25
            
            # ADX (trend strength)
            adx = indicators['adx']
            if adx > 25:
                score += min(10, adx / 5)
            max_score += 10
            
            # Stochastic
            if indicators['stoch_k'] > indicators['stoch_d'] and indicators['stoch_k'] < 80:
                score += 5
            max_score += 5
            
            # Volume
            if indicators['volume_ratio'] > 1.2:
                score += 10
            elif indicators['volume_ratio'] > 1:
                score += 5
            max_score += 10
            
            # ROC
            if indicators['roc'] > 0:
                score += min(10, indicators['roc'])
            max_score += 10
            
            # MFI
            mfi = indicators['mfi']
            if 40 < mfi < 80:
                score += 5
            max_score += 5
            
            # Williams %R
            wr = indicators['williams_r']
            if -50 < wr < -20:
                score += 5
            max_score += 5
            
            return (score / max_score) * 100 if max_score > 0 else 50.0
            
        except Exception as e:
            print(f"Error calculating momentum score: {e}")
            return 50.0
    
    def statistical_analysis(self):
        """Perform statistical analysis and volatility calculations"""
        if self.historical_data is None or self.historical_data.empty:
            print("Error: No historical data for statistical analysis")
            return None
        
        stats_results = {}
        
        try:
            # Historical volatility
            returns = self.historical_data['Close'].pct_change().dropna()
            
            if len(returns) == 0:
                print("Warning: Insufficient return data")
                return None
            
            stats_results['daily_volatility'] = returns.std()
            stats_results['annual_volatility'] = returns.std() * np.sqrt(252)
            
            # Calculate days to target assuming current trend
            recent_returns = returns.tail(30).mean()
            distance_to_target = self.target_price - self.current_price
            
            if recent_returns > 0:
                days_to_target = distance_to_target / (self.current_price * recent_returns)
                stats_results['estimated_days_to_target'] = max(0, days_to_target)
            else:
                stats_results['estimated_days_to_target'] = None
            
            # Monte Carlo simulation
            num_simulations = 10000
            days_forward = 250  # Rest of 2026
            
            simulations = np.zeros((num_simulations, days_forward))
            simulations[:, 0] = self.current_price
            
            for i in range(1, days_forward):
                random_returns = np.random.normal(recent_returns, stats_results['daily_volatility'], num_simulations)
                simulations[:, i] = simulations[:, i-1] * (1 + random_returns)
            
            # Calculate probability of hitting target
            hit_target = np.any(simulations >= self.target_price, axis=1)
            stats_results['monte_carlo_probability'] = np.mean(hit_target) * 100
            
            # Percentile analysis
            final_prices = simulations[:, -1]
            stats_results['expected_price_median'] = np.median(final_prices)
            stats_results['expected_price_mean'] = np.mean(final_prices)
            stats_results['price_10th_percentile'] = np.percentile(final_prices, 10)
            stats_results['price_90th_percentile'] = np.percentile(final_prices, 90)
            
            # Sharpe ratio (assuming 4% risk-free rate)
            excess_returns = returns - 0.04/252
            stats_results['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            stats_results['sortino_ratio'] = np.sqrt(252) * excess_returns.mean() / downside_std if len(downside_returns) > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            stats_results['max_drawdown'] = drawdown.min() * 100
            
            # Recent trend analysis
            stats_results['return_1d'] = returns.iloc[-1] * 100 if len(returns) > 0 else 0
            stats_results['return_5d'] = ((self.current_price / self.historical_data['Close'].iloc[-6]) - 1) * 100 if len(self.historical_data) > 5 else 0
            stats_results['return_20d'] = ((self.current_price / self.historical_data['Close'].iloc[-21]) - 1) * 100 if len(self.historical_data) > 20 else 0
            stats_results['return_60d'] = ((self.current_price / self.historical_data['Close'].iloc[-61]) - 1) * 100 if len(self.historical_data) > 60 else 0
            
            return stats_results
            
        except Exception as e:
            print(f"Error in statistical analysis: {e}")
            return None
    
    def machine_learning_prediction(self):
        """Use machine learning models to predict target probability"""
        try:
            if self.technical_df is None:
                return {'ml_probability': 50.0, 'confidence': 'low', 'method': 'no_data'}
            
            # Prepare features
            df = self.technical_df.copy()
            
            # Create features
            feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                            'Stoch_K', 'Stoch_D', 'ATR', 'ADX', 'CCI', 
                            'MFI', 'Williams_R', 'ROC', 'Volume_Ratio', 'BB_Width']
            
            df_clean = df[feature_cols].dropna()
            
            # Create target: did price increase in next 20 days?
            future_returns = df['Close'].shift(-20) / df['Close'] - 1
            df_clean['target'] = future_returns
            df_clean = df_clean.dropna()
            
            if len(df_clean) < 50:
                return {'ml_probability': 50.0, 'confidence': 'low', 'method': 'insufficient_data'}
            
            # Split data
            X = df_clean[feature_cols]
            y = df_clean['target']
            
            # Use last 80% for training
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Predict current situation
            current_features = df[feature_cols].iloc[-1:].values
            current_features_scaled = self.scaler.transform(current_features)
            
            rf_pred = rf_model.predict(current_features_scaled)[0]
            gb_pred = gb_model.predict(current_features_scaled)[0]
            
            # Average predictions
            avg_predicted_return = (rf_pred + gb_pred) / 2
            
            # Calculate required return to hit target
            required_return = (self.target_price / self.current_price) - 1
            
            # Calculate probability based on historical distribution of predictions
            y_pred_rf = rf_model.predict(X_test_scaled)
            y_pred_gb = gb_model.predict(X_test_scaled)
            avg_preds = (y_pred_rf + y_pred_gb) / 2
            
            # Calculate standard deviation of prediction errors
            pred_std = np.std(avg_preds - y_test)
            
            # Use normal distribution to estimate probability
            if pred_std > 0:
                z_score = (required_return - avg_predicted_return) / pred_std
                ml_probability = (1 - stats.norm.cdf(z_score)) * 100
            else:
                ml_probability = 50.0
            
            # Clip probability to reasonable range
            ml_probability = np.clip(ml_probability, 0, 100)
            
            # Feature importance
            rf_importance = dict(zip(feature_cols, rf_model.feature_importances_))
            top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'ml_probability': ml_probability,
                'predicted_return': avg_predicted_return * 100,
                'required_return': required_return * 100,
                'confidence': 'high' if len(df_clean) > 100 else 'medium',
                'top_features': top_features,
                'model_std': pred_std * 100
            }
        
        except Exception as e:
            print(f"ML error: {e}")
            return {'ml_probability': 50.0, 'confidence': 'low', 'method': 'error'}
    
    def calculate_composite_probability(self, technical_indicators, stats_results, ml_results):
        """Combine all analyses into final probability"""
        
        # Handle None cases
        if technical_indicators is None or stats_results is None or ml_results is None:
            print("Warning: Missing analysis data, using defaults")
            return {
                'composite_probability': 50.0,
                'momentum_score': 50.0,
                'statistical_probability': 50.0,
                'ml_probability': 50.0,
                'distance_factor': 50.0,
                'time_factor': 50.0,
                'confidence_level': 'LOW',
                'component_weights': {}
            }
        
        try:
            # Momentum score (0-100)
            momentum_score = self.calculate_momentum_score(technical_indicators)
            
            # Statistical probability from Monte Carlo
            mc_probability = stats_results['monte_carlo_probability']
            
            # ML probability
            ml_probability = ml_results['ml_probability']
            
            # Distance to target factor
            distance_pct = ((self.target_price - self.current_price) / self.current_price) * 100
            distance_factor = max(0, 100 - distance_pct * 2)  # Penalty for being far from target
            
            # Time factor (assuming rest of 2026)
            days_remaining = (datetime(2026, 12, 31) - datetime.now()).days
            time_factor = min(100, (days_remaining / 365) * 100)
            
            # Weighted composite
            weights = {
                'momentum': 0.25,
                'monte_carlo': 0.30,
                'ml': 0.30,
                'distance': 0.10,
                'time': 0.05
            }
            
            composite = (
                momentum_score * weights['momentum'] +
                mc_probability * weights['monte_carlo'] +
                ml_probability * weights['ml'] +
                distance_factor * weights['distance'] +
                time_factor * weights['time']
            )
            
            # Confidence level
            if ml_results['confidence'] == 'high' and len(self.historical_data) > 100:
                confidence = 'HIGH'
            elif ml_results['confidence'] == 'medium':
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'composite_probability': composite,
                'momentum_score': momentum_score,
                'statistical_probability': mc_probability,
                'ml_probability': ml_probability,
                'distance_factor': distance_factor,
                'time_factor': time_factor,
                'confidence_level': confidence,
                'component_weights': weights
            }
            
        except Exception as e:
            print(f"Error calculating composite probability: {e}")
            return {
                'composite_probability': 50.0,
                'momentum_score': 50.0,
                'statistical_probability': 50.0,
                'ml_probability': 50.0,
                'distance_factor': 50.0,
                'time_factor': 50.0,
                'confidence_level': 'LOW',
                'component_weights': {}
            }
    
    def generate_analysis_report(self):
        """Generate complete analysis report"""
        print(f"Analyzing {self.ticker}...")
        
        if not self.fetch_real_time_data():
            print(f"Error updating {self.ticker}: Unable to fetch data")
            return None
        
        print("Calculating technical indicators...")
        technical_indicators = self.calculate_technical_indicators()
        
        if technical_indicators is None:
            print(f"Error updating {self.ticker}: Failed to calculate technical indicators")
            return None
        
        print("Performing statistical analysis...")
        stats_results = self.statistical_analysis()
        
        if stats_results is None:
            print(f"Error updating {self.ticker}: Failed statistical analysis")
            return None
        
        print("Running machine learning models...")
        ml_results = self.machine_learning_prediction()
        
        print("Computing composite probability...")
        probability_results = self.calculate_composite_probability(
            technical_indicators, stats_results, ml_results
        )
        
        # Compile full report
        report = {
            'timestamp': datetime.now().isoformat(),
            'ticker': self.ticker,
            'current_price': round(self.current_price, 2),
            'target_price': self.target_price,
            'distance_to_target': round(self.target_price - self.current_price, 2),
            'distance_pct': round(((self.target_price - self.current_price) / self.current_price) * 100, 2),
            
            'probability': probability_results,
            
            'technical_indicators': {
                'rsi': round(technical_indicators['rsi'], 2),
                'macd': round(technical_indicators['macd'], 4),
                'macd_signal': round(technical_indicators['macd_signal'], 4),
                'stoch_k': round(technical_indicators['stoch_k'], 2),
                'adx': round(technical_indicators['adx'], 2),
                'cci': round(technical_indicators['cci'], 2),
                'mfi': round(technical_indicators['mfi'], 2),
                'williams_r': round(technical_indicators['williams_r'], 2),
                'volume_ratio': round(technical_indicators['volume_ratio'], 2),
                'trend_signals': technical_indicators['trend_signals']
            },
            
            'statistics': {
                'annual_volatility': round(stats_results['annual_volatility'] * 100, 2),
                'sharpe_ratio': round(stats_results['sharpe_ratio'], 2),
                'sortino_ratio': round(stats_results['sortino_ratio'], 2),
                'max_drawdown': round(stats_results['max_drawdown'], 2),
                'return_1d': round(stats_results['return_1d'], 2),
                'return_5d': round(stats_results['return_5d'], 2),
                'return_20d': round(stats_results['return_20d'], 2),
                'expected_price_median': round(stats_results['expected_price_median'], 2),
                'price_range_90': [
                    round(stats_results['price_10th_percentile'], 2),
                    round(stats_results['price_90th_percentile'], 2)
                ]
            },
            
            'machine_learning': {
                'probability': round(ml_results['ml_probability'], 2),
                'predicted_return': round(ml_results.get('predicted_return', 0), 2),
                'confidence': ml_results['confidence']
            }
        }
        
        return report

def save_report(report, filename='analysis_output.json'):
    """Save report to JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {filename}")

if __name__ == "__main__":
    # Example usage
    analyzer = StockAnalyzer(ticker="TSLA", target_price=500.0)
    report = analyzer.generate_analysis_report()
    
    if report:
        save_report(report)
        
        print("\n" + "="*60)
        print(f"{report['ticker']} STOCK ANALYSIS - TARGET: ${report['target_price']}")
        print("="*60)
        print(f"Current Price: ${report['current_price']}")
        print(f"Target Price: ${report['target_price']}")
        print(f"Distance: ${report['distance_to_target']} ({report['distance_pct']}%)")
        print()
        print(f"PROBABILITY OF HITTING ${report['target_price']}: {report['probability']['composite_probability']:.1f}%")
        print(f"Confidence Level: {report['probability']['confidence_level']}")
        print()
        print("Component Probabilities:")
        print(f"  - Momentum Score: {report['probability']['momentum_score']:.1f}%")
        print(f"  - Statistical (Monte Carlo): {report['probability']['statistical_probability']:.1f}%")
        print(f"  - Machine Learning: {report['probability']['ml_probability']:.1f}%")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ANALYSIS FAILED")
        print("="*60)
        print("Unable to generate report due to data fetch errors.")
        print("This is likely due to:")
        print("  1. Rate limiting from Yahoo Finance")
        print("  2. Network connectivity issues")
        print("  3. Invalid ticker symbol")
        print("\nPlease wait a few minutes and try again.")
        print("="*60)
