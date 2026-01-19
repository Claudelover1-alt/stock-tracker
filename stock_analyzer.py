"""
Multi-Stock Real-Time Analysis Engine
Combines technical indicators, statistical modeling, and machine learning
to predict probability of reaching custom target prices

Updates second-by-second for real-time probability tracking
Supports multiple stocks simultaneously
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
        
    def fetch_real_time_data(self):
        """Fetch real-time and historical data"""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get current price
            current_data = stock.history(period='1d', interval='1m')
            if not current_data.empty:
                self.current_price = current_data['Close'].iloc[-1]
            
            # Get historical data for analysis
            self.historical_data = stock.history(period='6mo', interval='1d')
            
            # Get intraday data for short-term analysis
            self.intraday_data = stock.history(period='5d', interval='1m')
            
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate comprehensive technical indicators"""
        df = self.historical_data.copy()
        indicators = {}
        
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
        
        # Store latest values
        latest = df.iloc[-1]
        indicators['current_price'] = self.current_price
        indicators['sma_20'] = latest['SMA_20']
        indicators['sma_50'] = latest['SMA_50']
        indicators['sma_200'] = latest['SMA_200']
        indicators['macd'] = latest['MACD']
        indicators['macd_signal'] = latest['MACD_Signal']
        indicators['macd_histogram'] = latest['MACD_Hist']
        indicators['rsi'] = latest['RSI']
        indicators['bb_position'] = (self.current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        indicators['bb_width'] = latest['BB_Width']
        indicators['stoch_k'] = latest['Stoch_K']
        indicators['stoch_d'] = latest['Stoch_D']
        indicators['atr'] = latest['ATR']
        indicators['adx'] = latest['ADX']
        indicators['cci'] = latest['CCI']
        indicators['mfi'] = latest['MFI']
        indicators['williams_r'] = latest['Williams_R']
        indicators['roc'] = latest['ROC']
        indicators['volume_ratio'] = latest['Volume_Ratio']
        
        # Trend signals
         indicators['trend_signals'] = {
             'price_above_sma20': bool(self.current_price > latest['SMA_20']),
             'price_above_sma50': bool(self.current_price > latest['SMA_50']),
             'price_above_sma200': bool(self.current_price > latest['SMA_200']),
             'golden_cross': bool(latest['SMA_50'] > latest['SMA_200']),
             'macd_bullish': bool(latest['MACD'] > latest['MACD_Signal']),
             'rsi_neutral': bool(30 < latest['RSI'] < 70),
             'strong_trend': bool(latest['ADX'] > 25)
        }
        
        self.technical_df = df
        return indicators
    
    def calculate_momentum_score(self, indicators):
        """Calculate momentum score from technical indicators"""
        score = 0
        max_score = 0
        
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
        
        return (score / max_score) * 100
    
    def statistical_analysis(self):
        """Perform statistical analysis and volatility calculations"""
        stats_results = {}
        
        # Historical volatility
        returns = self.historical_data['Close'].pct_change().dropna()
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
        stats_results['return_1d'] = returns.iloc[-1] * 100
        stats_results['return_5d'] = ((self.current_price / self.historical_data['Close'].iloc[-6]) - 1) * 100
        stats_results['return_20d'] = ((self.current_price / self.historical_data['Close'].iloc[-21]) - 1) * 100
        stats_results['return_60d'] = ((self.current_price / self.historical_data['Close'].iloc[-61]) - 1) * 100
        
        return stats_results
    
    def machine_learning_prediction(self):
        """Use machine learning models to predict target probability"""
        try:
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
    
    def generate_analysis_report(self):
        """Generate complete analysis report"""
        print(f"Analyzing {self.ticker}...")
        
        if not self.fetch_real_time_data():
            return None
        
        print("Calculating technical indicators...")
        technical_indicators = self.calculate_technical_indicators()
        
        print("Performing statistical analysis...")
        stats_results = self.statistical_analysis()
        
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
    analyzer = StockAnalyzer(ticker="AAPL", target_price=250.0)
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
