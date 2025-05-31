import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import find_peaks
import joblib
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class CrashDetectionConfig:
    """Configuration class for crash detection parameters"""
    short_window: int = 10
    long_window: int = 50
    volatility_window: int = 20
    return_threshold: float = -0.005
    volatility_threshold: float = 0.02
    drawdown_threshold: float = -0.1
    outlier_contamination: float = 0.1

class MarketCrashDetector:
    """Advanced Market Crash Detection System with ML capabilities"""

    def __init__(self, config: CrashDetectionConfig = None):
        self.config = config or CrashDetectionConfig()
        self.scaler = RobustScaler()
        self.isolation_forest = None
        self.random_forest = None
        self.is_fitted = False
        self.feature_importance = None

    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate market data with comprehensive error handling"""
        try:
            # Try different common separators and encodings
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                # If all separators fail, try different encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                else:
                    raise ValueError(
                        "Unable to read the CSV file with any common format"
                    )

            # Validate required columns
            required_columns = ['Date', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                # Try to find columns with similar names
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'value'])]

                if date_cols and price_cols:
                    df = df.rename(columns={date_cols[0]: 'Date', price_cols[0]: 'Close'})
                    print(f"Renamed columns: {date_cols[0]} -> Date, {price_cols[0]} -> Close")
                else:
                    raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate data types and handle conversion
            df = self._clean_and_convert_data(df)

            return df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Return sample data for demonstration
            return self._generate_sample_data()

    def _clean_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types with error handling"""
        # Handle Date column
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                    break
                except:
                    continue

        # Handle Close price column
        if df['Close'].dtype == 'object':
            # Remove currency symbols and commas
            df['Close'] = df['Close'].astype(str).str.replace(r'[₹$,]', '', regex=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        # Remove rows with invalid data
        df = df.dropna(subset=['Date', 'Close'])

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)

        return df

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data for demonstration"""
        print("Generating sample data for demonstration...")
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)

        # Generate realistic market data with trends and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some crash periods
        crash_periods = [500, 800, 1200]
        for period in crash_periods:
            if period < len(returns):
                returns[period:period+10] = np.random.normal(-0.05, 0.05, 10)

        prices = [1000]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'Date': dates[:len(prices)],
            'Close': prices
        })

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()

        # Basic price metrics
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        df[f'SMA_{self.config.short_window}'] = df['Close'].rolling(self.config.short_window).mean()
        df[f'SMA_{self.config.long_window}'] = df['Close'].rolling(self.config.long_window).mean()
        df[f'EMA_{self.config.short_window}'] = df['Close'].ewm(span=self.config.short_window).mean()

        # Volatility metrics
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(self.config.volatility_window).std()
        df['GARCH_Volatility'] = df['Daily_Return'].rolling(self.config.volatility_window).apply(
            lambda x: np.sqrt(np.var(x) * len(x) / (len(x) - 1)) if len(x) > 1 else 0
        )

        # Drawdown analysis
        df['Cumulative_Max'] = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max']
        df['Underwater'] = np.where(df['Drawdown'] < 0, 1, 0)

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(bb_window).mean()
        bb_rolling_std = df['Close'].rolling(bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_rolling_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_rolling_std * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Volume-based indicators (using synthetic volume if not available)
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.lognormal(10, 1, len(df))

        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Market stress indicators
        df['VIX_Proxy'] = df['Rolling_Volatility'] * 100  # Volatility as VIX proxy
        df['Stress_Index'] = (df['Rolling_Volatility'] * abs(df['Daily_Return'])) / df['Close']

        return df

    def detect_regime_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime changes using multiple methods"""
        df = df.copy()

        # Volatility regime detection
        vol_quantiles = df['Rolling_Volatility'].quantile([0.33, 0.67])
        df['Volatility_Regime'] = pd.cut(
            df['Rolling_Volatility'], 
            bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )

        # Return regime detection
        return_quantiles = df['Daily_Return'].quantile([0.2, 0.8])
        df['Return_Regime'] = pd.cut(
            df['Daily_Return'],
            bins=[-np.inf, return_quantiles.iloc[0], return_quantiles.iloc[1], np.inf],
            labels=['Bear', 'Neutral', 'Bull']
        )

        # Trend detection
        df['Price_Trend'] = np.where(
            df['Close'] > df[f'SMA_{self.config.long_window}'], 'Uptrend', 'Downtrend'
        )

        return df

    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning models"""
        df = df.copy()

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
            df[f'Volatility_Lag_{lag}'] = df['Rolling_Volatility'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Return_Mean_{window}'] = df['Daily_Return'].rolling(window).mean()
            df[f'Return_Std_{window}'] = df['Daily_Return'].rolling(window).std()
            df[f'Return_Skew_{window}'] = df['Daily_Return'].rolling(window).skew()
            df[f'Return_Kurt_{window}'] = df['Daily_Return'].rolling(window).kurt()

        # Price momentum features
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'].pct_change(period)

        # Volatility features
        df['Volatility_Ratio'] = df['Rolling_Volatility'] / df['Rolling_Volatility'].rolling(50).mean()
        df['Volatility_Breakout'] = np.where(
            df['Rolling_Volatility'] > df['Rolling_Volatility'].rolling(20).quantile(0.8), 1, 0
        )

        return df

    def label_crash_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels for crash events using multiple criteria"""
        df = df.copy()

        # Traditional crash criteria
        df['Traditional_Crash'] = np.where(
            (df['Daily_Return'] < self.config.return_threshold) & 
            (df['Rolling_Volatility'] > self.config.volatility_threshold),
            1, 0
        )

        # Drawdown-based crash
        df['Drawdown_Crash'] = np.where(
            df['Drawdown'] < self.config.drawdown_threshold, 1, 0
        )

        # Multi-day crash (consecutive negative returns)
        df['Consecutive_Negative'] = (df['Daily_Return'] < 0).astype(int)
        df['Multi_Day_Crash'] = np.where(
            df['Consecutive_Negative'].rolling(3).sum() >= 3, 1, 0
        )

        # Extreme return events (using z-score)
        returns_clean = df['Daily_Return'].dropna()
        if len(returns_clean) > 0:
            # Calculate z-scores for non-null values
            z_scores = stats.zscore(returns_clean)
            # Create a series with the same index as the original data, filled with NaN
            df['Return_ZScore'] = np.nan
            # Fill in the z-scores for the non-null positions
            df.loc[returns_clean.index, 'Return_ZScore'] = z_scores
        else:
            df['Return_ZScore'] = np.nan
    
        df['Extreme_Return'] = np.where(abs(df['Return_ZScore']) > 2.5, 1, 0)

        # Combined crash label
        df['Crash_Label'] = np.where(
            (df['Traditional_Crash'] == 1) | 
            (df['Drawdown_Crash'] == 1) | 
            (df['Multi_Day_Crash'] == 1) |
            (df['Extreme_Return'] == 1),
            1, 0
        )

        return df

    def fit_isolation_forest(self, df: pd.DataFrame) -> Dict:
        """Fit Isolation Forest for anomaly detection"""
        feature_cols = [col for col in df.columns if col not in [
            'Date', 'Close', 'Crash_Label', 'Traditional_Crash', 'Drawdown_Crash',
            'Multi_Day_Crash', 'Extreme_Return', 'Volatility_Regime', 'Return_Regime', 'Price_Trend'
        ]]

        # Select numeric features only
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_features].fillna(method='ffill').fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.config.outlier_contamination,
            random_state=42,
            n_estimators=200
        )

        anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
        df['Anomaly_Score'] = self.isolation_forest.decision_function(X_scaled)
        df['Is_Anomaly'] = np.where(anomaly_labels == -1, 1, 0)

        return {
            'model': self.isolation_forest,
            'scaler': self.scaler,
            'features': numeric_features,
            'anomaly_rate': np.mean(df['Is_Anomaly'])
        }

    def fit_supervised_model(self, df: pd.DataFrame) -> Dict:
        """Fit supervised learning model for crash prediction"""
        feature_cols = [col for col in df.columns if col not in [
            'Date', 'Close', 'Crash_Label', 'Traditional_Crash', 'Drawdown_Crash',
            'Multi_Day_Crash', 'Extreme_Return', 'Volatility_Regime', 'Return_Regime', 
            'Price_Trend', 'Is_Anomaly', 'Anomaly_Score'
        ]]

        # Select numeric features only
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_features].fillna(method='ffill').fillna(0)
        y = df['Crash_Label'].fillna(0)

        # Handle class imbalance
        if len(np.unique(y)) < 2 or np.sum(y) < 10:
            print("Warning: Insufficient crash events for supervised learning")
            return {'model': None, 'performance': None}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit Random Forest
        self.random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )

        self.random_forest.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.random_forest.predict(X_test_scaled)
        y_pred_proba = self.random_forest.predict_proba(X_test_scaled)[:, 1]

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': numeric_features,
            'importance': self.random_forest.feature_importances_
        }).sort_values('importance', ascending=False)

        # Cross validation
        cv_scores = cross_val_score(self.random_forest, X_train_scaled, y_train, cv=5)

        performance = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        # Add predictions to dataframe
        df_with_predictions = df.copy()
        X_all_scaled = self.scaler.transform(X)
        df_with_predictions['Crash_Probability'] = self.random_forest.predict_proba(X_all_scaled)[:, 1]
        df_with_predictions['ML_Prediction'] = self.random_forest.predict(X_all_scaled)

        self.is_fitted = True

        return {
            'model': self.random_forest,
            'performance': performance,
            'feature_importance': self.feature_importance,
            'df_with_predictions': df_with_predictions
        }

    def create_comprehensive_visualization(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))

        # 1. Price and Crash Events
        plt.subplot(6, 2, 1)
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue', alpha=0.7)
        crash_dates = df[df['Crash_Label'] == 1]
        if len(crash_dates) > 0:
            plt.scatter(crash_dates['Date'], crash_dates['Close'], 
                       color='red', label='Crash Events', s=50, alpha=0.8)
        plt.title('Stock Price with Crash Events', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Returns Distribution
        plt.subplot(6, 2, 2)
        plt.hist(df['Daily_Return'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df['Daily_Return'].mean(), color='red', linestyle='--', label='Mean')
        plt.axvline(df['Daily_Return'].quantile(0.05), color='orange', linestyle='--', label='5th Percentile')
        plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Volatility Analysis
        plt.subplot(6, 2, 3)
        plt.plot(df['Date'], df['Rolling_Volatility'], label='Rolling Volatility', color='orange')
        plt.axhline(y=self.config.volatility_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.config.volatility_threshold})')
        plt.title('Rolling Volatility', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Drawdown Analysis
        plt.subplot(6, 2, 4)
        plt.fill_between(df['Date'], df['Drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        plt.axhline(y=self.config.drawdown_threshold, color='darkred', linestyle='--', 
                   label=f'Threshold ({self.config.drawdown_threshold})')
        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Technical Indicators
        plt.subplot(6, 2, 5)
        plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        plt.title('RSI Indicator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. MACD
        plt.subplot(6, 2, 6)
        plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
        plt.plot(df['Date'], df['MACD_Signal'], label='Signal', color='red')
        plt.bar(df['Date'], df['MACD_Histogram'], alpha=0.3, label='Histogram')
        plt.title('MACD Indicator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 7. Anomaly Detection Results
        if 'Is_Anomaly' in df.columns:
            plt.subplot(6, 2, 7)
            plt.plot(df['Date'], df['Anomaly_Score'], label='Anomaly Score', color='green')
            anomaly_dates = df[df['Is_Anomaly'] == 1]
            if len(anomaly_dates) > 0:
                plt.scatter(anomaly_dates['Date'], anomaly_dates['Anomaly_Score'], 
                           color='red', label='Anomalies', s=30)
            plt.title('Anomaly Detection Results', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Anomaly Score')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 8. ML Predictions
        if 'Crash_Probability' in df.columns:
            plt.subplot(6, 2, 8)
            plt.plot(df['Date'], df['Crash_Probability'], label='Crash Probability', color='red')
            plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
            plt.title('ML Crash Probability', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 9. Bollinger Bands
        plt.subplot(6, 2, 9)
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.plot(df['Date'], df['BB_Upper'], label='Upper Band', color='red', alpha=0.5)
        plt.plot(df['Date'], df['BB_Lower'], label='Lower Band', color='green', alpha=0.5)
        plt.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], alpha=0.1)
        plt.title('Bollinger Bands', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 10. Feature Importance (if available)
        if self.feature_importance is not None:
            plt.subplot(6, 2, 10)
            top_features = self.feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Importance')
            plt.grid(True, alpha=0.3)

        # 11. Correlation Heatmap
        plt.subplot(6, 2, 11)
        corr_features = ['Daily_Return', 'Rolling_Volatility', 'Drawdown', 'RSI', 'MACD']
        corr_features = [f for f in corr_features if f in df.columns]
        if len(corr_features) > 1:
            corr_matrix = df[corr_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

        # 12. Risk Metrics Summary
        plt.subplot(6, 2, 12)
        risk_metrics = {
            'Total Crashes': df['Crash_Label'].sum(),
            'Max Drawdown': df['Drawdown'].min(),
            'Avg Volatility': df['Rolling_Volatility'].mean(),
            'Sharpe Ratio': df['Daily_Return'].mean() / df['Daily_Return'].std() * np.sqrt(252),
            'VaR (95%)': df['Daily_Return'].quantile(0.05),
        }

        plt.axis('off')
        risk_text = "Risk Metrics Summary\n" + "="*25 + "\n"
        for metric, value in risk_metrics.items():
            if isinstance(value, float):
                risk_text += f"{metric}: {value:.4f}\n"
            else:
                risk_text += f"{metric}: {value}\n"

        plt.text(0.1, 0.9, risk_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=3.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'data_summary': {
                'total_observations': len(df),
                'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
                'price_range': f"{df['Close'].min():.2f} to {df['Close'].max():.2f}",
                'total_return': f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%"
            },
            'crash_analysis': {
                'total_crashes': int(df['Crash_Label'].sum()),
                'crash_rate': f"{df['Crash_Label'].mean() * 100:.2f}%",
                'max_drawdown': f"{df['Drawdown'].min() * 100:.2f}%",
                'avg_recovery_time': 'N/A'  # Would need more complex calculation
            },
            'risk_metrics': {
                'daily_volatility': f"{df['Daily_Return'].std() * 100:.2f}%",
                'annualized_volatility': f"{df['Daily_Return'].std() * np.sqrt(252) * 100:.2f}%",
                'sharpe_ratio': f"{(df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252):.2f}",
                'var_95': f"{df['Daily_Return'].quantile(0.05) * 100:.2f}%",
                'skewness': f"{df['Daily_Return'].skew():.2f}",
                'kurtosis': f"{df['Daily_Return'].kurtosis():.2f}"
            }
        }

        if self.is_fitted and 'Crash_Probability' in df.columns:
            report['ml_performance'] = {
                'avg_crash_probability': f"{df['Crash_Probability'].mean():.2f}",
                'high_risk_days': int(np.sum(df['Crash_Probability'] > 0.7)),
                'model_accuracy': 'Available after backtesting'
            }

        return report

    def run_analysis(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Run complete analysis pipeline"""
        print("Starting Advanced Market Crash Detection Analysis...")
        print("=" * 50)

        # Load data
        print("1. Loading and validating data...")
        df = self.load_and_validate_data(file_path)

        # Calculate technical indicators
        print("2. Calculating technical indicators...")
        df = self.calculate_technical_indicators(df)

        # Detect regime changes
        print("3. Detecting market regime changes...")
        df = self.detect_regime_changes(df)

        # Create ML features
        print("4. Creating machine learning features...")
        df = self.create_ml_features(df)

        # Label crash events
        print("5. Labeling crash events...")
        df = self.label_crash_events(df)

        # Fit anomaly detection model
        print("6. Fitting anomaly detection model...")
        anomaly_results = self.fit_isolation_forest(df)

        # Fit supervised learning model
        print("7. Training supervised crash prediction model...")
        ml_results = self.fit_supervised_model(df)

        # Use predictions from ML results if available
        if ml_results["model"] is not None:
            df = ml_results["df_with_predictions"]

        # Generate comprehensive report
        print("8. Generating analysis report...")
        report = self.generate_report(df)

        # Create visualizations
        print("9. Creating visualizations...")
        self.create_comprehensive_visualization(df)

        print("\nAnalysis completed successfully!")
        print("=" * 50)

        return df, {
            "report": report,
            "anomaly_results": anomaly_results,
            "ml_results": ml_results,
        }

    def save_model(self, filepath: str):
        """Save trained models to disk"""
        if self.is_fitted:
            model_data = {
                "isolation_forest": self.isolation_forest,
                "random_forest": self.random_forest,
                "scaler": self.scaler,
                "config": self.config,
                "feature_importance": self.feature_importance,
            }
            joblib.dump(model_data, filepath)
            print(f"Models saved to: {filepath}")
        else:
            print("No fitted models to save. Please run analysis first.")

    def load_model(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.isolation_forest = model_data["isolation_forest"]
            self.random_forest = model_data["random_forest"]
            self.scaler = model_data["scaler"]
            self.config = model_data["config"]
            self.feature_importance = model_data["feature_importance"]
            self.is_fitted = True
            print(f"Models loaded from: {filepath}")
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def predict_crash_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict crash probability for new data"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please run analysis first.")

        # Calculate indicators and features
        df = self.calculate_technical_indicators(df)
        df = self.create_ml_features(df)

        # Prepare features
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "Date",
                "Close",
                "Crash_Label",
                "Traditional_Crash",
                "Drawdown_Crash",
                "Multi_Day_Crash",
                "Extreme_Return",
                "Volatility_Regime",
                "Return_Regime",
                "Price_Trend",
            ]
        ]

        numeric_features = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )
        X = df[numeric_features].fillna(method="ffill").fillna(0)
        X_scaled = self.scaler.transform(X)

        # Make predictions
        df["Crash_Probability"] = self.random_forest.predict_proba(X_scaled)[:, 1]
        df["ML_Prediction"] = self.random_forest.predict(X_scaled)
        df["Anomaly_Score"] = self.isolation_forest.decision_function(X_scaled)
        df["Is_Anomaly"] = np.where(self.isolation_forest.predict(X_scaled) == -1, 1, 0)

        return df

    def backtest_strategy(self, df: pd.DataFrame, threshold: float = 0.7) -> Dict:
        """Backtest crash prediction strategy"""
        if "Crash_Probability" not in df.columns:
            raise ValueError("Crash probabilities not available. Run analysis first.")

        # Create signals
        df["Signal"] = np.where(df["Crash_Probability"] > threshold, 1, 0)
        df["Actual"] = df["Crash_Label"]

        # Calculate performance metrics
        true_positives = np.sum((df["Signal"] == 1) & (df["Actual"] == 1))
        false_positives = np.sum((df["Signal"] == 1) & (df["Actual"] == 0))
        true_negatives = np.sum((df["Signal"] == 0) & (df["Actual"] == 0))
        false_negatives = np.sum((df["Signal"] == 0) & (df["Actual"] == 1))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / len(df)

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_signals": np.sum(df["Signal"]),
            "total_crashes": np.sum(df["Actual"]),
        }


def main():
    """Main function to run the analysis"""
    # Initialize detector
    config = CrashDetectionConfig(
        short_window=10,
        long_window=50,
        volatility_window=20,
        return_threshold=-0.02,
        volatility_threshold=0.03,
        drawdown_threshold=-0.1,
        outlier_contamination=0.1,
    )

    detector = MarketCrashDetector(config)

    # Example usage
    try:
        # Run analysis (replace with your data file path)
        df, results = detector.run_analysis("your_data_file.csv")

        # Print summary report
        print("\n" + "=" * 60)
        print("MARKET CRASH DETECTION ANALYSIS REPORT")
        print("=" * 60)

        report = results["report"]

        print("\nDATA SUMMARY:")
        for key, value in report["data_summary"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\nCRASH ANALYSIS:")
        for key, value in report["crash_analysis"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\nRISK METRICS:")
        for key, value in report["risk_metrics"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        if "ml_performance" in report:
            print("\nML PERFORMANCE:")
            for key, value in report["ml_performance"].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")

        # Backtest strategy
        if detector.is_fitted and "Crash_Probability" in df.columns:
            print("\nBACKTEST RESULTS:")
            backtest = detector.backtest_strategy(df, threshold=0.5)
            for key, value in backtest.items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Save model
        detector.save_model("crash_detection_model.pkl")

        print("\n" + "=" * 60)
        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Using sample data for demonstration...")

        # Run with sample data
        df, results = detector.run_analysis(None)  # Will generate sample data
        print("Sample analysis completed!")


if __name__ == "__main__":
    main()
