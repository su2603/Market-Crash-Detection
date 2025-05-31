import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats
import joblib
from datetime import datetime, timedelta
import io
import base64

# Configure page
st.set_page_config(
    page_title="Market Crash Detection System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suppress warnings
warnings.filterwarnings("ignore")


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


class StreamlitMarketCrashDetector:
    """Streamlit-optimized Market Crash Detection System"""

    def __init__(self, config: CrashDetectionConfig = None):
        self.config = config or CrashDetectionConfig()
        self.scaler = RobustScaler()
        self.isolation_forest = None
        self.random_forest = None
        self.is_fitted = False
        self.feature_importance = None

    def load_and_validate_data(self, uploaded_file) -> pd.DataFrame:
        """Load and validate market data from uploaded file"""
        try:
            if uploaded_file is not None:
                # Try to read the uploaded file
                df = pd.read_csv(uploaded_file)

                # Validate required columns
                if "Date" not in df.columns or "Close" not in df.columns:
                    # Try to find similar columns
                    date_cols = [
                        col
                        for col in df.columns
                        if "date" in col.lower() or "time" in col.lower()
                    ]
                    price_cols = [
                        col
                        for col in df.columns
                        if any(
                            x in col.lower() for x in ["close", "price", "value", "adj"]
                        )
                    ]

                    if date_cols and price_cols:
                        df = df.rename(
                            columns={date_cols[0]: "Date", price_cols[0]: "Close"}
                        )
                        st.success(
                            f"Renamed columns: {date_cols[0]} → Date, {price_cols[0]} → Close"
                        )
                    else:
                        st.error(
                            "Required columns 'Date' and 'Close' not found in the data"
                        )
                        return None

                # Clean and convert data
                df = self._clean_and_convert_data(df)
                return df
            else:
                # Generate sample data
                return self._generate_sample_data()

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return self._generate_sample_data()

    def _clean_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types"""
        # Handle Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Handle Close price column
        if df["Close"].dtype == "object":
            df["Close"] = df["Close"].astype(str).str.replace(r"[₹$,]", "", regex=True)
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        # Remove invalid data and sort
        df = (
            df.dropna(subset=["Date", "Close"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
        df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

        return df

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data"""
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
        np.random.seed(42)

        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add crash periods
        crash_periods = [500, 800, 1200, 1500]
        for period in crash_periods:
            if period < len(returns):
                returns[period : period + 10] = np.random.normal(-0.05, 0.05, 10)

        prices = [1000]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({"Date": dates[: len(prices)], "Close": prices})

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()

        # Basic metrics
        df["Daily_Return"] = df["Close"].pct_change()
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving averages
        df[f"SMA_{self.config.short_window}"] = (
            df["Close"].rolling(self.config.short_window).mean()
        )
        df[f"SMA_{self.config.long_window}"] = (
            df["Close"].rolling(self.config.long_window).mean()
        )
        df[f"EMA_{self.config.short_window}"] = (
            df["Close"].ewm(span=self.config.short_window).mean()
        )

        # Volatility
        df["Rolling_Volatility"] = (
            df["Daily_Return"].rolling(self.config.volatility_window).std()
        )

        # Drawdown
        df["Cumulative_Max"] = df["Close"].cummax()
        df["Drawdown"] = (df["Close"] - df["Cumulative_Max"]) / df["Cumulative_Max"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        df["BB_Middle"] = df["Close"].rolling(bb_window).mean()
        bb_std = df["Close"].rolling(bb_window).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        # MACD
        exp1 = df["Close"].ewm(span=12).mean()
        exp2 = df["Close"].ewm(span=26).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

        return df

    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features"""
        df = df.copy()

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)
            df[f"Volatility_Lag_{lag}"] = df["Rolling_Volatility"].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"Return_Mean_{window}"] = df["Daily_Return"].rolling(window).mean()
            df[f"Return_Std_{window}"] = df["Daily_Return"].rolling(window).std()

        # Momentum
        for period in [5, 10, 20]:
            df[f"Momentum_{period}"] = df["Close"].pct_change(period)

        return df

    def label_crash_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label crash events"""
        df = df.copy()

        # Traditional crash criteria
        df["Traditional_Crash"] = np.where(
            (df["Daily_Return"] < self.config.return_threshold)
            & (df["Rolling_Volatility"] > self.config.volatility_threshold),
            1,
            0,
        )

        # Drawdown crash
        df["Drawdown_Crash"] = np.where(
            df["Drawdown"] < self.config.drawdown_threshold, 1, 0
        )

        # Extreme returns
        returns_clean = df["Daily_Return"].dropna()
        if len(returns_clean) > 0:
            # Calculate z-scores for non-null values
            z_scores = stats.zscore(returns_clean)
            # Create a series with the same index as the original data, filled with NaN
            df["Return_ZScore"] = np.nan
            # Fill in the z-scores for the non-null positions
            df.loc[returns_clean.index, "Return_ZScore"] = z_scores
        else:
            df["Return_ZScore"] = np.nan

        df["Extreme_Return"] = np.where(abs(df["Return_ZScore"]) > 2.5, 1, 0)

        # Combined label
        df["Crash_Label"] = np.where(
            (df["Traditional_Crash"] == 1)
            | (df["Drawdown_Crash"] == 1)
            | (df["Extreme_Return"] == 1),
            1,
            0,
        )

        return df

    def fit_models(self, df: pd.DataFrame) -> Dict:
        """Fit ML models"""
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
                "Extreme_Return",
            ]
        ]

        numeric_features = (
            df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )
        X = df[numeric_features].fillna(method="ffill").fillna(0)
        y = df["Crash_Label"].fillna(0)

        # Isolation Forest
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest = IsolationForest(
            contamination=self.config.outlier_contamination, random_state=42
        )
        anomaly_labels = self.isolation_forest.fit_predict(X_scaled)
        df["Anomaly_Score"] = self.isolation_forest.decision_function(X_scaled)
        df["Is_Anomaly"] = np.where(anomaly_labels == -1, 1, 0)

        # Random Forest (if sufficient crash events)
        if np.sum(y) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.random_forest = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            )
            self.random_forest.fit(X_train_scaled, y_train)

            # Feature importance
            self.feature_importance = pd.DataFrame(
                {
                    "feature": numeric_features,
                    "importance": self.random_forest.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            # Predictions
            X_all_scaled = self.scaler.transform(X)
            df["Crash_Probability"] = self.random_forest.predict_proba(X_all_scaled)[
                :, 1
            ]
            df["ML_Prediction"] = self.random_forest.predict(X_all_scaled)

            # Performance
            y_pred = self.random_forest.predict(X_test_scaled)
            y_pred_proba = self.random_forest.predict_proba(X_test_scaled)[:, 1]

            performance = {
                "accuracy": np.mean(y_pred == y_test),
                "roc_auc": (
                    roc_auc_score(y_test, y_pred_proba)
                    if len(np.unique(y_test)) > 1
                    else 0
                ),
            }

            self.is_fitted = True
            return {"performance": performance, "df": df}

        return {"performance": None, "df": df}


def create_plotly_charts(df: pd.DataFrame, detector: StreamlitMarketCrashDetector):
    """Create interactive Plotly charts"""

    # Price chart with crash events
    fig_price = go.Figure()

    fig_price.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue", width=2),
        )
    )

    # Add crash events
    crash_events = df[df["Crash_Label"] == 1]
    if len(crash_events) > 0:
        fig_price.add_trace(
            go.Scatter(
                x=crash_events["Date"],
                y=crash_events["Close"],
                mode="markers",
                name="Crash Events",
                marker=dict(color="red", size=8, symbol="triangle-down"),
            )
        )

    fig_price.update_layout(
        title="Stock Price with Crash Events",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    # Volatility chart
    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Rolling_Volatility"],
            mode="lines",
            name="Rolling Volatility",
            line=dict(color="orange", width=2),
        )
    )
    fig_vol.add_hline(
        y=detector.config.volatility_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
    )
    fig_vol.update_layout(
        title="Rolling Volatility", xaxis_title="Date", yaxis_title="Volatility"
    )

    # Drawdown chart
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Drawdown"],
            mode="lines",
            name="Drawdown",
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="red", width=2),
        )
    )
    fig_dd.add_hline(
        y=detector.config.drawdown_threshold,
        line_dash="dash",
        line_color="darkred",
        annotation_text="Threshold",
    )
    fig_dd.update_layout(
        title="Drawdown Analysis", xaxis_title="Date", yaxis_title="Drawdown"
    )

    # Returns distribution
    fig_dist = go.Figure(data=[go.Histogram(x=df["Daily_Return"].dropna(), nbinsx=50)])
    fig_dist.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
    )

    return fig_price, fig_vol, fig_dd, fig_dist


def main():
    """Main Streamlit application"""

    # Header
    st.title("🚨 Market Crash Detection System")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Configuration parameters
    with st.sidebar.expander("Detection Parameters", expanded=True):
        short_window = st.slider("Short Window", 5, 30, 10)
        long_window = st.slider("Long Window", 30, 100, 50)
        volatility_window = st.slider("Volatility Window", 10, 50, 20)
        return_threshold = st.slider("Return Threshold", -0.1, 0.0, -0.02, 0.005)
        volatility_threshold = st.slider("Volatility Threshold", 0.01, 0.1, 0.03, 0.005)
        drawdown_threshold = st.slider("Drawdown Threshold", -0.5, 0.0, -0.1, 0.05)

    # Create config
    config = CrashDetectionConfig(
        short_window=short_window,
        long_window=long_window,
        volatility_window=volatility_window,
        return_threshold=return_threshold,
        volatility_threshold=volatility_threshold,
        drawdown_threshold=drawdown_threshold,
    )

    # Initialize detector
    detector = StreamlitMarketCrashDetector(config)

    # File upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with Date and Close columns",
        type=["csv"],
        help="File should contain 'Date' and 'Close' columns",
    )

    if st.sidebar.button(" Run Analysis", type="primary"):
        with st.spinner("Loading and analyzing data..."):
            # Load data
            df = detector.load_and_validate_data(uploaded_file)

            if df is not None:
                # Store in session state
                st.session_state.df = df
                st.session_state.detector = detector

                # Calculate indicators
                df = detector.calculate_technical_indicators(df)
                df = detector.create_ml_features(df)
                df = detector.label_crash_events(df)

                # Fit models
                results = detector.fit_models(df)
                df = results["df"]

                st.session_state.df_processed = df
                st.session_state.results = results

                st.success("✅ Analysis completed successfully!")

    # Display results if available
    if "df_processed" in st.session_state:
        df = st.session_state.df_processed
        detector = st.session_state.detector
        results = st.session_state.results

        # Summary metrics
        st.header("📊 Summary Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Observations", f"{len(df):,}", help="Total number of data points"
            )

        with col2:
            total_crashes = df["Crash_Label"].sum()
            crash_rate = (total_crashes / len(df)) * 100
            st.metric("Crash Events", f"{total_crashes}", f"{crash_rate:.2f}% rate")

        with col3:
            max_drawdown = df["Drawdown"].min() * 100
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.2f}%",
                help="Maximum peak-to-trough decline",
            )

        with col4:
            volatility = df["Daily_Return"].std() * np.sqrt(252) * 100
            st.metric(
                "Annualized Volatility",
                f"{volatility:.2f}%",
                help="Annualized price volatility",
            )

        # Charts
        st.header("📈 Interactive Charts")

        fig_price, fig_vol, fig_dd, fig_dist = create_plotly_charts(df, detector)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Price & Crashes", "Volatility", "Drawdown", "Returns Distribution"]
        )

        with tab1:
            st.plotly_chart(fig_price, use_container_width=True)

        with tab2:
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab3:
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab4:
            st.plotly_chart(fig_dist, use_container_width=True)

        # ML Results
        if results["performance"] is not None:
            st.header(" Machine Learning Results")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                perf = results["performance"]
                st.metric("Accuracy", f"{perf['accuracy']:.3f}")
                st.metric("ROC AUC", f"{perf['roc_auc']:.3f}")

            with col2:
                if detector.feature_importance is not None:
                    st.subheader("Top Features")
                    top_features = detector.feature_importance.head(10)
                    fig_importance = px.bar(
                        top_features,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Feature Importance",
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

            # Crash probability chart
            if "Crash_Probability" in df.columns:
                st.subheader("Crash Probability Over Time")
                fig_prob = go.Figure()
                fig_prob.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["Crash_Probability"],
                        mode="lines",
                        name="Crash Probability",
                        line=dict(color="red", width=2),
                    )
                )
                fig_prob.add_hline(y=0.5, line_dash="dash", line_color="black")
                fig_prob.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        # Data table
        st.header("📋 Detailed Data")

        # Show recent data
        display_cols = [
            "Date",
            "Close",
            "Daily_Return",
            "Rolling_Volatility",
            "Drawdown",
            "RSI",
            "Crash_Label",
        ]
        if "Crash_Probability" in df.columns:
            display_cols.append("Crash_Probability")

        recent_data = df[display_cols].tail(100).round(4)
        st.dataframe(recent_data, use_container_width=True)

        # Download processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Processed Data",
            data=csv,
            file_name=f"crash_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    else:
        # Welcome message
        st.header(" Welcome to Market Crash Detection System")
        st.markdown(
            """
        This advanced system uses machine learning and statistical methods to detect and predict market crashes.
        
        **Features:**
        - 📊 Technical indicator analysis
        - 🤖 Machine learning crash prediction
        - 📈 Interactive visualizations
        - 🔍 Anomaly detection
        - 📋 Comprehensive reporting
        
        **Getting Started:**
        1. Adjust parameters in the sidebar
        2. Upload your CSV file (with Date and Close columns) or use sample data
        3. Click "Run Analysis" to start
        
        **Sample Data:**
        If no file is uploaded, the system will generate sample market data for demonstration.
        """
        )

        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "Close": [100.0, 102.5, 98.3],
                "Volume": [1000000, 1200000, 1500000],
            }
        )
        st.dataframe(sample_df)
        st.caption(
            "Note: Only Date and Close columns are required. Volume is optional."
        )


if __name__ == "__main__":
    main()
