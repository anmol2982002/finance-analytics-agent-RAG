"""
Finance Analytics Agent - Streamlit Dashboard
Interactive web UI for financial analysis with AI
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import our modules
from src.data import YahooFetcher, CoinGeckoFetcher, NewsFetcher, SentimentFetcher
from src.models import (
    TechnicalAnalyzer, AnomalyDetector,
    SuggestionsEngine, PricePredictor, RegimeDetector, StrategyBacktester
)
from src.rag import FinanceVectorStore, FinanceRAGChain
from src.visualization import ChartGenerator, SuggestionsChartGenerator
from src.utils import setup_logger, get_settings

# Setup
setup_logger(log_level="INFO", log_to_file=False)

# Page config
st.set_page_config(
    page_title="🤖 Finance AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean Professional Theme (Icon-Safe)
st.markdown("""
<style>
    /* === GOOGLE FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* === GLOBAL STYLES - DO NOT override span/div fonts (breaks icons) === */
    .stApp {
        background: #F8F9FA;
    }
    
    /* Only apply font to specific text containers, NOT spans (icons use spans) */
    .stMarkdown p, .stMarkdown li, 
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stTextInput label, .stSelectbox label, .stRadio label, .stCheckbox label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1F2937;
    }
    
    /* === HEADERS === */
    h1 { color: #111827; font-weight: 700; font-size: 2rem; }
    h2 { color: #1F2937; font-weight: 600; font-size: 1.5rem; }
    h3 { color: #374151; font-weight: 600; font-size: 1.25rem; }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    
    [data-testid="stSidebar"] h1 { color: #111827; font-size: 1.4rem; }
    [data-testid="stSidebar"] .stMarkdown p { color: #374151; }
    
    /* === METRIC CARDS === */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    [data-testid="stMetricLabel"] { color: #6B7280; font-size: 0.8rem; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #111827; font-size: 1.75rem; font-weight: 700; }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #E5E7EB;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        color: #6B7280;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2563EB;
        color: #FFFFFF;
    }
    
    /* === BUTTONS === */
    .stButton > button {
        background: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: #1D4ED8;
    }
    
    /* === NEWS CARDS === */
    .news-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #2563EB;
    }
    .news-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* === ALERTS === */
    .stSuccess { background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; }
    .stWarning { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; }
    .stError { background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; }
    .stInfo { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; }
    
    /* === INPUT FIELDS === */
    .stTextInput input, .stSelectbox > div > div {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
    }
    
    /* === DIVIDERS === */
    hr { border-color: #E5E7EB; }
    
    /* === LINKS === */
    a { color: #2563EB; text-decoration: none; }
    a:hover { color: #1D4ED8; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# Educational tooltips for beginners
TOOLTIPS = {
    "rsi": "**RSI (Relative Strength Index)**: Measures if a stock is overbought (>70, might drop) or oversold (<30, might rise). Range: 0-100.",
    "macd": "**MACD**: Shows momentum direction. When MACD crosses above signal line = bullish. Below = bearish.",
    "bollinger": "**Bollinger Bands**: When price touches upper band = potentially overbought. Lower band = potentially oversold.",
    "var": "**VaR (Value at Risk)**: The maximum loss you might expect with 95% confidence. Example: -5% VaR means only 5% chance of losing more than 5%.",
    "sharpe": "**Sharpe Ratio**: Risk-adjusted return. Higher is better. Above 1 = good, above 2 = great, above 3 = excellent.",
    "confidence": "**Confidence Score**: How sure the AI is about this suggestion. Below 60% = wait for more confirmation.",
    "regime": "**Market Regime**: Current market condition - Bull (rising), Bear (falling), or Volatile (choppy). Strategies should adapt to regimes.",
    "entry_zone": "**Entry Zone**: Price levels where buying has historically worked well. Higher probability = stronger support.",
    "stop_loss": "**Stop Loss**: Price level to exit if the trade goes wrong, limiting your losses.",
    "fear_greed": "**Fear & Greed Index**: Market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed). Extreme values often signal reversals."
}


# Initialize session state
@st.cache_resource
def get_fetchers():
    """Initialize data fetchers"""
    return {
        "yahoo": YahooFetcher(),
        "crypto": CoinGeckoFetcher(),
        "news": NewsFetcher(),
        "sentiment": SentimentFetcher()
    }


@st.cache_resource
def get_analyzers():
    """Initialize analyzers"""
    return {
        "technical": TechnicalAnalyzer(),
        "anomaly": AnomalyDetector()
    }


@st.cache_resource
def get_chart_generator():
    """Initialize chart generator"""
    return ChartGenerator(dark_mode=False)


@st.cache_resource
def get_suggestions_chart_generator():
    """Initialize suggestions chart generator"""
    return SuggestionsChartGenerator(dark_mode=False)


@st.cache_resource
def get_ml_analyzers():
    """Initialize ML suggestion analyzers"""
    return {
        "suggestions": SuggestionsEngine(n_simulations=5000),
        "predictor": PricePredictor(n_simulations=3000),
        "regime": RegimeDetector(),
        "backtester": StrategyBacktester()
    }


@st.cache_resource
def get_rag_chain():
    """Initialize RAG chain (may fail without API key)"""
    try:
        vector_store = FinanceVectorStore()
        return FinanceRAGChain(vector_store=vector_store)
    except Exception as e:
        st.warning(f"RAG chain not available: {e}")
        return None


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = "AAPL"
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0  # For tab persistence
    if "ticker_search_results" not in st.session_state:
        st.session_state.ticker_search_results = []  # For autocomplete


def render_sidebar():
    """Render the sidebar with controls"""
    fetchers = get_fetchers()
    
    with st.sidebar:
        st.title("🤖 Finance AI Analyst")
        st.markdown("---")
        
        # Asset type selection
        asset_type = st.radio(
            "Asset Type",
            ["Stock", "Crypto"],
            horizontal=True
        )
        
        # Ticker input with autocomplete
        if asset_type == "Stock":
            # Text input for search
            search_query = st.text_input(
                "🔍 Search Ticker",
                value=st.session_state.current_ticker,
                placeholder="Type ticker or company name..."
            ).upper()
            
            # Search for suggestions if query changed
            if search_query and len(search_query) >= 1:
                suggestions = fetchers["yahoo"].search_tickers(search_query, limit=6)
                
                if suggestions:
                    # Create selectbox options
                    options = [f"{s['symbol']} - {s['name']}" for s in suggestions]
                    
                    selected = st.selectbox(
                        "Select from suggestions:",
                        options,
                        index=0,
                        label_visibility="collapsed"
                    )
                    
                    # Extract ticker from selection
                    ticker = selected.split(" - ")[0] if selected else search_query
                else:
                    ticker = search_query
                    st.caption("💡 Try: AAPL, MSFT, GOOGL, TSLA")
            else:
                ticker = search_query or "AAPL"
        else:
            crypto_options = {
                "Bitcoin": "BTC-USD",
                "Ethereum": "ETH-USD",
                "Solana": "SOL-USD",
                "Cardano": "ADA-USD",
                "XRP": "XRP-USD",
                "Dogecoin": "DOGE-USD",
                "Polygon": "MATIC-USD"
            }
            selected_crypto = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
            ticker = crypto_options[selected_crypto]
        
        st.session_state.current_ticker = ticker
        
        # Time period
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("Analysis Options")
        
        show_technical = st.checkbox("Technical Indicators", value=True)
        show_anomalies = st.checkbox("Anomaly Detection", value=True)
        show_news = st.checkbox("News & Sentiment", value=True)
        
        st.markdown("---")
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        # RAG Knowledge Base stats
        st.markdown("---")
        st.subheader("📚 Knowledge Base")
        try:
            rag_chain = get_rag_chain()
            if rag_chain and rag_chain.vector_store:
                stats = rag_chain.vector_store.get_stats()
                doc_count = stats.get("total_documents", 0)
                st.metric("Documents Stored", doc_count)
                if doc_count > 0:
                    st.caption("✅ RAG context available for AI Chat")
                else:
                    st.caption("⚠️ Analyze a stock to populate knowledge base")
        except:
            st.caption("RAG not initialized")
        
        # Info
        st.markdown("---")
        st.caption("Powered by:")
        st.caption("• Yahoo Finance & CoinGecko")
        st.caption("• Groq LLM (Llama 3.3)")
        st.caption("• ChromaDB + RAG")
        
        return {
            "ticker": ticker,
            "period": period,
            "asset_type": asset_type,
            "show_technical": show_technical,
            "show_anomalies": show_anomalies,
            "show_news": show_news,
            "analyze_clicked": analyze_clicked
        }


def fetch_and_analyze(config):
    """Fetch data and run analysis"""
    fetchers = get_fetchers()
    analyzers = get_analyzers()
    
    ticker = config["ticker"]
    period = config["period"]
    
    with st.spinner(f"Fetching data for {ticker}..."):
        # Fetch price data
        if config["asset_type"] == "Stock":
            df = fetchers["yahoo"].get_historical_data(ticker, period=period)
            price_info = fetchers["yahoo"].get_realtime_price(ticker)
            fundamentals = fetchers["yahoo"].get_fundamentals(ticker)
        else:
            df = fetchers["yahoo"].get_historical_data(ticker, period=period)
            price_info = fetchers["yahoo"].get_realtime_price(ticker)
            fundamentals = {}
        
        if df.empty:
            st.error(f"No data found for {ticker}. Please check the symbol.")
            return None
    
    with st.spinner("Running technical analysis..."):
        # Calculate indicators
        df = analyzers["technical"].calculate_all_indicators(df)
        signals = analyzers["technical"].generate_signals(df)
        support_resistance = analyzers["technical"].get_support_resistance(df)
    
    with st.spinner("Detecting anomalies..."):
        # Anomaly detection
        df = analyzers["anomaly"].detect_all_anomalies(df)
        anomaly_summary = analyzers["anomaly"].get_anomaly_summary(df)
    
    # Fetch news and sentiment
    news = []
    sentiment = {}
    
    if config["show_news"]:
        with st.spinner("Fetching news and sentiment..."):
            news = fetchers["news"].get_ticker_news(ticker.replace("-USD", ""))
            sentiment = fetchers["sentiment"].get_fear_greed_index()
            
            # AUTO-INGEST NEWS INTO VECTOR STORE FOR RAG
            if news:
                try:
                    rag_chain = get_rag_chain()
                    if rag_chain and rag_chain.vector_store:
                        ingested = rag_chain.vector_store.add_news_articles(news, ticker=ticker)
                        if ingested > 0:
                            st.toast(f"📰 Ingested {ingested} news articles into knowledge base")
                except Exception as e:
                    pass  # Silently fail if RAG not available
    
    return {
        "df": df,
        "price_info": price_info,
        "fundamentals": fundamentals,
        "signals": signals,
        "support_resistance": support_resistance,
        "anomaly_summary": anomaly_summary,
        "news": news[:10],
        "sentiment": sentiment
    }


def render_metrics(results, ticker):
    """Render key metrics cards"""
    price_info = results.get("price_info", {})
    signals = results.get("signals", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = price_info.get("current_price", 0)
    prev_close = price_info.get("previous_close", current_price)
    change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
    
    with col1:
        st.metric(
            label=f"{ticker} Price",
            value=f"${current_price:,.2f}" if current_price else "N/A",
            delta=f"{change:+.2f}%" if change else None
        )
    
    with col2:
        overall = signals.get("overall", {})
        signal_text = overall.get("signal", "N/A")
        confidence = overall.get("confidence", 0)
        st.metric(
            label="Signal",
            value=signal_text,
            delta=f"{confidence:.0f}% confidence"
        )
    
    with col3:
        rsi = results["df"]["rsi"].iloc[-1] if "rsi" in results["df"].columns else 0
        st.metric(
            label="RSI (14)",
            value=f"{rsi:.1f}",
            delta="Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
        )
    
    with col4:
        fg = results.get("sentiment", {})
        fg_value = fg.get("value", "N/A")
        fg_class = fg.get("classification", "")
        if isinstance(fg_value, (int, float)):
            st.metric(
                label="Fear & Greed",
                value=fg_value,
                delta=fg_class
            )
        else:
            st.metric(label="Fear & Greed", value="N/A")
    
    # Educational info for beginners
    with st.expander("ℹ️ What do these metrics mean? (click to learn)", expanded=False):
        st.markdown("""
        | Metric | What It Means | How to Use It |
        |--------|--------------|---------------|
        | **Price** | Current trading price and % change from yesterday | Green = up, Red = down |
        | **Signal** | AI's overall recommendation based on 30+ indicators | BULLISH = consider buying, BEARISH = consider selling, NEUTRAL = wait |
        | **RSI (14)** | Measures if the price moved too fast in one direction | Above 70 = "overbought" (might drop), Below 30 = "oversold" (might rise) |
        | **Fear & Greed** | Market sentiment (0-100) | Extreme Fear (<25) = potential buying opportunity, Extreme Greed (>75) = be cautious |
        
        > 💡 **Tip**: No single metric tells the whole story. Always look at multiple signals together!
        """)



def render_chart(results, ticker, config):
    """Render main analysis chart"""
    chart_gen = get_chart_generator()
    
    fig = chart_gen.create_analysis_chart(
        df=results["df"],
        ticker=ticker,
        show_anomalies=config["show_anomalies"]
    )
    
    st.plotly_chart(fig, width='stretch')


def render_signals(results):
    """Render signal analysis"""
    signals = results.get("signals", {})
    
    st.subheader("📊 Technical Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trend Analysis**")
        trend = signals.get("trend", {})
        overall_trend = trend.get("overall_trend", "N/A")
        
        if overall_trend == "BULLISH":
            st.success(f"🟢 {overall_trend}")
        elif overall_trend == "BEARISH":
            st.error(f"🔴 {overall_trend}")
        else:
            st.info(f"🟡 {overall_trend}")
        
        if "moving_averages" in trend:
            ma = trend["moving_averages"]
            st.caption(ma.get("description", ""))
    
    with col2:
        st.markdown("**Momentum**")
        momentum = signals.get("momentum", {})
        
        if "rsi" in momentum:
            rsi = momentum["rsi"]
            rsi_signal = rsi.get("signal", "N/A")
            if rsi_signal == "OVERBOUGHT":
                st.warning(f"⚠️ RSI: {rsi.get('value', 0):.1f} - Overbought")
            elif rsi_signal == "OVERSOLD":
                st.success(f"✅ RSI: {rsi.get('value', 0):.1f} - Oversold")
            else:
                st.info(f"ℹ️ RSI: {rsi.get('value', 0):.1f} - Neutral")
    
    with col3:
        st.markdown("**Volatility**")
        volatility = signals.get("volatility", {})
        
        if "volatility" in volatility:
            vol = volatility["volatility"]
            level = vol.get("level", "N/A")
            ann_vol = vol.get("annualized", 0)
            
            if level == "HIGH":
                st.error(f"🔥 {ann_vol:.1f}% - High Volatility")
            elif level == "LOW":
                st.success(f"😌 {ann_vol:.1f}% - Low Volatility")
            else:
                st.info(f"📊 {ann_vol:.1f}% - Moderate")
    
    # Support and Resistance
    st.markdown("---")
    sr = results.get("support_resistance", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Support Levels**")
        for i, level in enumerate(sr.get("support", []), 1):
            st.write(f"S{i}: ${level:,.2f}")
    
    with col2:
        st.markdown("**Resistance Levels**")
        for i, level in enumerate(sr.get("resistance", []), 1):
            st.write(f"R{i}: ${level:,.2f}")


def render_news(results):
    """Render news section"""
    news = results.get("news", [])
    
    st.subheader("📰 Latest News")
    
    if not news:
        st.info("No recent news available.")
        return
    
    for article in news[:5]:
        with st.container():
            st.markdown(f"""
            <div class="news-card">
                <strong>{article.get('title', 'No title')}</strong><br>
                <small>{article.get('source', 'Unknown')} | {article.get('published_at', '')[:10] if article.get('published_at') else 'Unknown date'}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if article.get('url'):
                st.markdown(f"[Read more]({article['url']})")


def render_suggestions_tab(results, ticker, config):
    """Render the ML Suggestions tab"""
    st.subheader("🎯 ML-Powered Suggestions")
    
    # Educational intro for beginners
    with st.expander("ℹ️ How to read this page (for beginners)", expanded=False):
        st.markdown("""
        ### Understanding ML Suggestions
        
        This page uses **machine learning** to analyze the stock and give you actionable insights:
        
        | Section | What It Shows | Why It Matters |
        |---------|--------------|----------------|
        | **Signal** | BULLISH/BEARISH/NEUTRAL | Overall AI recommendation based on 30+ factors |
        | **Confidence** | 0-100% | How sure the AI is. Below 60% = be cautious |
        | **Market Regime** | Bull/Bear/Volatile | Current market conditions - strategies should adapt |
        | **Entry Zones** | Price levels to buy | Areas where the stock has historically found support |
        | **Target Zones** | Price levels to sell | Areas where the stock has historically faced resistance |
        | **Stop Loss** | Where to exit if wrong | Protects you from big losses |
        | **Risk Analysis** | VaR, loss probability | How much you could lose in worst-case scenarios |
        
        > ⚠️ **Disclaimer**: This is for educational purposes only. Always do your own research before investing!
        """)
    
    df = results.get("df")
    signals = results.get("signals", {})
    
    if df is None or df.empty:
        st.warning("No data available for suggestions analysis.")
        return
    
    # Get ML analyzers

    ml_analyzers = get_ml_analyzers()
    suggestions_chart = get_suggestions_chart_generator()
    
    with st.spinner("Running ML analysis..."):
        # Generate suggestions
        suggestion_report = ml_analyzers["suggestions"].generate_suggestions(
            ticker=ticker,
            df=df,
            technical_signals=signals,
            include_monte_carlo=True
        )
        
        # Detect regime
        regime_state = ml_analyzers["regime"].detect_current_regime(df)
        
        # Generate predictions
        predictions = ml_analyzers["predictor"].predict(df, horizon=7, ticker=ticker)
    
    # Store in results for RAG chain
    results["suggestion_report"] = suggestion_report.to_dict()
    results["regime_state"] = regime_state.to_dict()
    results["predictions"] = predictions.to_dict()
    
    # === Primary Signal Card ===
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        signal = suggestion_report.primary_signal
        confidence = suggestion_report.confidence
        
        if signal == "BULLISH":
            st.success(f"### 📈 {signal}")
        elif signal == "BEARISH":
            st.error(f"### 📉 {signal}")
        else:
            st.info(f"### ⏸️ {signal}")
        
        st.metric("Confidence", f"{confidence:.1f}%")
        st.caption(suggestion_report.confidence_explanation)
    
    with col2:
        st.markdown("**Market Regime**")
        regime = regime_state.regime
        if regime == "BULL_TREND":
            st.success(f"🟢 {regime}")
        elif regime == "BEAR_TREND":
            st.error(f"🔴 {regime}")
        elif regime == "HIGH_VOLATILITY":
            st.warning(f"🟠 {regime}")
        else:
            st.info(f"🔵 {regime}")
        st.caption(f"Duration: {regime_state.duration_days} days")
    
    with col3:
        st.markdown("**Regime Change**")
        st.metric("Probability", f"{regime_state.probability_of_change:.0f}%")
        st.caption(f"Next likely: {regime_state.most_likely_next_regime}")
    
    st.markdown("---")
    
    # === Summary Card ===
    st.markdown(suggestion_report.summary)
    
    if suggestion_report.warnings:
        for warning in suggestion_report.warnings:
            st.warning(warning)
    
    st.markdown("---")
    
    # === Tabs for detailed analysis ===
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 Entry/Exit Zones", "📈 Price Forecast", "⚠️ Risk Analysis", "📜 Historical Patterns"
    ])
    
    with sub_tab1:
        st.subheader("Entry & Exit Zones")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🟢 Entry Zones (Support)**")
            for zone in suggestion_report.entry_zones:
                strength_emoji = "💪" if zone.strength == "STRONG" else "👍" if zone.strength == "MODERATE" else "👌"
                st.write(f"{strength_emoji} ${zone.price_low:.2f} - ${zone.price_high:.2f}")
                st.caption(f"Probability: {zone.probability:.0f}%")
        
        with col2:
            st.markdown("**🔴 Target Zones (Resistance)**")
            for zone in suggestion_report.target_zones:
                st.write(f"🎯 ${zone.price_low:.2f} - ${zone.price_high:.2f}")
                st.caption(f"Probability: {zone.probability:.0f}%")
        
        with col3:
            st.markdown("**🛑 Stop Loss Zones**")
            for zone in suggestion_report.stop_loss_zones:
                st.write(f"⛔ ${zone.price_low:.2f} - ${zone.price_high:.2f}")
        
        # Key levels
        st.markdown("---")
        st.markdown("**Key Levels**")
        key_levels = suggestion_report.key_levels
        cols = st.columns(4)
        for i, (key, value) in enumerate(key_levels.items()):
            if value is not None:
                with cols[i % 4]:
                    st.metric(key.replace("_", " ").title(), f"${value:.2f}")
    
    with sub_tab2:
        st.subheader(f"7-Day Price Forecast for {ticker}")
        
        if predictions.median_prediction:
            # Create prediction cone chart
            fig = suggestions_chart.create_prediction_cone(
                df=df.tail(60),
                prediction_result=predictions.to_dict(),
                ticker=ticker
            )
            st.plotly_chart(fig, width='stretch')
            
            # Prediction table
            st.markdown("**Forecast Details**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Median (Day 7)", f"${predictions.median_prediction[-1]:.2f}")
            with col2:
                if predictions.band_80.upper:
                    st.metric("80% Upper", f"${predictions.band_80.upper[-1]:.2f}")
            with col3:
                if predictions.band_80.lower:
                    st.metric("80% Lower", f"${predictions.band_80.lower[-1]:.2f}")
            
            st.caption(f"Prediction Quality: {predictions.prediction_quality}")
            
            for warning in predictions.warnings:
                st.warning(warning)
        else:
            st.info("Insufficient data for price prediction.")
    
    with sub_tab3:
        st.subheader("Risk Analysis (Monte Carlo Simulation)")
        
        if suggestion_report.risk_metrics:
            risk = suggestion_report.risk_metrics
            
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("95% VaR (7d)", f"{risk.var_95:.1f}%",
                         delta="Risk" if risk.var_95 < -5 else None,
                         delta_color="inverse")
            
            with col2:
                st.metric("Loss Probability", f"{risk.probability_of_loss:.0f}%")
            
            with col3:
                st.metric("Median Outcome", f"{risk.median_outcome_7d:+.1f}%",
                         delta="Positive" if risk.median_outcome_7d > 0 else "Negative")
            
            with col4:
                st.metric("Expected Shortfall", f"{risk.expected_shortfall:.1f}%")
            
            st.markdown("---")
            
            # Outcome range
            st.markdown("**7-Day Outcome Range**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔴 Worst Case", f"{risk.worst_case_7d:+.1f}%")
            with col2:
                st.metric("🟡 Median", f"{risk.median_outcome_7d:+.1f}%")
            with col3:
                st.metric("🟢 Best Case", f"{risk.best_case_7d:+.1f}%")
            
            # Risk dashboard chart
            fig = suggestions_chart.create_risk_dashboard({
                "var_95": risk.var_95,
                "var_99": risk.var_99,
                "expected_shortfall": risk.expected_shortfall,
                "probability_of_loss": risk.probability_of_loss,
                "best_case_7d": risk.best_case_7d,
                "worst_case_7d": risk.worst_case_7d,
                "median_outcome_7d": risk.median_outcome_7d,
                "max_drawdown_expected": risk.max_drawdown_expected
            })
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Risk analysis not available.")
    
    with sub_tab4:
        st.subheader("Similar Historical Periods")
        
        if suggestion_report.similar_periods:
            for period in suggestion_report.similar_periods:
                outcome_emoji = "📈" if period.outcome == "BULLISH" else "📉" if period.outcome == "BEARISH" else "➡️"
                
                with st.expander(f"{outcome_emoji} {period.start_date} to {period.end_date} (Similarity: {period.similarity_score:.0f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("7-Day Return", f"{period.return_after_7d:+.1f}%")
                    with col2:
                        st.metric("30-Day Return", f"{period.return_after_30d:+.1f}%")
                    
                    st.write(period.description)
        else:
            st.info("No similar historical patterns found.")
        
        # Multi-timeframe confluence
        st.markdown("---")
        st.markdown("**Multi-Timeframe Confluence**")
        
        confluence = suggestion_report.timeframe_confluence
        cols = st.columns(len(confluence) if confluence else 3)
        
        for i, (timeframe, signal) in enumerate(confluence.items()):
            with cols[i]:
                if signal == "BULLISH":
                    st.success(f"**{timeframe.title()}**: 🟢 {signal}")
                elif signal == "BEARISH":
                    st.error(f"**{timeframe.title()}**: 🔴 {signal}")
                else:
                    st.info(f"**{timeframe.title()}**: 🟡 {signal}")
        
        st.metric("Confluence Score", f"{suggestion_report.confluence_score:+.0f}%")


def render_chat_interface(results, ticker):
    """Render the AI chat interface"""
    st.subheader("💬 Ask the AI Analyst")
    
    rag_chain = get_rag_chain()
    
    if not rag_chain:
        st.warning("AI chat requires GROQ_API_KEY. Set it in your .env file.")
        return
    
    # Show RAG context status
    if rag_chain.vector_store:
        doc_count = rag_chain.vector_store.collection.count()
        if doc_count > 0:
            st.success(f"📚 Knowledge Base: **{doc_count} documents** available for context")
        else:
            st.info("📚 No documents in knowledge base yet. Analyze stocks to populate it.")
    
    st.caption("Ask any question about the technical analysis, market signals, or get AI-powered insights.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if message.get("sources"):
                with st.expander("📎 Sources used"):
                    for source in message["sources"]:
                        st.caption(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask about the analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with RAG context..."):
                try:
                    # Get RAG context first
                    sources = []
                    if rag_chain.vector_store:
                        context_results = rag_chain.vector_store.search(
                            query=f"{ticker} {prompt}",
                            n_results=3,
                            ticker_filter=ticker
                        )
                        for result in context_results.get("results", []):
                            meta = result.get("metadata", {})
                            if meta.get("title"):
                                sources.append(f"{meta.get('source', 'Unknown')} - {meta.get('title', '')[:50]}...")
                    
                    response = rag_chain.analyze(
                        ticker=ticker,
                        question=prompt,
                        market_data=results.get("price_info", {}),
                        technical_signals=results.get("signals", {}),
                        sentiment_data=results.get("sentiment", {})
                    )
                    
                    st.markdown(response)
                    
                    # Show sources inline
                    if sources:
                        with st.expander("📎 Sources used for this response"):
                            for source in sources:
                                st.caption(f"• {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main application"""
    init_session_state()
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Main content
    if config["analyze_clicked"] or st.session_state.analysis_results:
        
        if config["analyze_clicked"]:
            results = fetch_and_analyze(config)
            if results:
                st.session_state.analysis_results = results
                st.session_state.current_data = results["df"]
        else:
            results = st.session_state.analysis_results
        
        if results:
            ticker = config["ticker"]
            
            # Header
            st.title(f"📊 {ticker} Analysis")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Metrics row
            render_metrics(results, ticker)
            
            st.markdown("---")
            
            # Main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Chart", "🎯 ML Suggestions", "📊 Signals", "📰 News", "💬 AI Chat"
            ])
            
            with tab1:
                render_chart(results, ticker, config)
            
            with tab2:
                render_suggestions_tab(results, ticker, config)
            
            with tab3:
                render_signals(results)
                
                # Anomaly summary
                if config["show_anomalies"]:
                    st.markdown("---")
                    st.subheader("🔍 Anomaly Detection")
                    anomaly = results.get("anomaly_summary", {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", anomaly.get("total_anomalies", 0))
                        st.metric("Anomaly Rate", anomaly.get("anomaly_rate", "0%"))
                    
                    with col2:
                        methods = anomaly.get("methods_triggered", {})
                        st.write("**Detection Methods:**")
                        for method, count in methods.items():
                            st.write(f"- {method}: {count}")
            
            with tab4:
                render_news(results)
                
                # Fear & Greed
                sentiment = results.get("sentiment", {})
                if sentiment and "value" in sentiment:
                    st.markdown("---")
                    chart_gen = get_chart_generator()
                    fig = chart_gen.create_fear_greed_gauge(sentiment["value"])
                    st.plotly_chart(fig, width='stretch')
                    st.caption(sentiment.get("interpretation", ""))
            
            with tab5:
                render_chat_interface(results, ticker)
    
    else:
        # Welcome screen
        st.title("🤖 Finance AI Analyst")
        st.markdown("""
        Welcome to the **AI-powered Financial Analytics Agent**!
        
        ### Features:
        - 📈 **Technical Analysis** - 30+ indicators with signal generation
        - 🔍 **Anomaly Detection** - Identify unusual price movements
        - 📰 **News Integration** - Latest headlines and sentiment
        - 💬 **AI Chat** - Ask questions about any asset
        - 📊 **Interactive Charts** - Professional visualizations
        
        ### Getting Started:
        1. Select **Stock** or **Crypto** in the sidebar
        2. Enter a ticker symbol (e.g., AAPL, BTC-USD)
        3. Click **Analyze** to start
        
        ---
        
        *Powered by Yahoo Finance, CoinGecko, Groq LLM, and more.*
        """)
        
        # Quick access buttons
        st.subheader("Quick Start")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📱 Apple (AAPL)", use_container_width=True):
                st.session_state.current_ticker = "AAPL"
                st.rerun()
        
        with col2:
            if st.button("🚀 NVIDIA (NVDA)", use_container_width=True):
                st.session_state.current_ticker = "NVDA"
                st.rerun()
        
        with col3:
            if st.button("₿ Bitcoin (BTC)", use_container_width=True):
                st.session_state.current_ticker = "BTC-USD"
                st.rerun()
        
        with col4:
            if st.button("⟠ Ethereum (ETH)", use_container_width=True):
                st.session_state.current_ticker = "ETH-USD"
                st.rerun()


if __name__ == "__main__":
    main()
