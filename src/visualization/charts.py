"""
Interactive Chart Generator
Creates professional financial charts using Plotly
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from src.utils import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """
    Generates interactive financial charts with Plotly.
    
    Features:
    - Candlestick charts with overlays
    - Technical indicator charts (RSI, MACD, etc.)
    - Volume analysis
    - Multi-panel layouts
    - Anomaly highlighting
    """
    
    # Color scheme
    COLORS = {
        "bullish": "#059669",  # Professional green
        "bearish": "#DC2626",  # Professional red
        "neutral": "#6B7280",
        "ma_20": "#F59E0B",    # Amber
        "ma_50": "#2563EB",    # Blue
        "ma_200": "#7C3AED",   # Purple
        "bb_band": "rgba(107, 114, 128, 0.15)",
        "volume_up": "rgba(5, 150, 105, 0.7)",
        "volume_down": "rgba(220, 38, 38, 0.7)",
        "anomaly": "#F59E0B",  # Amber for visibility on light bg
        "background": "#FFFFFF",
        "grid": "#E5E7EB"
    }
    
    def __init__(self, dark_mode: bool = True):
        """
        Initialize the chart generator.
        
        Args:
            dark_mode: Use dark theme (default: True)
        """
        self.dark_mode = dark_mode
        self.template = "plotly_dark" if dark_mode else "plotly_white"
    
    def create_candlestick_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        show_volume: bool = True,
        show_ma: bool = True,
        show_bb: bool = True,
        height: int = 600
    ) -> go.Figure:
        """
        Create a candlestick chart with optional overlays.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol for title
            show_volume: Include volume subplot
            show_ma: Show moving averages
            show_bb: Show Bollinger Bands
            height: Chart height in pixels
        
        Returns:
            Plotly Figure object
        """
        # Create subplots
        row_heights = [0.7, 0.3] if show_volume else [1.0]
        rows = 2 if show_volume else 1
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(ticker, "Volume") if show_volume else (ticker,)
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
                increasing_line_color=self.COLORS["bullish"],
                decreasing_line_color=self.COLORS["bearish"]
            ),
            row=1, col=1
        )
        
        # Moving Averages
        if show_ma:
            ma_configs = [
                ('sma_20', 'SMA 20', self.COLORS["ma_20"]),
                ('sma_50', 'SMA 50', self.COLORS["ma_50"]),
                ('sma_200', 'SMA 200', self.COLORS["ma_200"])
            ]
            
            for col, name, color in ma_configs:
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=1.5)
                        ),
                        row=1, col=1
                    )
        
        # Bollinger Bands
        if show_bb and 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    fill='tonexty',
                    fillcolor=self.COLORS["bb_band"],
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Volume
        if show_volume and 'Volume' in df.columns:
            colors = [
                self.COLORS["volume_up"] if close >= open_ 
                else self.COLORS["volume_down"]
                for close, open_ in zip(df['Close'], df['Open'])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Layout
        fig.update_layout(
            height=height,
            template=self.template,
            xaxis_rangeslider_visible=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor=self.COLORS["grid"])
        fig.update_yaxes(showgrid=True, gridcolor=self.COLORS["grid"])
        
        return fig
    
    def create_analysis_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        show_anomalies: bool = True,
        height: int = 900
    ) -> go.Figure:
        """
        Create comprehensive analysis chart with multiple panels.
        
        Args:
            df: DataFrame with indicators calculated
            ticker: Ticker symbol
            show_anomalies: Highlight anomaly points
            height: Chart height
        
        Returns:
            Plotly Figure with 4 panels
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f'{ticker} Price',
                'RSI (14)',
                'MACD',
                'Volume'
            )
        )
        
        # Panel 1: Candlestick with MAs and BBs
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color=self.COLORS["bullish"],
                decreasing_line_color=self.COLORS["bearish"]
            ),
            row=1, col=1
        )
        
        # Add MAs
        for col, name, color in [
            ('sma_20', 'SMA 20', self.COLORS["ma_20"]),
            ('sma_50', 'SMA 50', self.COLORS["ma_50"]),
            ('sma_200', 'SMA 200', self.COLORS["ma_200"])
        ]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col],
                        mode='lines', name=name,
                        line=dict(color=color, width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['bb_upper'],
                    mode='lines', name='BB',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['bb_lower'],
                    mode='lines', name='BB Lower',
                    fill='tonexty',
                    fillcolor=self.COLORS["bb_band"],
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Highlight anomalies
        if show_anomalies and 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            if len(anomalies) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies.index,
                        y=anomalies['High'] * 1.02,
                        mode='markers',
                        name='Anomaly',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.COLORS["anomaly"]
                        )
                    ),
                    row=1, col=1
                )
        
        # Panel 2: RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['rsi'],
                    mode='lines', name='RSI',
                    line=dict(color='#AB47BC', width=1.5)
                ),
                row=2, col=1
            )
            
            # Overbought/Oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         line_width=1, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                         line_width=1, row=2, col=1)
        
        # Panel 3: MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['macd'],
                    mode='lines', name='MACD',
                    line=dict(color='#26C6DA', width=1.5)
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['macd_signal'],
                    mode='lines', name='Signal',
                    line=dict(color='#FFA726', width=1.5)
                ),
                row=3, col=1
            )
            
            if 'macd_histogram' in df.columns:
                colors = [
                    self.COLORS["bullish"] if val >= 0 else self.COLORS["bearish"]
                    for val in df['macd_histogram']
                ]
                fig.add_trace(
                    go.Bar(
                        x=df.index, y=df['macd_histogram'],
                        name='Histogram',
                        marker_color=colors,
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        # Panel 4: Volume
        if 'Volume' in df.columns:
            colors = [
                self.COLORS["volume_up"] if close >= open_
                else self.COLORS["volume_down"]
                for close, open_ in zip(df['Close'], df['Open'])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=df.index, y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=4, col=1
            )
            
            # Volume SMA
            if 'volume_sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['volume_sma_20'],
                        mode='lines', name='Vol SMA',
                        line=dict(color='yellow', width=1)
                    ),
                    row=4, col=1
                )
        
        # Layout
        fig.update_layout(
            height=height,
            template=self.template,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)"
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        for i in range(1, 5):
            fig.update_xaxes(showgrid=True, gridcolor=self.COLORS["grid"], row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor=self.COLORS["grid"], row=i, col=1)
        
        return fig
    
    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix"
    ) -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            correlation_matrix: Pandas correlation matrix
            title: Chart title
        
        Returns:
            Plotly heatmap figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=500,
            margin=dict(l=100, r=50, t=80, b=100)
        )
        
        return fig
    
    def create_fear_greed_gauge(
        self,
        value: int,
        title: str = "Fear & Greed Index"
    ) -> go.Figure:
        """
        Create a gauge chart for Fear & Greed Index.
        
        Args:
            value: Current F&G value (0-100)
            title: Chart title
        
        Returns:
            Plotly gauge figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#FF5252'},
                    {'range': [25, 45], 'color': '#FF9800'},
                    {'range': [45, 55], 'color': '#FFC107'},
                    {'range': [55, 75], 'color': '#8BC34A'},
                    {'range': [75, 100], 'color': '#4CAF50'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template=self.template,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        
        return fig
    
    def create_performance_comparison(
        self,
        data: Dict[str, pd.Series],
        title: str = "Performance Comparison"
    ) -> go.Figure:
        """
        Create a line chart comparing multiple assets' performance.
        
        Args:
            data: Dict of {ticker: normalized_price_series}
            title: Chart title
        
        Returns:
            Plotly line chart
        """
        fig = go.Figure()
        
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
        
        for i, (ticker, series) in enumerate(data.items()):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode='lines',
                    name=ticker,
                    line=dict(color=color, width=2)
                )
            )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        
        fig.update_layout(
            title=title,
            yaxis_title="Return (%)",
            template=self.template,
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
