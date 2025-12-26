"""
Suggestions Visualization
Premium charts for displaying ML suggestions and analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.utils import get_logger

logger = get_logger(__name__)


class SuggestionsChartGenerator:
    """
    Generate premium visualizations for suggestions and predictions.
    
    Charts:
    - Entry/Exit Zone Chart
    - Confidence Gauge
    - Similar Periods Comparison
    - Risk Dashboard
    - Prediction Cone
    - Regime Timeline
    """
    
    # Premium color palette - Light Theme
    COLORS = {
        'bullish': '#059669',        # Professional green
        'bearish': '#DC2626',        # Professional red
        'neutral': '#D97706',        # Amber
        'support': 'rgba(5, 150, 105, 0.2)',
        'resistance': 'rgba(220, 38, 38, 0.2)',
        'background': '#FFFFFF',
        'card': '#F9FAFB',
        'text': '#111827',
        'text_muted': '#6B7280',
        'grid': '#E5E7EB',
        'accent': '#2563EB',         # Blue accent
        'warning': '#F59E0B',        # Amber warning
        'prediction_50': 'rgba(37, 99, 235, 0.25)',
        'prediction_80': 'rgba(37, 99, 235, 0.12)',
        'prediction_95': 'rgba(37, 99, 235, 0.05)',
    }
    
    def __init__(self, dark_mode: bool = True):
        """Initialize the chart generator."""
        self.dark_mode = dark_mode
        self.template = 'plotly_dark' if dark_mode else 'plotly_white'
        logger.info("Initialized SuggestionsChartGenerator")
    
    def create_entry_exit_chart(
        self,
        df: pd.DataFrame,
        entry_zones: List[Dict[str, Any]],
        exit_zones: List[Dict[str, Any]],
        stop_loss_zones: Optional[List[Dict[str, Any]]] = None,
        ticker: str = ""
    ) -> go.Figure:
        """
        Create candlestick chart with probability-weighted entry/exit zones.
        
        Args:
            df: OHLCV DataFrame
            entry_zones: List of entry zone dicts with price_low, price_high, probability
            exit_zones: List of exit zone dicts
            stop_loss_zones: Optional stop loss zones
            ticker: Ticker symbol for title
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color=self.COLORS['bullish'],
                decreasing_line_color=self.COLORS['bearish']
            ),
            row=1, col=1
        )
        
        # Add entry zones (support)
        for zone in entry_zones:
            opacity = zone.get('probability', 50) / 100 * 0.5
            fig.add_hrect(
                y0=zone['price_low'],
                y1=zone['price_high'],
                fillcolor=f"rgba(0, 200, 83, {opacity})",
                line=dict(color=self.COLORS['bullish'], width=1),
                annotation_text=f"Entry Zone ({zone.get('probability', 50):.0f}%)",
                annotation_position="right",
                row=1, col=1
            )
        
        # Add exit zones (resistance)
        for zone in exit_zones:
            opacity = zone.get('probability', 50) / 100 * 0.5
            fig.add_hrect(
                y0=zone['price_low'],
                y1=zone['price_high'],
                fillcolor=f"rgba(255, 23, 68, {opacity})",
                line=dict(color=self.COLORS['bearish'], width=1),
                annotation_text=f"Target ({zone.get('probability', 50):.0f}%)",
                annotation_position="right",
                row=1, col=1
            )
        
        # Add stop loss zones
        if stop_loss_zones:
            for zone in stop_loss_zones:
                fig.add_hrect(
                    y0=zone['price_low'],
                    y1=zone['price_high'],
                    fillcolor="rgba(255, 145, 0, 0.3)",
                    line=dict(color=self.COLORS['warning'], width=2, dash='dash'),
                    annotation_text="Stop Loss",
                    annotation_position="right",
                    row=1, col=1
                )
        
        # Volume
        colors = [self.COLORS['bullish'] if c >= o else self.COLORS['bearish'] 
                 for c, o in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                opacity=0.5,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Layout
        fig.update_layout(
            title=f"{ticker} - Entry/Exit Zones Analysis",
            template=self.template,
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'])
        )
        
        fig.update_xaxes(gridcolor=self.COLORS['grid'])
        fig.update_yaxes(gridcolor=self.COLORS['grid'])
        
        return fig
    
    def create_confidence_gauge(
        self,
        confidence: float,
        signal: str,
        factors: Optional[List[Dict[str, Any]]] = None
    ) -> go.Figure:
        """
        Create a radial gauge showing confidence level.
        
        Args:
            confidence: Confidence score 0-100
            signal: BULLISH, BEARISH, or NEUTRAL
            factors: Optional list of contributing factors
        
        Returns:
            Plotly Figure
        """
        # Determine color based on signal
        if signal == "BULLISH":
            color = self.COLORS['bullish']
        elif signal == "BEARISH":
            color = self.COLORS['bearish']
        else:
            color = self.COLORS['neutral']
        
        fig = go.Figure()
        
        # Main gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            domain={'x': [0, 1], 'y': [0.3, 1]},
            title={'text': f"<b>{signal}</b><br>Signal Confidence", 'font': {'size': 20}},
            number={'suffix': '%', 'font': {'size': 50}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': self.COLORS['text_muted']},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': self.COLORS['card'],
                'borderwidth': 2,
                'bordercolor': self.COLORS['grid'],
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 23, 68, 0.2)'},
                    {'range': [40, 60], 'color': 'rgba(255, 214, 0, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(0, 200, 83, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': self.COLORS['accent'], 'width': 4},
                    'thickness': 0.75,
                    'value': confidence
                }
            }
        ))
        
        # Add factor bars if provided
        if factors:
            annotations = []
            for i, factor in enumerate(factors[:4]):  # Max 4 factors
                factor_name = factor.get('name', f'Factor {i+1}')
                factor_score = factor.get('score', 50)
                factor_signal = factor.get('signal', 'NEUTRAL')
                
                annotations.append(dict(
                    x=0.1 + i * 0.225,
                    y=0.15,
                    text=f"<b>{factor_name}</b><br>{factor_signal}: {factor_score:.0f}%",
                    showarrow=False,
                    font=dict(size=10, color=self.COLORS['text_muted']),
                    xanchor='center'
                ))
            
            fig.update_layout(annotations=annotations)
        
        fig.update_layout(
            template=self.template,
            height=350,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'])
        )
        
        return fig
    
    def create_similar_periods_chart(
        self,
        current_pattern: pd.Series,
        similar_periods: List[Dict[str, Any]],
        ticker: str = ""
    ) -> go.Figure:
        """
        Create overlay chart comparing current pattern to historical matches.
        
        Args:
            current_pattern: Current price series (normalized)
            similar_periods: List of similar period data
            ticker: Ticker symbol
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Normalize current pattern
        if len(current_pattern) > 0:
            normalized_current = (current_pattern / current_pattern.iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=list(range(len(normalized_current))),
                y=normalized_current,
                mode='lines',
                name='Current Pattern',
                line=dict(color=self.COLORS['accent'], width=3)
            ))
        
        # Add historical patterns
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#F38181']
        
        for i, period in enumerate(similar_periods[:5]):
            similarity = period.get('similarity_score', 0)
            outcome = period.get('outcome', 'N/A')
            return_30d = period.get('return_after_30d', 0)
            
            # Create synthetic normalized pattern based on return
            pattern_length = len(current_pattern) if len(current_pattern) > 0 else 20
            # Use return to simulate historical pattern + future
            future_days = 10
            total_days = pattern_length + future_days
            
            # Create a smooth path that ends at the return_30d value
            x_vals = list(range(total_days))
            # Simple linear interpolation for demo
            pattern_values = list(np.linspace(0, return_30d/3, pattern_length)) + \
                           list(np.linspace(return_30d/3, return_30d, future_days))
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=pattern_values,
                mode='lines',
                name=f"{period.get('start_date', 'Historical')} ({similarity:.0f}%)",
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                opacity=0.7
            ))
        
        # Add vertical line at current point
        if len(current_pattern) > 0:
            fig.add_vline(
                x=len(current_pattern) - 1,
                line=dict(color=self.COLORS['text_muted'], width=2, dash='dash'),
                annotation_text="Today"
            )
        
        fig.update_layout(
            title=f"{ticker} - Similar Historical Patterns",
            xaxis_title="Days",
            yaxis_title="Return (%)",
            template=self.template,
            height=400,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor=self.COLORS['grid'])
        fig.update_yaxes(gridcolor=self.COLORS['grid'])
        
        return fig
    
    def create_prediction_cone(
        self,
        df: pd.DataFrame,
        prediction_result: Dict[str, Any],
        ticker: str = ""
    ) -> go.Figure:
        """
        Create price chart with prediction cone showing confidence bands.
        
        Args:
            df: Historical OHLCV data
            prediction_result: PredictionResult as dict
            ticker: Ticker symbol
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Historical price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color=self.COLORS['text'], width=2)
        ))
        
        # Create future dates
        if hasattr(df.index[-1], 'strftime'):
            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_result.get('horizon_days', 7)
            )
        else:
            future_dates = list(range(len(df), len(df) + prediction_result.get('horizon_days', 7)))
        
        median = prediction_result.get('median_prediction', [])
        bands = prediction_result.get('confidence_bands', {})
        
        if median and bands:
            # 95% band (widest)
            if '95' in bands:
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=bands['95']['upper'] + bands['95']['lower'][::-1],
                    fill='toself',
                    fillcolor=self.COLORS['prediction_95'],
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% Confidence',
                    showlegend=True
                ))
            
            # 80% band
            if '80' in bands:
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=bands['80']['upper'] + bands['80']['lower'][::-1],
                    fill='toself',
                    fillcolor=self.COLORS['prediction_80'],
                    line=dict(color='rgba(0,0,0,0)'),
                    name='80% Confidence'
                ))
            
            # 50% band
            if '50' in bands:
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=bands['50']['upper'] + bands['50']['lower'][::-1],
                    fill='toself',
                    fillcolor=self.COLORS['prediction_50'],
                    line=dict(color='rgba(0,0,0,0)'),
                    name='50% Confidence'
                ))
            
            # Median prediction line
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=median,
                mode='lines',
                name='Median Prediction',
                line=dict(color=self.COLORS['accent'], width=3)
            ))
        
        # Quality indicator
        quality = prediction_result.get('prediction_quality', 'UNKNOWN')
        quality_color = {
            'HIGH': self.COLORS['bullish'],
            'MEDIUM': self.COLORS['neutral'],
            'LOW': self.COLORS['bearish']
        }.get(quality, self.COLORS['text_muted'])
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Prediction Quality: {quality}</b>",
            showarrow=False,
            font=dict(size=14, color=quality_color),
            bgcolor=self.COLORS['card'],
            bordercolor=quality_color,
            borderwidth=2,
            borderpad=4
        )
        
        fig.update_layout(
            title=f"{ticker} - Price Prediction with Uncertainty Bands",
            xaxis_title="Date",
            yaxis_title="Price",
            template=self.template,
            height=500,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(gridcolor=self.COLORS['grid'])
        fig.update_yaxes(gridcolor=self.COLORS['grid'])
        
        return fig
    
    def create_risk_dashboard(
        self,
        risk_metrics: Dict[str, Any]
    ) -> go.Figure:
        """
        Create risk metrics visualization dashboard.
        
        Args:
            risk_metrics: Dict with VaR, Expected Shortfall, etc.
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk', 'Expected Outcomes', 'Loss Probability', 'Risk Indicators'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # VaR Gauge
        var_95 = risk_metrics.get('var_95', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=abs(var_95),
                title={'text': "95% VaR (%)"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': self.COLORS['warning']},
                    'steps': [
                        {'range': [0, 5], 'color': 'rgba(0, 200, 83, 0.3)'},
                        {'range': [5, 10], 'color': 'rgba(255, 214, 0, 0.3)'},
                        {'range': [10, 20], 'color': 'rgba(255, 23, 68, 0.3)'}
                    ]
                }
            ),
            row=1, col=1
        )
        
        # Outcome distribution bars
        outcomes = ['Worst', 'Median', 'Best']
        values = [
            risk_metrics.get('worst_case_7d', 0),
            risk_metrics.get('median_outcome_7d', 0),
            risk_metrics.get('best_case_7d', 0)
        ]
        colors = [
            self.COLORS['bearish'],
            self.COLORS['neutral'],
            self.COLORS['bullish']
        ]
        
        fig.add_trace(
            go.Bar(
                x=outcomes,
                y=values,
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Loss probability gauge
        loss_prob = risk_metrics.get('probability_of_loss', 50)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=loss_prob,
                title={'text': "Loss Probability (%)"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self.COLORS['bearish'] if loss_prob > 50 else self.COLORS['bullish']},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 200, 83, 0.3)'},
                        {'range': [30, 50], 'color': 'rgba(255, 214, 0, 0.3)'},
                        {'range': [50, 100], 'color': 'rgba(255, 23, 68, 0.3)'}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Risk indicators
        indicators = ['VaR 99%', 'Exp. Shortfall', 'Max DD']
        indicator_values = [
            abs(risk_metrics.get('var_99', 0)),
            abs(risk_metrics.get('expected_shortfall', 0)),
            abs(risk_metrics.get('max_drawdown_expected', 0))
        ]
        
        fig.add_trace(
            go.Bar(
                x=indicators,
                y=indicator_values,
                marker_color=self.COLORS['warning'],
                text=[f"{v:.1f}%" for v in indicator_values],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Risk Analysis Dashboard",
            showlegend=False,
            template=self.template,
            height=500,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'])
        )
        
        return fig
    
    def create_regime_timeline(
        self,
        regime_history: List[Dict[str, Any]],
        current_regime: str
    ) -> go.Figure:
        """
        Create a timeline showing regime changes.
        
        Args:
            regime_history: List of regime periods
            current_regime: Current regime
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Color mapping for regimes
        regime_colors = {
            'BULL_TREND': self.COLORS['bullish'],
            'BEAR_TREND': self.COLORS['bearish'],
            'HIGH_VOLATILITY': self.COLORS['warning'],
            'LOW_VOLATILITY': self.COLORS['accent'],
            'CRASH': '#FF0000',
            'UNKNOWN': self.COLORS['text_muted']
        }
        
        # Create timeline bars
        for i, period in enumerate(regime_history):
            regime = period.get('regime', 'UNKNOWN')
            start = period.get('start_date', i)
            end = period.get('end_date', i + 1)
            duration = period.get('duration', 0)
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[0],
                orientation='h',
                name=regime,
                marker_color=regime_colors.get(regime, self.COLORS['text_muted']),
                text=f"{regime}<br>{start} - {end}",
                textposition='inside',
                showlegend=True if i == 0 else False
            ))
        
        # Alternative: Use shapes for cleaner timeline
        fig = go.Figure()
        
        y_position = 0.5
        for i, period in enumerate(regime_history):
            regime = period.get('regime', 'UNKNOWN')
            duration = period.get('duration', 30)
            start_idx = sum(p.get('duration', 30) for p in regime_history[:i])
            end_idx = start_idx + duration
            
            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=0,
                y1=1,
                fillcolor=regime_colors.get(regime, self.COLORS['text_muted']),
                opacity=0.7,
                line=dict(width=1, color=self.COLORS['text'])
            )
            
            # Add label
            fig.add_annotation(
                x=(start_idx + end_idx) / 2,
                y=0.5,
                text=f"<b>{regime}</b><br>{duration}d",
                showarrow=False,
                font=dict(size=10, color='white')
            )
        
        # Current regime indicator
        fig.add_annotation(
            x=0.98,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"<b>Current: {current_regime}</b>",
            showarrow=False,
            font=dict(size=14, color=regime_colors.get(current_regime, self.COLORS['text'])),
            bgcolor=self.COLORS['card'],
            bordercolor=regime_colors.get(current_regime, self.COLORS['text']),
            borderwidth=2,
            borderpad=4
        )
        
        fig.update_layout(
            title="Market Regime Timeline",
            xaxis_title="Days",
            yaxis_visible=False,
            template=self.template,
            height=200,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text']),
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor=self.COLORS['grid'])
        
        return fig
    
    def create_correlation_heatmap(
        self,
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Create correlation heatmap for portfolio.
        
        Args:
            correlation_matrix: Correlation matrix as nested dict
        
        Returns:
            Plotly Figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(correlation_matrix)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale=[
                [0, self.COLORS['bearish']],
                [0.5, self.COLORS['text_muted']],
                [1, self.COLORS['bullish']]
            ],
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in df.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Correlation between %{x} and %{y}: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Asset Correlation Heatmap",
            template=self.template,
            height=400,
            paper_bgcolor=self.COLORS['background'],
            plot_bgcolor=self.COLORS['background'],
            font=dict(color=self.COLORS['text'])
        )
        
        return fig
