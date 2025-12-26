"""
RAG Chain for Financial Analysis
Combines retrieval with LLM generation for context-aware insights
"""

import os
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.vector_store import FinanceVectorStore
from src.utils import get_logger, get_settings

logger = get_logger(__name__)


class FinanceRAGChain:
    """
    RAG chain for financial analysis.
    
    Combines:
    - Vector store retrieval for context
    - LLM generation for insights
    - Structured prompts for analysis
    """
    
    ANALYSIS_PROMPT = """You are Alex, a friendly and experienced senior financial advisor at a top investment firm.
You have 15+ years of experience analyzing markets and helping clients make informed decisions.

Your personality:
- Warm, approachable, and conversational - like chatting with a knowledgeable friend
- You explain complex concepts in simple terms
- You're honest about uncertainty and never pretend to know everything
- You inject personality into responses while staying professional
- You adapt your response LENGTH and STYLE to match the question

IMPORTANT RULES:
1. For casual greetings (hi, hello, how are you): Respond naturally as a person would. Keep it brief and friendly.
2. For simple questions: Give direct, concise answers. No need for bullet points or sections.
3. For detailed analysis requests: Provide thorough analysis but in a conversational tone, not rigid bullet-point structures.
4. NEVER use the same rigid format for every response. Adapt to what the user actually needs.

## Available Context
**Recent News & Research:**
{context}

**Current Market Data for reference:**
{market_data}

**Technical Indicators:**
{technical_analysis}

**Market Sentiment:**
{sentiment_data}

## User's Question
{question}

Now respond naturally as Alex the financial advisor. Remember:
- If it's a casual question, be casual back
- If they want analysis, provide it conversationally
- Use "I think", "In my view", "From what I'm seeing" - first person
- Reference specific numbers when relevant
- Always end by asking if they have follow-up questions (unless it's a greeting)"""

    SIMPLE_PROMPT = """You are an expert financial analyst AI assistant.
Answer the following question based on the provided context.

## Context
{context}

## Question
{question}

Provide a clear, informative answer. If the context doesn't contain relevant information, 
say so and provide general knowledge on the topic."""

    SUGGESTIONS_PROMPT = """You are an expert financial analyst AI assistant providing ML-backed suggestions.
Based on the quantitative analysis and historical patterns, provide actionable suggestions.

## Current Market Data
{market_data}

## Technical Signals
{technical_signals}

## ML Suggestions Analysis
**Primary Signal**: {primary_signal} (Confidence: {confidence}%)
**Confidence Breakdown**: {confidence_explanation}

## Similar Historical Periods
{similar_periods}

## Risk Analysis (Monte Carlo Simulation)
{risk_metrics}

## Market Regime
{regime_info}

## Entry/Exit Zones
{entry_exit_zones}

Based on ALL the above data, provide:

1. **Actionable Suggestion**: Clear recommendation with specific conditions
   - What to do: BUY / SELL / HOLD / WAIT
   - Entry price levels (if applicable)
   - Position size recommendation (small/medium/normal)

2. **Confidence Justification**: Why this confidence level? Reference specific factors.

3. **Historical Context**: What happened in similar past situations? Be specific with dates and returns.

4. **Risk Warning**: What are the specific risks? Quantify with the VaR and probability numbers.

5. **Key Levels to Watch**:
   - Entry zones with probabilities
   - Stop-loss levels
   - Take-profit targets

6. **Regime Considerations**: How does the current market regime affect this suggestion?

IMPORTANT: 
- Be honest about uncertainty. If confidence is below 60%, emphasize waiting for confirmation.
- Always include the disclaimer that this is for educational purposes, not financial advice.
- Use the actual numbers from the analysis - don't make up statistics."""

    def __init__(
        self,
        vector_store: Optional[FinanceVectorStore] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3  # Slightly higher for more natural conversation
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Vector store instance for retrieval
            model: LLM model to use
            temperature: LLM temperature setting
        """
        settings = get_settings()
        
        # Initialize LLM
        groq_key = settings.llm.groq_api_key
        if not groq_key:
            groq_key = os.getenv("GROQ_API_KEY")
        
        if not groq_key:
            logger.warning("GROQ_API_KEY not set. LLM calls will fail.")
        
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=groq_key
        )
        
        # Vector store
        self.vector_store = vector_store or FinanceVectorStore()
        
        # Create chains
        self.analysis_chain = self._create_analysis_chain()
        self.simple_chain = self._create_simple_chain()
        self.suggestions_chain = self._create_suggestions_chain()
        
        logger.info(f"Initialized RAG chain with model: {model}")
    
    def _create_analysis_chain(self):
        """Create the full analysis chain"""
        prompt = ChatPromptTemplate.from_template(self.ANALYSIS_PROMPT)
        return prompt | self.llm | StrOutputParser()
    
    def _create_simple_chain(self):
        """Create a simple Q&A chain"""
        prompt = ChatPromptTemplate.from_template(self.SIMPLE_PROMPT)
        return prompt | self.llm | StrOutputParser()
    
    def _create_suggestions_chain(self):
        """Create the ML suggestions chain"""
        prompt = ChatPromptTemplate.from_template(self.SUGGESTIONS_PROMPT)
        return prompt | self.llm | StrOutputParser()
    
    def _format_market_data(self, data: Dict[str, Any]) -> str:
        """Format market data for the prompt"""
        if not data:
            return "No market data available."
        
        lines = []
        for key, value in data.items():
            if value is not None:
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    if "percent" in key.lower() or "change" in key.lower():
                        lines.append(f"- {formatted_key}: {value:.2f}%")
                    elif "price" in key.lower() or "cap" in key.lower():
                        lines.append(f"- {formatted_key}: ${value:,.2f}")
                    else:
                        lines.append(f"- {formatted_key}: {value:.4f}")
                else:
                    lines.append(f"- {formatted_key}: {value}")
        
        return "\n".join(lines) if lines else "No market data available."
    
    def _format_technical_analysis(self, signals: Dict[str, Any]) -> str:
        """Format technical analysis signals for the prompt"""
        if not signals:
            return "No technical analysis available."
        
        lines = []
        
        # Overall signal
        if "overall" in signals:
            overall = signals["overall"]
            lines.append(f"**Overall Signal**: {overall.get('signal', 'N/A')} "
                        f"(Confidence: {overall.get('confidence', 0)}%)")
        
        # Trend
        if "trend" in signals:
            trend = signals["trend"]
            lines.append(f"\n**Trend**:")
            if "overall_trend" in trend:
                lines.append(f"- Direction: {trend['overall_trend']}")
            if "moving_averages" in trend and isinstance(trend["moving_averages"], dict):
                ma = trend["moving_averages"]
                lines.append(f"- Moving Averages: {ma.get('signal', 'N/A')}")
            if "macd" in trend and isinstance(trend["macd"], dict):
                macd = trend["macd"]
                lines.append(f"- MACD: {macd.get('signal', 'N/A')}")
        
        # Momentum
        if "momentum" in signals:
            momentum = signals["momentum"]
            lines.append(f"\n**Momentum**:")
            if "rsi" in momentum:
                rsi = momentum["rsi"]
                lines.append(f"- RSI: {rsi.get('value', 'N/A')} ({rsi.get('signal', 'N/A')})")
            if "stochastic" in momentum:
                stoch = momentum["stochastic"]
                lines.append(f"- Stochastic: {stoch.get('signal', 'N/A')}")
        
        # Volatility
        if "volatility" in signals:
            vol = signals["volatility"]
            lines.append(f"\n**Volatility**:")
            if "volatility" in vol:
                lines.append(f"- Level: {vol['volatility'].get('level', 'N/A')}")
            if "bollinger" in vol:
                bb = vol["bollinger"]
                lines.append(f"- Bollinger: {bb.get('signal', 'N/A')}")
        
        return "\n".join(lines) if lines else "No technical analysis available."
    
    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        """Format sentiment data for the prompt"""
        if not sentiment:
            return "No sentiment data available."
        
        lines = []
        
        # Fear & Greed
        if "fear_greed" in sentiment:
            fg = sentiment["fear_greed"]
            if "value" in fg:
                lines.append(f"- Fear & Greed Index: {fg['value']} ({fg.get('classification', 'N/A')})")
        
        # Reddit
        if "reddit" in sentiment:
            reddit = sentiment["reddit"]
            if "post_count" in reddit:
                lines.append(f"- Reddit Activity: {reddit['post_count']} posts, "
                           f"Avg Score: {reddit.get('avg_score', 0):.0f}")
        
        # Overall
        if "overall_sentiment_score" in sentiment:
            lines.append(f"- Overall Sentiment: {sentiment['overall_sentiment_score']:.0f}/100 "
                        f"({sentiment.get('sentiment_label', 'N/A')})")
        
        return "\n".join(lines) if lines else "No sentiment data available."
    
    def analyze(
        self,
        ticker: str,
        question: str,
        market_data: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        n_context_docs: int = 5
    ) -> str:
        """
        Perform comprehensive analysis with RAG.
        
        Args:
            ticker: Asset ticker symbol
            question: User's question
            market_data: Current market data dict
            technical_signals: Technical analysis signals
            sentiment_data: Sentiment data
            n_context_docs: Number of context documents to retrieve
        
        Returns:
            Analysis response string
        """
        # Retrieve relevant context
        context = self.vector_store.get_context_for_analysis(
            ticker=ticker,
            query=question,
            n_results=n_context_docs
        )
        
        # Format inputs
        formatted_market = self._format_market_data(market_data or {})
        formatted_technical = self._format_technical_analysis(technical_signals or {})
        formatted_sentiment = self._format_sentiment(sentiment_data or {})
        
        # Run the chain
        try:
            response = self.analysis_chain.invoke({
                "context": context,
                "market_data": formatted_market,
                "technical_analysis": formatted_technical,
                "sentiment_data": formatted_sentiment,
                "question": question
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Analysis chain error: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def simple_query(
        self,
        question: str,
        ticker: Optional[str] = None,
        n_context_docs: int = 5
    ) -> str:
        """
        Answer a simple question with RAG.
        
        Args:
            question: User's question
            ticker: Optional ticker for context filtering
            n_context_docs: Number of context documents
        
        Returns:
            Response string
        """
        # Retrieve context
        search_query = f"{ticker} {question}" if ticker else question
        results = self.vector_store.search(
            query=search_query,
            n_results=n_context_docs,
            ticker_filter=ticker
        )
        
        # Format context
        context_parts = []
        for result in results.get('results', []):
            context_parts.append(result['content'])
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Run the chain
        try:
            response = self.simple_chain.invoke({
                "context": context,
                "question": question
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Simple query error: {e}")
            return f"Error generating response: {str(e)}"
    
    def ingest_news_for_ticker(
        self,
        ticker: str,
        news_articles: list
    ) -> int:
        """
        Ingest news articles for a ticker into the vector store.
        
        Args:
            ticker: Ticker symbol
            news_articles: List of news articles
        
        Returns:
            Number of articles added
        """
        return self.vector_store.add_news_articles(news_articles, ticker=ticker)
    
    def analyze_with_suggestions(
        self,
        ticker: str,
        market_data: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
        suggestion_report: Optional[Dict[str, Any]] = None,
        regime_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate LLM analysis incorporating ML suggestions.
        
        Args:
            ticker: Asset ticker symbol
            market_data: Current market data dict
            technical_signals: Technical analysis signals
            suggestion_report: SuggestionReport as dict from SuggestionsEngine
            regime_info: RegimeState as dict from RegimeDetector
        
        Returns:
            Comprehensive analysis response string
        """
        # Format inputs
        formatted_market = self._format_market_data(market_data or {})
        formatted_technical = self._format_technical_analysis(technical_signals or {})
        
        # Format ML suggestions data
        if suggestion_report:
            primary_signal = suggestion_report.get('primary_signal', 'NEUTRAL')
            confidence = suggestion_report.get('confidence', 0)
            confidence_explanation = suggestion_report.get('confidence_explanation', 'N/A')
            
            # Format similar periods
            similar_periods_list = suggestion_report.get('similar_periods', [])
            similar_periods_str = "\n".join([
                f"- {p.get('start_date')}: {p.get('outcome')} ({p.get('return_after_30d', 0):+.1f}% in 30d, similarity: {p.get('similarity_score', 0):.0f}%)"
                for p in similar_periods_list[:3]
            ]) if similar_periods_list else "No similar historical periods found."
            
            # Format risk metrics
            risk = suggestion_report.get('risk_metrics', {})
            risk_metrics_str = f"""- 95% VaR (7-day): {risk.get('var_95', 0):.1f}%
- 99% VaR (7-day): {risk.get('var_99', 0):.1f}%
- Expected Shortfall: {risk.get('expected_shortfall', 0):.1f}%
- Probability of Loss: {risk.get('probability_of_loss', 50):.0f}%
- Best Case (7d): {risk.get('best_case_7d', 0):+.1f}%
- Worst Case (7d): {risk.get('worst_case_7d', 0):+.1f}%
- Median Outcome (7d): {risk.get('median_outcome_7d', 0):+.1f}%"""
            
            # Format entry/exit zones
            entry_zones = suggestion_report.get('entry_zones', [])
            target_zones = suggestion_report.get('target_zones', [])
            stop_zones = suggestion_report.get('stop_loss_zones', [])
            
            zones_str = "**Entry Zones (Support):**\n"
            for z in entry_zones[:2]:
                zones_str += f"- ${z.get('price_low', 0):.2f} - ${z.get('price_high', 0):.2f} ({z.get('probability', 0):.0f}% probability, {z.get('strength', 'N/A')})\n"
            
            zones_str += "\n**Target Zones (Resistance):**\n"
            for z in target_zones[:2]:
                zones_str += f"- ${z.get('price_low', 0):.2f} - ${z.get('price_high', 0):.2f} ({z.get('probability', 0):.0f}% probability)\n"
            
            zones_str += "\n**Stop Loss Zones:**\n"
            for z in stop_zones[:2]:
                zones_str += f"- ${z.get('price_low', 0):.2f} - ${z.get('price_high', 0):.2f}\n"
        else:
            primary_signal = "NEUTRAL"
            confidence = 0
            confidence_explanation = "No ML analysis available"
            similar_periods_str = "No data"
            risk_metrics_str = "No risk analysis available"
            zones_str = "No zones calculated"
        
        # Format regime info
        if regime_info:
            regime_str = f"""Current Regime: {regime_info.get('regime', 'UNKNOWN')} (Confidence: {regime_info.get('confidence', 0):.0f}%)
Duration: {regime_info.get('duration_days', 0)} days
Probability of regime change: {regime_info.get('probability_of_change', 50):.0f}%
Most likely next regime: {regime_info.get('most_likely_next_regime', 'Unknown')}"""
        else:
            regime_str = "No regime analysis available"
        
        # Run the suggestions chain
        try:
            response = self.suggestions_chain.invoke({
                "market_data": formatted_market,
                "technical_signals": formatted_technical,
                "primary_signal": primary_signal,
                "confidence": confidence,
                "confidence_explanation": confidence_explanation,
                "similar_periods": similar_periods_str,
                "risk_metrics": risk_metrics_str,
                "regime_info": regime_str,
                "entry_exit_zones": zones_str
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Suggestions analysis error: {e}")
            return f"Error generating suggestions analysis: {str(e)}"
