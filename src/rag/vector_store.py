"""
RAG (Retrieval-Augmented Generation) Pipeline
Vector store and retrieval system for financial documents and news
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.utils import get_logger, get_settings

logger = get_logger(__name__)


class FinanceVectorStore:
    """
    Vector store for financial documents using ChromaDB.
    
    Features:
    - Store and retrieve news articles, reports, and analysis
    - Semantic search with embedding similarity
    - Metadata filtering by ticker, date, source
    - Automatic document deduplication
    """
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "finance_docs"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        settings = get_settings()
        self.persist_dir = persist_directory or settings.storage.vector_db_path
        
        # Ensure directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB with PERSISTENT client (data saved to disk)
        logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized vector store with {self.collection.count()} documents")
    
    def _generate_id(self, content: str, source: str) -> str:
        """Generate unique ID for a document"""
        hash_input = f"{content[:500]}{source}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'content', 'source', 'date', 'ticker' keys
            batch_size: Number of documents per batch
        
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            texts = []
            metadatas = []
            ids = []
            
            for doc in batch:
                content = doc.get("content", "")
                if not content:
                    continue
                
                doc_id = self._generate_id(content, doc.get("source", ""))
                
                # Skip if already exists
                existing = self.collection.get(ids=[doc_id])
                if existing and existing['ids']:
                    continue
                
                texts.append(content)
                metadatas.append({
                    "source": doc.get("source", "unknown"),
                    "date": doc.get("date", datetime.now().isoformat()),
                    "ticker": doc.get("ticker", ""),
                    "title": doc.get("title", "")[:200],
                    "url": doc.get("url", "")
                })
                ids.append(doc_id)
            
            if texts:
                # Generate embeddings
                embeddings = self.embedder.encode(texts).tolist()
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                added += len(texts)
                logger.info(f"Added {len(texts)} documents to vector store")
        
        return added
    
    def add_news_articles(self, articles: List[Dict[str, Any]], ticker: str = "") -> int:
        """
        Add news articles to the vector store.
        
        Args:
            articles: List of news articles from fetchers
            ticker: Associated ticker symbol
        
        Returns:
            Number of articles added
        """
        documents = []
        
        for article in articles:
            content = f"{article.get('title', '')}\n\n{article.get('description', '')}"
            
            documents.append({
                "content": content,
                "source": article.get("source", "news"),
                "date": article.get("published_at", datetime.now().isoformat()),
                "ticker": ticker,
                "title": article.get("title", ""),
                "url": article.get("url", "")
            })
        
        return self.add_documents(documents)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        ticker_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        date_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Semantic search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            ticker_filter: Filter by ticker symbol
            source_filter: Filter by source
            date_from: Filter by date (ISO format)
        
        Returns:
            Dict with documents, distances, and metadatas
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Build where filter
        where_filter = None
        where_conditions = []
        
        if ticker_filter:
            where_conditions.append({"ticker": ticker_filter})
        if source_filter:
            where_conditions.append({"source": source_filter})
        
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}
        
        # Query the collection
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    processed_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "relevance_score": 1 - results['distances'][0][i] if results['distances'] else 0
                    })
            
            return {
                "query": query,
                "results": processed_results,
                "total_results": len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"query": query, "results": [], "total_results": 0, "error": str(e)}
    
    def get_context_for_analysis(
        self,
        ticker: str,
        query: str,
        n_results: int = 5
    ) -> str:
        """
        Get formatted context for LLM analysis.
        
        Args:
            ticker: Ticker symbol
            query: Analysis query
            n_results: Number of documents to include
        
        Returns:
            Formatted context string
        """
        search_query = f"{ticker} {query}"
        results = self.search(
            query=search_query,
            n_results=n_results,
            ticker_filter=ticker
        )
        
        if not results['results']:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results['results'], 1):
            meta = result['metadata']
            context_parts.append(
                f"[Source {i}: {meta.get('source', 'Unknown')} - {meta.get('date', 'Unknown date')}]\n"
                f"{result['content'][:500]}...\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        # Sample to get source distribution
        sample = self.collection.peek(limit=min(100, count))
        
        sources = {}
        tickers = {}
        
        if sample['metadatas']:
            for meta in sample['metadatas']:
                source = meta.get('source', 'unknown')
                ticker = meta.get('ticker', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                if ticker:
                    tickers[ticker] = tickers.get(ticker, 0) + 1
        
        return {
            "total_documents": count,
            "sources": sources,
            "tickers": tickers,
            "persist_directory": str(self.persist_dir)
        }
    
    def clear(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store")
