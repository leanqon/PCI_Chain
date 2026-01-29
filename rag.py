#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Module for PCIChain
Hybrid retrieval using Vector Embeddings (SentenceTransformers + FAISS) + BM25.
"""

import os
import re
import pickle
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HybridKnowledgeRetriever:
    """
    Hybrid knowledge retrieval combining:
    1. Dense retrieval via SentenceTransformers + FAISS
    2. Sparse retrieval via BM25
    3. Reciprocal Rank Fusion (RRF) for merging results
    """

    def __init__(self, knowledge_dir: Optional[str] = None, use_cache: bool = True):
        """
        Initialize the retriever with a knowledge directory.

        Args:
            knowledge_dir: Path to directory containing knowledge files.
                          Defaults to 'pci_chain/knowledge' relative to this file.
            use_cache: Whether to use cached embeddings if available.
        """
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent / "knowledge"
        
        self.knowledge_dir = Path(knowledge_dir)
        self.cache_dir = self.knowledge_dir / ".cache"
        self.use_cache = use_cache
        
        # Document storage
        self.documents: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        
        # Models and indices (lazy loaded)
        self._embedding_model = None
        self._faiss_index = None
        self._bm25 = None
        self._embeddings = None
        
        # Load documents
        self._load_knowledge()
        
        # Build indices
        self._build_indices()

    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[RAG] Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
                self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("[RAG] Embedding model loaded successfully")
            except ImportError:
                print("[RAG] Warning: sentence-transformers not installed. Using keyword fallback.")
                self._embedding_model = None
            except Exception as e:
                print(f"[RAG] Error loading embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model

    def _load_knowledge(self) -> None:
        """Load all .txt, .md, and .pdf files from the knowledge directory."""
        if not self.knowledge_dir.exists():
            print(f"[RAG] Warning: Knowledge directory not found: {self.knowledge_dir}")
            return

        # Load text files
        for file_path in self.knowledge_dir.glob("*.txt"):
            self._load_file(file_path)
        for file_path in self.knowledge_dir.glob("*.md"):
            self._load_file(file_path)
        
        # Load PDF files
        for file_path in self.knowledge_dir.glob("*.pdf"):
            self._load_pdf(file_path)

        # Extract text list for BM25
        self.doc_texts = [doc["content"] for doc in self.documents]
        
        print(f"[RAG] Loaded {len(self.documents)} document chunks from {self.knowledge_dir}")

    def _load_file(self, file_path: Path) -> None:
        """Load and chunk a single text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = re.split(r'\n\s*\n', content)

            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) > 20:
                    self.documents.append({
                        "source": file_path.name,
                        "chunk_id": i,
                        "content": chunk,
                    })
        except Exception as e:
            print(f"[RAG] Error loading {file_path}: {e}")

    def _load_pdf(self, file_path: Path) -> None:
        """Load and chunk a PDF file using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            print(f"[RAG] Warning: pdfplumber not installed. Cannot load {file_path}")
            return
        
        try:
            import sys
            import io
            
            # Suppress pdfplumber stderr warnings (e.g., "Cannot set gray stroke color")
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                with pdfplumber.open(file_path) as pdf:
                    all_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            all_text.append(text)
                    
                    full_text = "\n\n".join(all_text)
            finally:
                # Restore stderr
                sys.stderr = old_stderr
            
            chunks = re.split(r'\n\s*\n', full_text)
            
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if len(chunk) > 50:
                    letters = sum(1 for c in chunk if c.isalpha())
                    digits = sum(1 for c in chunk if c.isdigit())
                    if letters > digits * 0.5 or letters > 50:
                        self.documents.append({
                            "source": file_path.name,
                            "chunk_id": i,
                            "content": chunk[:2000],
                        })
        except Exception as e:
            print(f"[RAG] Error loading PDF {file_path}: {e}")


    def _build_indices(self) -> None:
        """Build FAISS and BM25 indices."""
        if not self.documents:
            return
        
        # Try to load from cache
        if self.use_cache and self._load_cache():
            return
        
        # Build BM25 index
        self._build_bm25_index()
        
        # Build FAISS index (if embedding model available)
        self._build_faiss_index()
        
        # Save to cache
        if self.use_cache:
            self._save_cache()

    def _build_bm25_index(self) -> None:
        """Build BM25 index for sparse retrieval."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents (simple whitespace + Chinese character tokenization)
            tokenized_docs = [self._tokenize(doc) for doc in self.doc_texts]
            self._bm25 = BM25Okapi(tokenized_docs)
            print(f"[RAG] BM25 index built with {len(tokenized_docs)} documents")
        except ImportError:
            print("[RAG] Warning: rank_bm25 not installed. BM25 search disabled.")
            self._bm25 = None
        except Exception as e:
            print(f"[RAG] Error building BM25 index: {e}")
            self._bm25 = None

    def _build_faiss_index(self) -> None:
        """Build FAISS index for dense retrieval."""
        if self.embedding_model is None:
            return
        
        try:
            import faiss
            import numpy as np
            
            print(f"[RAG] Embedding {len(self.doc_texts)} document chunks...")
            self._embeddings = self.embedding_model.encode(
                self.doc_texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self._embeddings)
            
            # Build index
            dim = self._embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            self._faiss_index.add(self._embeddings)
            
            print(f"[RAG] FAISS index built: {dim}-dim, {self._faiss_index.ntotal} vectors")
        except ImportError:
            print("[RAG] Warning: faiss not installed. Vector search disabled.")
            self._faiss_index = None
        except Exception as e:
            print(f"[RAG] Error building FAISS index: {e}")
            self._faiss_index = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25: split on whitespace and Chinese characters."""
        # Split on whitespace
        tokens = text.lower().split()
        
        # Also split Chinese text into characters (simple approach)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        # Add 2-gram Chinese
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i+1])
        
        return tokens

    def _load_cache(self) -> bool:
        """Load cached embeddings and indices."""
        cache_file = self.cache_dir / "rag_cache.pkl"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            
            # Verify cache matches current documents
            if cache.get('doc_count') != len(self.documents):
                print("[RAG] Cache outdated, rebuilding indices...")
                return False
            
            self._embeddings = cache.get('embeddings')
            self._bm25 = cache.get('bm25')
            
            # Rebuild FAISS index from embeddings
            if self._embeddings is not None:
                import faiss
                import numpy as np
                dim = self._embeddings.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_index.add(self._embeddings)
            
            print(f"[RAG] Loaded indices from cache ({len(self.documents)} docs)")
            return True
        except Exception as e:
            print(f"[RAG] Error loading cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save embeddings and BM25 index to cache."""
        try:
            self.cache_dir.mkdir(exist_ok=True)
            cache_file = self.cache_dir / "rag_cache.pkl"
            
            cache = {
                'doc_count': len(self.documents),
                'embeddings': self._embeddings,
                'bm25': self._bm25,
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            
            print(f"[RAG] Saved indices to cache")
        except Exception as e:
            print(f"[RAG] Error saving cache: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 retrieval with RRF.

        Args:
            query: Search query
            top_k: Maximum number of results to return

        Returns:
            List of relevant document chunks with scores
        """
        if not self.documents:
            return []
        
        # Get results from both methods
        vector_results = self._vector_search(query, k=10)
        bm25_results = self._bm25_search(query, k=10)
        
        # If only one method available, return its results
        if not vector_results and not bm25_results:
            return self._keyword_search(query, top_k)  # Fallback
        if not vector_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return vector_results[:top_k]
        
        # Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        k_param = 60  # RRF parameter
        
        for rank, (idx, score) in enumerate(vector_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_param + rank)
        
        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_param + rank)
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        
        # Build results
        results = []
        for idx, rrf_score in sorted_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "source": doc["source"],
                "score": rrf_score,
                "chunk_id": doc["chunk_id"],
            })
        
        return results

    def _vector_search(self, query: str, k: int = 10) -> List[tuple]:
        """Vector similarity search using FAISS."""
        if self._faiss_index is None or self.embedding_model is None:
            return []
        
        try:
            import numpy as np
            
            # Encode query
            query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
            import faiss
            faiss.normalize_L2(query_emb)
            
            # Search
            scores, indices = self._faiss_index.search(query_emb, k)
            
            return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        except Exception as e:
            print(f"[RAG] Vector search error: {e}")
            return []

    def _bm25_search(self, query: str, k: int = 10) -> List[tuple]:
        """BM25 sparse search."""
        if self._bm25 is None:
            return []
        
        try:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            
            return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        except Exception as e:
            print(f"[RAG] BM25 search error: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        medical_terms = [
            "阿司匹林", "氯吡格雷", "替格瑞洛", "阿托伐他汀", "他汀",
            "ACEI", "ARB", "β受体阻滞剂", "美托洛尔",
            "心肌梗死", "STEMI", "NSTEMI", "心衰", "PCI", "DAPT",
        ]
        
        query_lower = query.lower()
        query_terms = [t for t in medical_terms if t.lower() in query_lower or t in query]
        
        scored_docs = []
        for i, doc in enumerate(self.documents):
            score = sum(1 for t in query_terms if t in doc["content"])
            if score > 0:
                scored_docs.append((i, score))
        
        scored_docs.sort(key=lambda x: -x[1])
        
        return [
            {"content": self.documents[idx]["content"], 
             "source": self.documents[idx]["source"],
             "score": score,
             "chunk_id": self.documents[idx]["chunk_id"]}
            for idx, score in scored_docs[:top_k]
        ]

    def retrieve_for_medication(self, diagnosis: str, comorbidities: List[str] = None, language: str = "zh") -> str:
        """
        Convenience method for medication recommendation.
        Combines diagnosis and comorbidities into a search query.

        Args:
            diagnosis: Primary diagnosis
            comorbidities: List of comorbidities
            language: Output language ("zh" for Chinese, "en" for English)

        Returns:
            Formatted string of relevant guidelines for injection into prompt
        """
        query_parts = [diagnosis]
        if comorbidities:
            query_parts.extend(comorbidities)

        query = " ".join(query_parts)
        results = self.search(query, top_k=3)

        if not results:
            return ""

        # Format results for prompt injection based on language
        if language == "en":
            guidelines_text = "## Relevant Clinical Guidelines\n\n"
        else:
            guidelines_text = "## 相关临床指南参考\n\n"
            
        for i, result in enumerate(results, 1):
            guidelines_text += f"**[{result['source']}]**\n"
            guidelines_text += f"{result['content'][:1000]}\n\n"

        return guidelines_text


# Singleton instance for easy import
_retriever_instance: Optional[HybridKnowledgeRetriever] = None


def get_retriever() -> HybridKnowledgeRetriever:
    """Get or create the singleton HybridKnowledgeRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridKnowledgeRetriever()
    return _retriever_instance


# Backward compatibility alias
KnowledgeRetriever = HybridKnowledgeRetriever
