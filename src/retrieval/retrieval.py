"""
Text retrieval functions for the Simple Local RAG project
"""
import torch
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple, Union
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import CONFIG
from src.embedding.embedding import EmbeddingManager

class Retriever:
    """
    Retrieves relevant text chunks based on a query
    """
    
    def __init__(self, 
                embedding_manager: EmbeddingManager = None,
                embeddings: torch.Tensor = None,
                chunks: List[Dict[str, Any]] = None):
        """
        Initialize the retriever with embeddings and chunks
        
        Args:
            embedding_manager: EmbeddingManager instance (optional)
            embeddings: Pre-computed embeddings tensor (optional)
            chunks: List of dictionaries containing text chunks (optional)
        """
        self.embedding_manager = embedding_manager
        self.embeddings = embeddings
        self.chunks = chunks
        
        # If embeddings or chunks are not provided, but embedding_manager is,
        # try to get them from the embedding_manager
        if embedding_manager is not None:
            if embeddings is None and embedding_manager.embeddings is not None:
                self.embeddings = embedding_manager.embeddings
            
            if chunks is None and embedding_manager.text_chunks is not None:
                self.chunks = embedding_manager.text_chunks
                
        # Validate that we have required components
        if self.embeddings is None or self.chunks is None:
            raise ValueError("Retriever needs both embeddings and chunks. Either provide them directly "
                            "or provide an EmbeddingManager with loaded embeddings and chunks.")
        
        # Ensure we have the same number of embeddings and chunks
        if len(self.embeddings) != len(self.chunks):
            raise ValueError(f"Number of embeddings ({len(self.embeddings)}) does not match "
                            f"number of chunks ({len(self.chunks)})")
        
        # Initialize reranker if enabled
        self.reranker = None
        if CONFIG.get("use_reranking", False):
            self._initialize_reranker()
            
        # Initialize keyword search if hybrid search is enabled
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        if CONFIG.get("use_hybrid_search", False):
            self._initialize_keyword_search()
    
    def _initialize_reranker(self):
        """Initialize the reranker model"""
        rerank_model_name = CONFIG.get("rerank_model_name", "BAAI/bge-reranker-large")
        print(f"[INFO] Loading reranker model '{rerank_model_name}'")
        self.reranker = CrossEncoder(rerank_model_name, device=CONFIG["device"])
        
    def _initialize_keyword_search(self):
        """Initialize the keyword search using TF-IDF"""
        print("[INFO] Initializing TF-IDF for keyword search")
        # Extract text from chunks
        texts = [chunk["sentence_chunk"] for chunk in self.chunks]
        
        # Create and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.85,
            min_df=2
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
    def _normalize_scores(self, scores, use_sigmoid=True, floor_value=0.1):
        """
        Normalize scores using min-max scaling and optionally sigmoid for better distribution
        
        Args:
            scores: numpy array of scores to normalize
            use_sigmoid: whether to apply sigmoid normalization as well
            floor_value: minimum value to floor scores to
            
        Returns:
            numpy array: Normalized scores
        """
        # Min-max normalization
        def min_max_normalize(s):
            min_val = np.min(s)
            max_val = np.max(s)
            if max_val > min_val:
                return (s - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(s)
        
        # Sigmoid normalization for smoother distribution
        def sigmoid_normalize(s, sharpness=8):
            median = np.median(s)
            centered = (s - median) * sharpness
            return 1 / (1 + np.exp(-centered))
        
        if np.max(scores) <= 0:
            return scores
            
        # Apply normalizations
        scores_norm1 = min_max_normalize(scores)
        
        if use_sigmoid:
            scores_norm2 = sigmoid_normalize(scores)
            # Weight min-max higher for better scaling
            normalized = 0.7 * scores_norm1 + 0.3 * scores_norm2
        else:
            normalized = scores_norm1
            
        # Floor the scores to avoid extremely low values
        if floor_value > 0:
            normalized = np.maximum(normalized, floor_value)
            
        return normalized
    
    def _compute_similarity(self, query_embedding: torch.Tensor, 
                           corpus_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between query embedding and corpus embeddings
        
        Args:
            query_embedding: Query embedding tensor of shape [embedding_dim]
            corpus_embeddings: Corpus embeddings tensor of shape [num_docs, embedding_dim]
            
        Returns:
            torch.Tensor: Similarity scores tensor of shape [num_docs]
        """
        # Properly normalize the embeddings - this is crucial for cosine similarity
        if query_embedding.norm().item() > 0:
            query_embedding = query_embedding / query_embedding.norm(dim=0, keepdim=True)
        
        # Normalize each corpus embedding individually
        norms = corpus_embeddings.norm(dim=1, keepdim=True)
        # Avoid division by zero
        valid_indices = norms.squeeze() > 0
        if valid_indices.sum() > 0:
            corpus_embeddings[valid_indices] = corpus_embeddings[valid_indices] / norms[valid_indices]
        
        # Compute cosine similarity 
        raw_scores = torch.matmul(corpus_embeddings, query_embedding.unsqueeze(1)).squeeze()
        
        # Apply a higher temperature for smoother distribution (higher temp = smoother differences)
        temperature = 0.2  # Increased from 0.1
        softmax_scores = torch.softmax(raw_scores / temperature, dim=0)
        
        # Adjust alpha to give more weight to raw scores
        alpha = 0.85  # Increased from 0.7
        blended_scores = alpha * raw_scores + (1 - alpha) * softmax_scores
        
        return blended_scores
    
    def _keyword_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using TF-IDF
        
        Args:
            query: Query string
            top_k: Number of top chunks to return
            
        Returns:
            List[Dict]: List of dictionaries with chunks and scores
        """
        if top_k is None:
            top_k = CONFIG["num_chunks_to_retrieve"]
            
        # Transform query to TF-IDF
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Compute similarity scores
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        
        # Create results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()  # Copy to avoid modifying original
            chunk["tfidf_score"] = float(tfidf_scores[idx])
            results.append(chunk)
            
        return results
    
    def _expand_query(self, query: str) -> str:
        """
        Expand the query using the embedding model to generate related terms
        
        Args:
            query: Original query string
            
        Returns:
            str: Expanded query
        """
        # Improved query expansion with more domain-specific nutrition terms
        nutrition_terms = {
            "fat": "lipid fatty acid triglyceride saturated unsaturated cholesterol",
            "vitamin": "micronutrient supplement deficiency bioavailability",
            "mineral": "micronutrient electrolyte calcium iron zinc magnesium potassium",
            "protein": "amino acid peptide enzyme collagen albumin",
            "carbohydrate": "sugar starch fiber glucose fructose sucrose maltose polysaccharide",
            "nutrient": "nutrition nourishment macronutrient micronutrient digestion absorption",
            "calorie": "energy kilocalorie kcal joule metabolism",
            "diet": "nutrition food eating meal plan macronutrient balance",
            "metabolism": "metabolic process anabolism catabolism energy expenditure",
            "digestion": "digestive absorption enzyme breakdown hydrolysis",
            "disease": "disorder condition syndrome pathology inflammation",
            "health": "wellness wellbeing fitness condition",
            "obesity": "overweight body mass bmi adipose fat",
            "diabetes": "insulin glucose hyperglycemia blood sugar",
            "heart": "cardiovascular cardiac circulation blood",
            "cancer": "tumor malignant cell growth oncology",
            "liver": "hepatic detoxification bile organ",
            "kidney": "renal filtration excretion nephron",
            "gut": "intestine microbiome flora digestive tract",
            "brain": "neural cognitive central nervous system"
        }
        
        expansion_factor = CONFIG.get("expansion_factor", 1)
        expanded_query = query
        
        # Add nutrition domain terms based on query keywords
        for term, synonyms in nutrition_terms.items():
            if re.search(r'\b' + term + r'\b', query, re.IGNORECASE):
                # Split synonyms and select based on expansion factor
                synonym_list = synonyms.split()
                num_terms_to_add = min(len(synonym_list), round(expansion_factor * 3))
                selected_terms = synonym_list[:num_terms_to_add]
                expanded_query += f" {' '.join(selected_terms)}"
                
        # Look for specific nutritional concepts and add relevant technical terms
        if "vitamin" in query.lower():
            if "d" in query.lower() or "d3" in query.lower():
                expanded_query += " cholecalciferol calcitriol calcidiol"
            elif "b12" in query.lower():
                expanded_query += " cobalamin methylcobalamin cyanocobalamin"
            elif "c" in query.lower():
                expanded_query += " ascorbic ascorbate"
        
        # Look for specific questions about nutrition
        if "how much" in query.lower() or "recommended" in query.lower() or "daily" in query.lower():
            expanded_query += " recommended dietary allowance rda dri intake"
            
        if "deficiency" in query.lower() or "lack" in query.lower():
            expanded_query += " insufficiency inadequate malnutrition symptoms"
            
        return expanded_query
    
    def _rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model
        
        Args:
            query: Query string
            results: List of initial retrieval results
            top_k: Number of top chunks to keep after reranking
            
        Returns:
            List[Dict]: Reranked list of chunks
        """
        if top_k is None:
            top_k = CONFIG.get("rerank_top_k", 5)
            
        if self.reranker is None:
            self._initialize_reranker()
            
        if len(results) <= 1:
            return results
        
        # Prepare passages for reranking
        pairs = [(query, chunk["sentence_chunk"]) for chunk in results]
        
        # Get scores from cross-encoder
        rerank_scores = self.reranker.predict(pairs)
        
        # Print raw scores for debugging
        print(f"[DEBUG] Raw reranker scores: min={min(rerank_scores):.4f}, max={max(rerank_scores):.4f}, avg={sum(rerank_scores)/len(rerank_scores):.4f}")
        
        # Convert rerank_scores to numpy array for easier processing 
        rerank_scores_np = np.array(rerank_scores)
        
        # Normalize rerank scores with the same approach, using a floor of 0.3
        normalized_scores_np = self._normalize_scores(rerank_scores_np, floor_value=0.3)
        
        # Blend original and normalized scores
        blend_factor = 0.8
        blended_scores = blend_factor * rerank_scores_np + (1 - blend_factor) * normalized_scores_np
        
        # Convert back to list
        blended_scores = blended_scores.tolist()
        
        # Track both original and blended scores
        for i, (score, blended) in enumerate(zip(rerank_scores, blended_scores)):
            results[i]["original_rerank_score"] = float(score)
            results[i]["rerank_score"] = float(blended)
        
        # Sort by reranker scores (blended)
        reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Perform diversification if enabled
        if CONFIG.get("ensure_diverse_chunks", False):
            diverse_results = self._diversify_results(reranked_results, top_k)
            return diverse_results
        else:
            # Return top-k results without diversification
            return reranked_results[:top_k]
            
    def _diversify_results(self, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Ensure diversity in the results by avoiding too similar chunks
        
        Args:
            results: List of results to diversify
            top_k: Number of diverse results to return
            
        Returns:
            List[Dict]: List of diversified results
        """
        if top_k is None:
            top_k = CONFIG.get("rerank_top_k", 5)
            
        if len(results) <= 1:
            return results
            
        # Get threshold from config
        diversity_threshold = CONFIG.get("diversity_threshold", 0.85)
        
        # Always keep the first (highest scoring) result
        diverse_results = [results[0]]
        
        # Process the remaining results
        for candidate in results[1:]:
            # Check if this result is too similar to any already selected result
            too_similar = False
            for selected in diverse_results:
                # Simple text similarity using Jaccard index
                similarity = self._compute_text_similarity(
                    candidate["sentence_chunk"], 
                    selected["sentence_chunk"]
                )
                if similarity > diversity_threshold:
                    too_similar = True
                    break
                    
            # Add the result if it's not too similar to any selected result
            if not too_similar:
                diverse_results.append(candidate)
                
            # Stop if we have enough diverse results
            if len(diverse_results) >= top_k:
                break
                
        return diverse_results
        
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two text strings using Jaccard index
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Tokenize into words (simple approach)
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        # Compute Jaccard index: intersection / union
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks for a given query
        
        Args:
            query: Query string
            top_k: Number of top chunks to return (default: from CONFIG)
            
        Returns:
            List[Dict]: List of dictionaries with chunks and scores
        """
        if top_k is None:
            top_k = CONFIG["num_chunks_to_retrieve"]
        
        # Query expansion if enabled
        if CONFIG.get("use_query_expansion", False):
            expanded_query = self._expand_query(query)
            print(f"[INFO] Expanded query: '{expanded_query}'")
        else:
            expanded_query = query
        
        # Hybrid search if enabled
        if CONFIG.get("use_hybrid_search", False) and self.tfidf_vectorizer is not None:
            # Perform semantic search
            query_embedding = self.embedding_manager.embed_texts([expanded_query], show_progress=False)[0]
            semantic_scores = self._compute_similarity(query_embedding, self.embeddings)
            
            # Convert to numpy for easier manipulation
            semantic_scores_np = semantic_scores.cpu().numpy()
            
            # Perform keyword search using TF-IDF directly
            query_vector = self.tfidf_vectorizer.transform([expanded_query])
            keyword_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Print debug info about scores
            print(f"[DEBUG] Semantic scores - min: {np.min(semantic_scores_np):.4f}, max: {np.max(semantic_scores_np):.4f}, mean: {np.mean(semantic_scores_np):.4f}")
            print(f"[DEBUG] Keyword scores - min: {np.min(keyword_scores):.4f}, max: {np.max(keyword_scores):.4f}, mean: {np.mean(keyword_scores):.4f}")
            
            # Normalize scores using class method
            semantic_scores_np = self._normalize_scores(semantic_scores_np)
            keyword_scores = self._normalize_scores(keyword_scores)
                
            # Combine scores with weights
            semantic_weight = CONFIG.get("semantic_weight", 0.7)
            keyword_weight = CONFIG.get("keyword_weight", 0.3)
            combined_scores = (semantic_weight * semantic_scores_np) + (keyword_weight * keyword_scores)
            
            # Get top-k indices
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            top_scores = combined_scores[top_indices]
            
            # Create results
            results = []
            for score, idx in zip(top_scores, top_indices):
                # Skip chunks below minimum threshold
                if score < CONFIG.get("minimum_score_threshold", 0):
                    continue
                    
                chunk = self.chunks[idx].copy()  # Copy to avoid modifying original
                chunk["similarity_score"] = float(score)
                chunk["semantic_score"] = float(semantic_scores_np[idx])
                chunk["keyword_score"] = float(keyword_scores[idx])
                results.append(chunk)
        else:
            # Regular semantic search
            query_embedding = self.embedding_manager.embed_texts([expanded_query], show_progress=False)[0]
            similarity_scores = self._compute_similarity(query_embedding, self.embeddings)
            top_scores, top_indices = torch.topk(similarity_scores, min(top_k, len(similarity_scores)))
            
            # Create results
            results = []
            for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
                # Skip chunks below minimum threshold
                if score < CONFIG.get("minimum_score_threshold", 0):
                    continue
                    
                chunk = self.chunks[idx].copy()  # Copy to avoid modifying original
                chunk["similarity_score"] = score
                results.append(chunk)
        
        # If we have too few results due to thresholding, try to get more
        if len(results) < 2 and CONFIG.get("minimum_score_threshold", 0) > 0:
            print("[INFO] Too few results above threshold, lowering threshold to get more results")
            # Temporarily lower the threshold
            original_threshold = CONFIG.get("minimum_score_threshold", 0)
            CONFIG["minimum_score_threshold"] = original_threshold * 0.5
            # Recursive call with lower threshold
            results = self.search(query, top_k=top_k)
            # Restore original threshold
            CONFIG["minimum_score_threshold"] = original_threshold
        
        # Apply reranking if enabled
        if CONFIG.get("use_reranking", False) and self.reranker is not None and len(results) > 1:
            results = self._rerank(query, results, top_k=CONFIG.get("rerank_top_k", 5))
            
        return results
    
    def format_results(self, results: List[Dict[str, Any]], include_scores: bool = True) -> str:
        """
        Format search results for display or inclusion in a prompt
        
        Args:
            results: List of dictionaries with chunks and scores
            include_scores: Whether to include similarity scores in the output
            
        Returns:
            str: Formatted results string
        """
        if not results:
            return "No relevant chunks found."
            
        formatted_chunks = []
        
        # Add score statistics if requested
        if include_scores and len(results) > 0:
            score_key = "rerank_score" if "rerank_score" in results[0] else "similarity_score"
            scores = [result[score_key] for result in results if score_key in result]
            
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                
                formatted_chunks.append(f"[Score Statistics] Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}\n")
        
        for i, result in enumerate(results):
            chunk = result["sentence_chunk"]
            page = result["page_number"]
            
            # Create score string based on what's available
            score_info = ""
            if include_scores:
                if "rerank_score" in result:
                    score_info = f", Score: {result['rerank_score']:.4f}"
                    if "original_rerank_score" in result:
                        score_info += f" (Orig: {result['original_rerank_score']:.4f})"
                elif "similarity_score" in result:
                    score_info = f", Score: {result['similarity_score']:.4f}"
                    if "semantic_score" in result and "keyword_score" in result:
                        score_info += f" (S: {result['semantic_score']:.3f}, K: {result['keyword_score']:.3f})"
            
            # Format the chunk with source and score information
            formatted_chunks.append(f"[Chunk {i+1}, Page {page}{score_info}]\n{chunk}\n")
                
        return "\n".join(formatted_chunks) 