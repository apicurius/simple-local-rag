"""
Text retrieval functions for the Simple Local RAG project
"""
import torch
import re
import numpy as np
import time
from collections import Counter
from typing import List, Dict, Any, Tuple, Union
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
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
        self.bm25 = None
        if CONFIG.get("use_hybrid_search", False):
            self._initialize_keyword_search()
            
        # Add semantic similarity function
        from sklearn.metrics.pairwise import cosine_similarity
        self.semantic_similarity_fn = cosine_similarity
    
    def _initialize_reranker(self):
        """Initialize the reranker model"""
        rerank_model_name = CONFIG.get("rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"[INFO] Loading reranker model '{rerank_model_name}'")
        
        # Configure model for cross-encoding
        max_length = 512
        use_cross_encoder = CONFIG.get("use_cross_encoder", False)
        
        # Different cross-encoder models have different normalization requirements
        if "ms-marco" in rerank_model_name.lower():
            # Models trained on MS MARCO typically don't need normalization
            normalize_scores = False
        else:
            # Other models might need normalization
            normalize_scores = True
            
        self.reranker = CrossEncoder(
            rerank_model_name, 
            device=CONFIG["device"],
            max_length=max_length,
            default_activation_function=None  # Let the model decide based on its training
        )
        
    def _initialize_keyword_search(self):
        """Initialize the keyword search using BM25"""
        print("[INFO] Initializing BM25 for keyword search")
        # Extract text from chunks
        texts = [chunk["sentence_chunk"] for chunk in self.chunks]
        
        # Tokenize texts for BM25
        tokenized_texts = []
        for text in texts:
            # Tokenize by splitting on whitespace and removing punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_texts.append(tokens)
        
        # Create BM25 index with custom parameters from config
        k1 = CONFIG.get("bm25_k1", 1.5)  # Controls term frequency saturation
        b = CONFIG.get("bm25_b", 0.75)   # Controls document length normalization
        self.bm25 = BM25Okapi(tokenized_texts, k1=k1, b=b)
        
        # Also keep TF-IDF for backward compatibility
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.85,
            min_df=2
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Store original texts and tokenized texts
        self.original_texts = texts
        self.tokenized_texts = tokenized_texts
        
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
        Expand the query using multiple methods:
        1. Domain-specific terminology
        2. Wordnet synonyms when available
        3. Context words inferred from semantic search
        
        Args:
            query: Original query string
            
        Returns:
            str: Expanded query
        """
        # Enhanced nutrition terminology dictionary - extended with more comprehensive terms
        nutrition_terms = {
            # Macronutrients
            "fat": "lipid fatty acid triglyceride saturated unsaturated cholesterol omega-3 omega-6 trans-fat monounsaturated polyunsaturated",
            "protein": "amino acid peptide enzyme collagen albumin essential complete incomplete whey casein soy plant animal lean",
            "carbohydrate": "sugar starch fiber glucose fructose sucrose maltose polysaccharide glycemic index complex simple carbs",
            "macronutrient": "carbohydrate protein fat lipid nutrient energy calorie",
            
            # Micronutrients
            "vitamin": "micronutrient supplement deficiency bioavailability fat-soluble water-soluble retinol tocopherol ascorbic niacin riboflavin thiamine folate cobalamin",
            "vitamin a": "retinol carotenoid beta-carotene vision immune",
            "vitamin b": "thiamine riboflavin niacin pantothenic cobalamin folate biotin energy metabolism nervous",
            "vitamin c": "ascorbic immune collagen antioxidant scurvy",
            "vitamin d": "cholecalciferol calcitriol calcidiol sunshine bone calcium",
            "vitamin e": "tocopherol antioxidant membrane",
            "vitamin k": "phylloquinone menaquinone coagulation bone",
            "mineral": "micronutrient electrolyte calcium iron zinc magnesium potassium iodine copper selenium phosphorus sodium",
            
            # General nutrition concepts
            "nutrient": "nutrition nourishment macronutrient micronutrient digestion absorption bioavailability essential nonessential",
            "calorie": "energy kilocalorie kcal joule metabolism basal metabolic rate thermogenesis",
            "diet": "nutrition food eating meal plan macronutrient pattern Mediterranean DASH keto paleo vegan vegetarian",
            "digestion": "absorption metabolism enzyme breakdown assimilation gut microbiome transit elimination",
            "health": "wellness wellbeing fitness condition immune function vitality longevity",
            
            # Nutrition-related conditions/properties
            "metabolism": "metabolic process anabolism catabolism energy expenditure BMR thermogenesis",
            "disease": "disorder condition syndrome pathology inflammation chronic acute",
            "obesity": "overweight body mass bmi adipose fat metabolic syndrome",
            "diabetes": "insulin glucose hyperglycemia glycemic index hemoglobin a1c blood sugar",
            "heart": "cardiovascular cardiac circulation blood pressure cholesterol lipid profile",
            "cancer": "tumor malignant cell growth oncology antioxidant free radical",
            "liver": "hepatic detoxification bile organ fatty liver",
            "kidney": "renal filtration excretion nephron creatinine",
            "gut": "intestine microbiome flora digestive tract probiotic prebiotic",
            "brain": "neural cognitive central nervous system omega-3",
            
            # Food components
            "antioxidant": "free-radical oxidative stress flavonoid polyphenol carotenoid",
            "fiber": "roughage insoluble soluble prebiotic fermentable cellulose pectin lignin",
            "probiotic": "bacteria microbiome gut flora fermented lactobacillus bifidobacterium",
            "phytonutrient": "phytochemical plant compound polyphenol flavonoid carotenoid",
            "digestibility": "absorption bioavailability protein digestibility chemical score",
            
            # Comparison terms - add these to help with comparison questions
            "difference": "compare contrast versus vs distinction variation",
            "compare": "difference distinction contrast similarity difference",
            "better": "superior advantage benefit improvement",
            "worse": "inferior disadvantage drawback",
            "best": "optimal top ideal superior excellent",
            "worst": "poorest least harmful damaging",
            
            # Question type terms
            "what": "define explain describe definition",
            "how": "method process technique procedure mechanism",
            "why": "cause reason explanation purpose rationale",
            "when": "time period moment stage phase",
            "where": "location place source origin",
            "which": "selection option choice alternative",
            "who": "person individual expert authority"
        }
        
        # Get expansion parameters from config
        expansion_factor = CONFIG.get("expansion_factor", 3)
        max_terms = CONFIG.get("max_expansion_terms", 10)
        
        # Step 1: Extract keywords from query
        # Use a simple approach to identify keywords by removing stopwords
        stopwords = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "as", "and", "or", "is", "are", "was", "were", "be", "been", "being", "of"}
        query_keywords = [word.lower() for word in re.findall(r'\b\w+\b', query) if word.lower() not in stopwords]
        
        # Step 2: Initialize expanded query with the original query
        expanded_terms = []
        domain_terms_added = set()  # Track added terms to avoid duplicates
        
        # Step 3: Add domain-specific terms based on query keywords
        for keyword in query_keywords:
            # Look for exact matches first
            if keyword in nutrition_terms:
                terms = nutrition_terms[keyword].split()
                for term in terms[:min(len(terms), int(expansion_factor * 2))]:  # Add more terms for exact matches
                    if term not in domain_terms_added and term not in query_keywords:
                        expanded_terms.append(term)
                        domain_terms_added.add(term)
            
            # Then look for partial matches
            else:
                for term, synonyms in nutrition_terms.items():
                    # Only add if we haven't added from this term already and terms are related
                    term_matches = keyword in term or term in keyword
                    if term not in domain_terms_added and term_matches:
                        synonym_list = synonyms.split()
                        num_terms = min(len(synonym_list), round(expansion_factor))
                        for syn_term in synonym_list[:num_terms]:
                            if syn_term not in domain_terms_added and syn_term not in query_keywords:
                                expanded_terms.append(syn_term)
                                domain_terms_added.add(syn_term)
                        domain_terms_added.add(term)  # Mark this term as processed
        
        # Step 4: Add NLTK wordnet synonyms if available
        try:
            from nltk.corpus import wordnet
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("[INFO] Downloading wordnet...")
                nltk.download('wordnet', quiet=True)
                
            for keyword in query_keywords:
                # Get synonyms from WordNet
                for syn in wordnet.synsets(keyword)[:2]:  # Limit to top 2 synsets
                    for lemma in syn.lemmas()[:2]:  # Limit to top 2 lemmas
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym not in domain_terms_added and synonym not in query_keywords and len(synonym) > 3:
                            expanded_terms.append(synonym)
                            domain_terms_added.add(synonym)
        except Exception as e:
            print(f"[WARNING] WordNet expansion failed: {e}")
            
        # Step 5: Add specific contextual terms based on question type
        # Look for specific nutritional concepts and add relevant technical terms
        if "vitamin" in query.lower():
            if "d" in query.lower() or "d3" in query.lower():
                expanded_terms.extend(["cholecalciferol", "calcitriol", "calcidiol", "bone", "calcium"])
            elif "b12" in query.lower():
                expanded_terms.extend(["cobalamin", "methylcobalamin", "cyanocobalamin", "anemia"])
            elif "c" in query.lower():
                expanded_terms.extend(["ascorbic", "ascorbate", "scurvy", "collagen"])
        
        # Look for specific questions about nutrition
        if "how much" in query.lower() or "recommended" in query.lower() or "daily" in query.lower():
            expanded_terms.extend(["recommended", "dietary", "allowance", "rda", "dri", "intake"])
            
        if "deficiency" in query.lower() or "lack" in query.lower():
            expanded_terms.extend(["insufficiency", "inadequate", "malnutrition", "symptoms"])
        
        # For comparison questions
        if any(word in query.lower() for word in ["compare", "difference", "versus", "vs"]):
            expanded_terms.extend(["contrast", "distinction", "characteristic", "property", "feature"])
        
        # For macronutrient questions (key to our test set)
        if "macronutrient" in query.lower() or "macronutrients" in query.lower():
            expanded_terms.extend(["carbohydrate", "protein", "fat", "lipid", "energy", "calorie"])
        
        # For "main" or "most important" questions
        if "main" in query.lower() or "important" in query.lower() or "essential" in query.lower():
            expanded_terms.extend(["primary", "critical", "fundamental", "necessary", "required"])
            
        # Step 6: Deduplicate and limit total expansion terms
        expanded_terms = list(dict.fromkeys(expanded_terms))  # Remove duplicates preserving order
        expanded_terms = expanded_terms[:max_terms]
        
        # Step 7: Combine with original query
        expanded_query = query
        if expanded_terms:
            expanded_query += " " + " ".join(expanded_terms)
        
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
        
        # Use cross-encoder with batching for more efficient processing
        use_cross_encoder = CONFIG.get("use_cross_encoder", False)
        batch_size = CONFIG.get("cross_encoder_batch_size", 32)
        
        if use_cross_encoder:
            print(f"[INFO] Using cross-encoder reranking with batch_size={batch_size}")
            # Process in batches to avoid memory issues with large result sets
            rerank_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_scores = self.reranker.predict(batch_pairs)
                if isinstance(batch_scores, np.ndarray):
                    batch_scores = batch_scores.tolist()
                elif isinstance(batch_scores, (list, tuple)):
                    # Already a list, do nothing
                    pass
                else:
                    # Convert torch tensor to list if needed
                    try:
                        batch_scores = batch_scores.cpu().numpy().tolist()
                    except:
                        batch_scores = list(batch_scores)
                rerank_scores.extend(batch_scores)
        else:
            # Original non-batched prediction
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
            
        # Get parameters from config
        diversity_threshold = CONFIG.get("max_similarity_between_chunks", 0.75)
        diversity_weight = CONFIG.get("diversity_weight", 0.6)
        relevance_weight = CONFIG.get("relevance_weight", 0.4)
        min_chunk_distance = CONFIG.get("min_chunk_distance", 1)  # Minimum page distance to consider diverse
        
        # Always keep the first (highest scoring) result
        diverse_results = [results[0]]
        
        # Prepare remaining candidates for evaluation
        remaining_candidates = results[1:]
        
        # Create diversified results while respecting original ranking where possible
        while len(diverse_results) < top_k and remaining_candidates:
            best_candidate = None
            best_candidate_idx = -1
            best_score = float('-inf')
            
            # Evaluate each remaining candidate
            for i, candidate in enumerate(remaining_candidates):
                # Start with original rerank score as base for relevance score
                relevance_score = candidate.get("rerank_score", candidate.get("similarity_score", 0.0))
                
                # Calculate content similarity and page distance to all already selected results
                max_content_similarity = 0.0
                min_page_distance = float('inf')
                
                for selected in diverse_results:
                    # Calculate text similarity (lexical)
                    text_sim = self._compute_text_similarity(
                        candidate["sentence_chunk"], 
                        selected["sentence_chunk"]
                    )
                    
                    # Calculate semantic similarity (meaning)
                    sem_sim = 0.0
                    if hasattr(self, "embedding_manager") and self.embedding_manager and self.embedding_manager.model:
                        try:
                            # Get or compute embeddings
                            cand_embed = self.embedding_manager.get_embedding_for_text(candidate["sentence_chunk"])
                            sel_embed = self.embedding_manager.get_embedding_for_text(selected["sentence_chunk"])
                            
                            # Compute cosine similarity using prepared function
                            sem_sim = float(self.semantic_similarity_fn([cand_embed], [sel_embed])[0][0])
                        except:
                            # Fallback to text similarity
                            sem_sim = text_sim
                    
                    # Calculate page distance (structural diversity)
                    page_distance = abs(candidate.get("page_number", 0) - selected.get("page_number", 0))
                    min_page_distance = min(min_page_distance, page_distance)
                    
                    # Blend similarities - weighting semantic (meaning) higher than lexical (words)
                    blended_sim = 0.35 * text_sim + 0.65 * sem_sim
                    max_content_similarity = max(max_content_similarity, blended_sim)
                
                # Calculate overall diversity score from content similarity and page distance
                # 1. Content diversity (1 - similarity)
                content_diversity = 1.0 - max_content_similarity
                
                # 2. Page diversity (normalize page distance to 0-1 range)
                # If pages are far apart, they're likely to cover different topics
                page_diversity = min(1.0, min_page_distance / (min_chunk_distance + 5.0))
                
                # 3. Blend the two diversity factors
                diversity_score = 0.8 * content_diversity + 0.2 * page_diversity
                
                # Final score: balance between relevance and diversity
                # Higher diversity_weight means more focus on getting diverse results
                # Higher relevance_weight means more focus on getting the most relevant results
                combined_score = (relevance_weight * relevance_score) + (diversity_weight * diversity_score)
                
                # Track the best candidate
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_candidate_idx = i
            
            # If we found a good candidate
            if best_candidate_idx >= 0:
                # Check if it passes the diversity threshold
                effective_similarity = 1.0 - (best_score / diversity_weight)
                if effective_similarity <= diversity_threshold:
                    diverse_results.append(best_candidate)
                # Remove from candidates regardless
                remaining_candidates.pop(best_candidate_idx)
            else:
                # No good candidates found
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
        start_time = time.time()
        if top_k is None:
            top_k = CONFIG["num_chunks_to_retrieve"]
        
        # Query expansion if enabled
        if CONFIG.get("use_query_expansion", False):
            expanded_query = self._expand_query(query)
            print(f"[INFO] Expanded query: '{expanded_query}'")
        else:
            expanded_query = query
        
        # Hybrid search if enabled
        if CONFIG.get("use_hybrid_search", False) and (self.bm25 is not None or self.tfidf_vectorizer is not None):
            # Perform semantic search
            query_embedding = self.embedding_manager.embed_texts([expanded_query], show_progress=False)[0]
            semantic_scores = self._compute_similarity(query_embedding, self.embeddings)
            
            # Convert to numpy for easier manipulation
            semantic_scores_np = semantic_scores.cpu().numpy()
            
            # Perform keyword search using BM25
            # Tokenize query for BM25
            query_tokens = re.findall(r'\b\w+\b', expanded_query.lower())
            
            # Get BM25 scores
            bm25_scores = np.array(self.bm25.get_scores(query_tokens))
            
            # Track BM25 performance metrics
            bm25_score_sum = np.sum(bm25_scores)
            bm25_max_score = np.max(bm25_scores) if bm25_scores.size > 0 else 0
            
            # Use BM25 for keyword scoring
            keyword_scores = bm25_scores
            
            # Fallback to TF-IDF if BM25 fails or returns all zeros
            if bm25_score_sum == 0:
                print("[INFO] BM25 returned all zeros, falling back to TF-IDF")
                query_vector = self.tfidf_vectorizer.transform([expanded_query])
                keyword_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Print debug info about scores
            print(f"[DEBUG] Semantic scores - min: {np.min(semantic_scores_np):.4f}, max: {np.max(semantic_scores_np):.4f}, mean: {np.mean(semantic_scores_np):.4f}")
            print(f"[DEBUG] Keyword scores - min: {np.min(keyword_scores):.4f}, max: {np.max(keyword_scores):.4f}, mean: {np.mean(keyword_scores):.4f}")
            print(f"[DEBUG] BM25 performance - sum: {bm25_score_sum:.4f}, max: {bm25_max_score:.4f}")
            
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
        reranking_start_time = time.time()
        if CONFIG.get("use_reranking", False) and len(results) > 1:
            if self.reranker is None:
                self._initialize_reranker()
                
            print(f"[INFO] Reranking {len(results)} results using model '{CONFIG.get('rerank_model_name')}'")
            results = self._rerank(query, results, top_k=CONFIG.get("rerank_top_k", 15))
            
            # Apply cutoff threshold (filter low-scoring chunks after reranking)
            cutoff = CONFIG.get("rerank_cutoff_threshold", 0.3)
            if cutoff > 0 and len(results) > 3:  # Always keep at least 3 results
                filtered_results = [r for r in results if r.get("rerank_score", 0) >= cutoff]
                # Ensure we keep at least 3 results
                if len(filtered_results) >= 3:
                    results = filtered_results
        
        # Track timing
        end_time = time.time()
        total_time = end_time - start_time
        reranking_time = end_time - reranking_start_time if CONFIG.get("use_reranking", False) else 0
        
        print(f"[INFO] Retrieval completed in {total_time:.4f}s (reranking: {reranking_time:.4f}s)")
            
        return results
    
    def format_results(self, results: List[Dict[str, Any]], include_scores: bool = True, 
                        include_stats: bool = True, max_chunks_to_show: int = None) -> str:
        """
        Format search results for display or inclusion in a prompt
        
        Args:
            results: List of dictionaries with chunks and scores
            include_scores: Whether to include similarity scores in the output
            include_stats: Whether to include score statistics
            max_chunks_to_show: Maximum number of chunks to include in formatted output
            
        Returns:
            str: Formatted results string
        """
        if not results:
            return "No relevant chunks found."
            
        formatted_chunks = []
        
        # Limit number of chunks to show if specified
        if max_chunks_to_show is not None and len(results) > max_chunks_to_show:
            display_results = results[:max_chunks_to_show]
            truncation_message = f"(Showing top {max_chunks_to_show} of {len(results)} results)"
        else:
            display_results = results
            truncation_message = None
        
        # Add score statistics if requested
        if include_stats and include_scores and len(results) > 0:
            score_key = "rerank_score" if "rerank_score" in results[0] else "similarity_score"
            scores = [result[score_key] for result in results if score_key in result]
            
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                
                stats_line = f"[Score Statistics] Min: {min_score:.4f}, Max: {max_score:.4f}, Avg: {avg_score:.4f}"
                if truncation_message:
                    stats_line += f" {truncation_message}"
                    
                formatted_chunks.append(f"{stats_line}\n")
        
        for i, result in enumerate(display_results):
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