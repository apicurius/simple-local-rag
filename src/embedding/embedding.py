"""
Text embedding functions for the Simple Local RAG project
"""
import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer
from src.config import CONFIG

# Conditionally import faiss if available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[INFO] FAISS not available. Vector database features will be disabled.")
    print("[INFO] To enable vector database support, install FAISS: pip install faiss-cpu or faiss-gpu")

class EmbeddingManager:
    """
    Manages the creation, storage, and retrieval of text embeddings
    with support for caching, incremental updates, and vector database integration
    """
    
    def __init__(self, 
                model_name: str = None, 
                device: str = None,
                use_vector_db: bool = False,
                vector_db_config: Dict[str, Any] = None):
        """
        Initialize the embedding manager with a model
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run the model on (cuda, cpu, mps)
            use_vector_db: Whether to use a vector database for faster similarity search
            vector_db_config: Configuration for the vector database
        """
        if model_name is None:
            model_name = CONFIG["embedding_model_name"]
        
        if device is None:
            device = CONFIG["device"]
            
        self.model_name = model_name
        self.device = device
        
        print(f"[INFO] Loading embedding model '{model_name}' on {device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
        # Track whether embeddings have been loaded or created
        self.embeddings = None
        self.text_chunks = None
        
        # Vector database settings
        self.use_vector_db = use_vector_db and FAISS_AVAILABLE
        self.vector_db = None
        self.vector_db_config = vector_db_config or {}
        
        if self.use_vector_db and not FAISS_AVAILABLE:
            print("[WARNING] Vector database requested but FAISS is not available. Falling back to standard search.")
            self.use_vector_db = False
        
    def embed_texts(self, 
                   texts: List[str], 
                   batch_size: int = 32,
                   show_progress: bool = True,
                   use_cache: bool = True) -> torch.Tensor:
        """
        Embed a list of text chunks with caching support
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
            use_cache: Whether to use embedding cache to avoid recomputing
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Initialize cache dictionary if not exists
        if not hasattr(self, 'embedding_cache'):
            self.embedding_cache = {}
            
        # Find which texts need embedding (not in cache)
        texts_to_embed = []
        indices_to_embed = []
        cached_embeddings = []
        embedding_dim = None  # Will be set when we get the first embedding
        
        if use_cache:
            # Determine which texts need embedding
            for i, text in enumerate(texts):
                text_hash = hash(text)  # Use hash as cache key
                if text_hash in self.embedding_cache:
                    # Get from cache
                    cached_emb = self.embedding_cache[text_hash]
                    if embedding_dim is None and cached_emb is not None:
                        embedding_dim = cached_emb.shape[0]
                    cached_embeddings.append((i, cached_emb))
                else:
                    # Need to compute
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            # Embed all texts (no cache)
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
        
        # If there are texts to embed
        if texts_to_embed:
            # Create iterator with optional progress bar
            iterator = range(0, len(texts_to_embed), batch_size)
            if show_progress:
                desc = f"Embedding {len(texts_to_embed)} text chunks"
                if use_cache and len(texts) != len(texts_to_embed):
                    desc += f" ({len(texts) - len(texts_to_embed)} from cache)"
                iterator = tqdm(iterator, desc=desc)
            
            # Process in batches
            all_new_embeddings = []
            for i in iterator:
                batch = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                all_new_embeddings.append(batch_embeddings)
                
                # Update cache for this batch
                if use_cache:
                    batch_indices = indices_to_embed[i:i + batch_size]
                    for b_idx, text_idx in enumerate(batch_indices):
                        text_hash = hash(texts[text_idx])
                        self.embedding_cache[text_hash] = batch_embeddings[b_idx].cpu()  # Store on CPU to save GPU memory
            
            # Concatenate all new batch embeddings
            if all_new_embeddings:
                new_embeddings = torch.cat(all_new_embeddings)
                
                # Get embedding dimension if not set yet
                if embedding_dim is None and len(new_embeddings) > 0:
                    embedding_dim = new_embeddings.shape[1]
            else:
                new_embeddings = torch.tensor([], device=self.device)
        else:
            new_embeddings = torch.tensor([], device=self.device)
        
        # If everything was cached
        if not texts_to_embed and cached_embeddings:
            # Just return the cached embeddings in the right order
            cached_embeddings.sort(key=lambda x: x[0])  # Sort by original index
            result = torch.stack([emb for _, emb in cached_embeddings]).to(self.device)
            return result
            
        # If we have a mix of new and cached embeddings
        if use_cache and cached_embeddings:
            # Create a tensor to hold all embeddings
            all_embeddings = torch.zeros((len(texts), embedding_dim), device=self.device)
            
            # Fill in the new embeddings at their indices
            for i, idx in enumerate(indices_to_embed):
                if i < len(new_embeddings):
                    all_embeddings[idx] = new_embeddings[i]
            
            # Fill in the cached embeddings at their indices
            for idx, emb in cached_embeddings:
                all_embeddings[idx] = emb.to(self.device)
                
            return all_embeddings
        
        # If we only have new embeddings, just return them
        return new_embeddings
    
    def embed_chunks(self, 
                     chunks: List[Dict[str, Any]], 
                     batch_size: int = 32,
                     incremental: bool = False,
                     use_cache: bool = True) -> torch.Tensor:
        """
        Embed text chunks with option for incremental updates
        
        Args:
            chunks: List of dictionaries containing text chunks
            batch_size: Batch size for embedding creation
            incremental: If True, add new chunks to existing ones; if False, replace
            use_cache: Whether to use embedding cache
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Extract text from chunks
        texts = [chunk["sentence_chunk"] for chunk in chunks]
        
        # Handle incremental mode
        if incremental and self.text_chunks is not None and self.embeddings is not None:
            # Compute unique chunk IDs based on content for deduplication
            existing_chunk_ids = {self._get_chunk_id(chunk): i for i, chunk in enumerate(self.text_chunks)}
            
            # Track which new chunks to add (skip duplicates)
            new_chunks = []
            for chunk in chunks:
                chunk_id = self._get_chunk_id(chunk)
                if chunk_id not in existing_chunk_ids:
                    new_chunks.append(chunk)
            
            # If there are new chunks to add
            if new_chunks:
                # Get embeddings for new chunks
                new_texts = [chunk["sentence_chunk"] for chunk in new_chunks]
                new_embeddings = self.embed_texts(new_texts, batch_size=batch_size, use_cache=use_cache)
                
                # Combine with existing ones
                self.text_chunks.extend(new_chunks)
                self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
                
                print(f"[INFO] Added {len(new_chunks)} new chunks to existing {len(existing_chunk_ids)} chunks")
            else:
                print(f"[INFO] No new chunks to add (all {len(chunks)} chunks already exist)")
        else:
            # Regular mode - replace existing data
            # Store chunks for later use
            self.text_chunks = chunks
            
            # Create embeddings
            self.embeddings = self.embed_texts(texts, batch_size=batch_size, use_cache=use_cache)
        
        return self.embeddings
        
    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a chunk based on its content
        
        Args:
            chunk: Dictionary containing chunk data
            
        Returns:
            str: Unique ID
        """
        # Use the text content and page number as the identifier
        chunk_text = chunk.get("sentence_chunk", "")
        page_num = chunk.get("page_number", "")
        return f"{page_num}:{hash(chunk_text)}"
    
    def save_embeddings(self, filename: str, save_cache: bool = True):
        """
        Save embeddings, chunks, and optionally the cache to a file
        
        Args:
            filename: Path to the file to save to
            save_cache: Whether to save the embedding cache
        """
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks available to save.")
            
        # Convert embeddings to list for JSON serialization
        embeddings_list = self.embeddings.cpu().numpy().tolist()
        
        # Create data dictionary
        data = {
            "model_name": self.model_name,
            "embeddings": embeddings_list,
            "chunks": self.text_chunks,
            "version": "2.0",  # Add versioning for future compatibility
            "metadata": {
                "embedding_dim": self.embeddings.shape[1],
                "num_chunks": len(self.text_chunks),
                "created_at": str(np.datetime64('now'))
            }
        }
        
        # Save cache if requested
        if save_cache and hasattr(self, 'embedding_cache') and self.embedding_cache:
            # Convert cache to serializable format
            cache_data = {}
            for text_hash, embedding in self.embedding_cache.items():
                # Convert tensor to list 
                emb_list = embedding.cpu().numpy().tolist()
                # Use string hash as key
                cache_data[str(text_hash)] = emb_list
                
            data["cache"] = cache_data
            print(f"[INFO] Including embedding cache with {len(cache_data)} entries")
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f)
            
        print(f"[INFO] Saved {len(embeddings_list)} embeddings to {filename}")
        
    def load_embeddings(self, filename: str, load_cache: bool = True):
        """
        Load embeddings, chunks, and optionally the cache from a file
        
        Args:
            filename: Path to the file to load from
            load_cache: Whether to load the embedding cache
            
        Returns:
            tuple: (embeddings, chunks)
        """
        if not os.path.exists(filename):
            raise ValueError(f"Embeddings file {filename} not found.")
            
        # Load from file
        with open(filename, "r") as f:
            data = json.load(f)
            
        # Extract data
        model_name = data.get("model_name", self.model_name)
        embeddings_list = data["embeddings"]
        chunks = data["chunks"]
        version = data.get("version", "1.0")
        
        # Convert embeddings to tensor
        embeddings = torch.tensor(embeddings_list, device=self.device)
        
        # Update instance variables
        self.model_name = model_name
        self.embeddings = embeddings
        self.text_chunks = chunks
        
        # Load cache if available and requested
        if load_cache and "cache" in data:
            cache_data = data["cache"]
            # Initialize cache dictionary if not exists
            if not hasattr(self, 'embedding_cache'):
                self.embedding_cache = {}
                
            # Load cache entries
            for text_hash_str, emb_list in cache_data.items():
                # Convert string hash back to int and list back to tensor
                self.embedding_cache[int(text_hash_str)] = torch.tensor(emb_list)
                
            print(f"[INFO] Loaded embedding cache with {len(cache_data)} entries")
                
        print(f"[INFO] Loaded {len(embeddings)} embeddings from {filename} (version {version})")
        
        # Initialize vector database if needed
        if self.use_vector_db:
            self._initialize_vector_db()
        
        return embeddings, chunks
        
    def _initialize_vector_db(self):
        """
        Initialize the vector database (FAISS) for fast similarity search
        """
        if not FAISS_AVAILABLE:
            print("[WARNING] Cannot initialize vector database: FAISS not available")
            return
            
        if self.embeddings is None:
            print("[WARNING] Cannot initialize vector database: no embeddings available")
            return
            
        # Get embedding dimension
        embedding_dim = self.embeddings.shape[1]
        
        # Create index
        index_type = self.vector_db_config.get("index_type", "flat")
        
        if index_type == "flat":
            # Simple, exact search (slower but most accurate)
            self.vector_db = faiss.IndexFlatIP(embedding_dim)  # Inner product (normalized vectors = cosine sim)
        elif index_type == "ivf":
            # Inverted file index (faster, slight accuracy trade-off)
            n_clusters = min(self.vector_db_config.get("n_clusters", 100), len(self.embeddings) // 10)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.vector_db = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            # Train the index
            if not self.vector_db.is_trained:
                embeddings_np = self.embeddings.cpu().numpy().astype(np.float32)
                self.vector_db.train(embeddings_np)
        else:
            # Default to flat index
            self.vector_db = faiss.IndexFlatIP(embedding_dim)
            
        # Add vectors to the index
        embeddings_np = self.embeddings.cpu().numpy().astype(np.float32)
        self.vector_db.add(embeddings_np)
        
        print(f"[INFO] Initialized vector database with {len(self.embeddings)} vectors")
        
    def vector_search(self, 
                      query_embedding: torch.Tensor, 
                      top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Perform vector similarity search using the vector database
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top results to return
            
        Returns:
            Tuple[List[int], List[float]]: Tuple of (indices, distances)
        """
        if not self.use_vector_db or self.vector_db is None:
            raise ValueError("Vector database not initialized or disabled")
            
        # Convert query embedding to numpy
        query_np = query_embedding.cpu().numpy().reshape(1, -1).astype(np.float32)
        
        # Normalize query vector (FAISS IndexFlatIP assumes normalized vectors for cosine sim)
        faiss.normalize_L2(query_np)
        
        # Search
        top_k = min(top_k, len(self.embeddings))
        distances, indices = self.vector_db.search(query_np, top_k)
        
        # Return results
        return indices[0].tolist(), distances[0].tolist()
        
    def reset_vector_db(self):
        """
        Reset the vector database (e.g., after embeddings have changed)
        """
        if self.use_vector_db:
            self.vector_db = None
            self._initialize_vector_db()
        
    def clear_cache(self):
        """
        Clear the embedding cache to free memory
        """
        if hasattr(self, 'embedding_cache'):
            cache_size = len(self.embedding_cache)
            self.embedding_cache = {}
            print(f"[INFO] Cleared embedding cache ({cache_size} entries)")
        else:
            print("[INFO] No embedding cache to clear")
            
    def get_cache_stats(self):
        """
        Get statistics about the embedding cache
        
        Returns:
            dict: Cache statistics
        """
        stats = {
            "cache_exists": hasattr(self, 'embedding_cache'),
            "cache_size": 0,
            "memory_usage_mb": 0,
        }
        
        if stats["cache_exists"]:
            stats["cache_size"] = len(self.embedding_cache)
            
            # Estimate memory usage (approximate)
            if stats["cache_size"] > 0:
                # Get a sample entry to determine embedding size
                sample_embed = next(iter(self.embedding_cache.values()))
                embed_size_bytes = sample_embed.element_size() * sample_embed.nelement()
                stats["memory_usage_mb"] = (embed_size_bytes * stats["cache_size"]) / (1024 * 1024)
                
        return stats 