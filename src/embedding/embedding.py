"""
Text embedding functions for the Simple Local RAG project
"""
import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Union, Tuple
from sentence_transformers import SentenceTransformer
from src.config import CONFIG

class EmbeddingManager:
    """
    Manages the creation, storage, and retrieval of text embeddings
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding manager with a model
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run the model on (cuda, cpu, mps)
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
        
    def embed_texts(self, 
                   texts: List[str], 
                   batch_size: int = 32,
                   show_progress: bool = True) -> torch.Tensor:
        """
        Embed a list of text chunks
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Create iterator with optional progress bar
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding text chunks")
        
        # Process in batches
        all_embeddings = []
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        embeddings = torch.cat(all_embeddings)
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> torch.Tensor:
        """
        Embed text chunks
        
        Args:
            chunks: List of dictionaries containing text chunks
            batch_size: Batch size for embedding creation
            
        Returns:
            torch.Tensor: Tensor of embeddings
        """
        # Extract text from chunks
        texts = [chunk["sentence_chunk"] for chunk in chunks]
        
        # Store chunks for later use
        self.text_chunks = chunks
        
        # Create embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        # Store embeddings for later use
        self.embeddings = embeddings
        
        return embeddings
    
    def save_embeddings(self, filename: str):
        """
        Save embeddings and chunks to a file
        
        Args:
            filename: Path to the file to save to
        """
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No embeddings or text chunks available to save.")
            
        # Convert embeddings to list for JSON serialization
        embeddings_list = self.embeddings.cpu().numpy().tolist()
        
        # Create data dictionary
        data = {
            "model_name": self.model_name,
            "embeddings": embeddings_list,
            "chunks": self.text_chunks
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f)
            
        print(f"[INFO] Saved {len(embeddings_list)} embeddings to {filename}")
        
    def load_embeddings(self, filename: str):
        """
        Load embeddings and chunks from a file
        
        Args:
            filename: Path to the file to load from
            
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
        
        # Convert embeddings to tensor
        embeddings = torch.tensor(embeddings_list, device=self.device)
        
        # Update instance variables
        self.model_name = model_name
        self.embeddings = embeddings
        self.text_chunks = chunks
        
        print(f"[INFO] Loaded {len(embeddings)} embeddings from {filename}")
        
        return embeddings, chunks 