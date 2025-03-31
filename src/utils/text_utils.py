"""
Text processing utilities for the Simple Local RAG project
"""
import re
from typing import List, Dict, Any
import spacy
from tqdm.auto import tqdm
from src.config import CONFIG

# Try to load the spaCy English model, or fall back to the English language class
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    print("[INFO] Loaded spaCy model 'en_core_web_sm'")
except OSError:
    # If the model is not available, use the English language class
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe("sentencizer")
    print("[INFO] Using spaCy English language class with sentencizer")

def process_sentences(pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process text into sentences using spaCy
    
    Args:
        pages_and_texts: List of dictionaries containing page numbers and text
        
    Returns:
        List[Dict]: List with page numbers and processed sentences
    """
    result = []
    
    # Process each page
    for page_dict in tqdm(pages_and_texts, desc="Processing sentences"):
        page_number = page_dict["page_number"]
        text = page_dict["text"]
        
        # Skip empty text
        if not text or not text.strip():
            continue
        
        # Process with spaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Skip if no sentences
        if not sentences:
            continue
        
        # Add to result
        result.append({"page_number": page_number, "sentences": sentences})
    
    return result

def create_sentence_chunks(pages_and_sentences: List[Dict[str, Any]], 
                           chunk_size: int = None, 
                           overlap: int = None) -> List[Dict[str, Any]]:
    """
    Create chunks of sentences
    
    Args:
        pages_and_sentences: List of dictionaries containing page numbers and sentences
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List[Dict]: List of dictionaries with page numbers and sentence chunks
    """
    # Use config values if not provided
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    
    if overlap is None:
        overlap = CONFIG.get("chunk_overlap", 0)
    
    result = []
    
    # Process each page
    for page_dict in pages_and_sentences:
        page_number = page_dict["page_number"]
        sentences = page_dict["sentences"]
        
        # Create chunks with overlap
        for i in range(0, len(sentences), chunk_size - overlap):
            # Skip last chunk if it's too small
            if i + chunk_size > len(sentences):
                if len(sentences) - i < max(3, chunk_size // 3):  # Minimum size is 3 or 1/3 of chunk_size
                    break
            
            # Get sentences for this chunk (limited by available sentences)
            chunk_sentences = sentences[i:min(i + chunk_size, len(sentences))]
            chunk_text = ' '.join(chunk_sentences)
            
            result.append({
                "page_number": page_number,
                "sentence_chunk": chunk_text
            })
    
    return result

def process_chunks_to_items(chunks: List[Dict[str, Any]], min_token_length: int = None) -> List[Dict[str, Any]]:
    """
    Process chunks into items for embedding, filtering by token length
    
    Args:
        chunks: List of dictionaries containing page numbers and sentence chunks
        min_token_length: Minimum token length for chunks to be included
        
    Returns:
        List[Dict]: List of dictionaries with processed chunks
    """
    # Use config value if not provided
    if min_token_length is None:
        min_token_length = CONFIG["min_token_length"]
    
    result = []
    
    # Process each chunk
    for chunk in chunks:
        page_number = chunk["page_number"]
        chunk_text = chunk["sentence_chunk"]
        
        # Skip empty chunks
        if not chunk_text or not chunk_text.strip():
            continue
        
        # Calculate token count (rough approximation)
        # A more accurate method would use the actual tokenizer from the embedding model
        token_count = len(re.findall(r'\b\w+\b', chunk_text)) * 1.3  # Add 30% for tokenization differences
        
        # Skip chunks that are too small
        if token_count < min_token_length:
            continue
        
        # Calculate character and word counts
        char_count = len(chunk_text)
        word_count = len(re.findall(r'\b\w+\b', chunk_text))
        
        # Add to result
        result.append({
            "page_number": page_number,
            "sentence_chunk": chunk_text,
            "chunk_char_count": char_count,
            "chunk_word_count": word_count,
            "chunk_token_count": token_count
        })
    
    return result 