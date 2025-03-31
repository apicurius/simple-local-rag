"""
Configuration settings for the Simple Local RAG project
"""
import os
import json
import torch

# Global configuration dictionary
CONFIG = {
    # Device settings
    "device": "cpu",  # Will be updated based on availability
    "device_map": "auto",
    "load_in_8bit": False,
    "load_in_4bit": False,
    
    # PDF settings
    "pdf_path": "human-nutrition-text.pdf",
    
    # Text processing settings
    "chunk_size": 10,  # Number of sentences per chunk
    "min_token_length": 30,  # Minimum token length for chunks
    "chunk_overlap": 3,      # Number of sentences to overlap between chunks (increased from default)
    
    # Embedding settings
    "embedding_model_name": "BAAI/bge-large-en-v1.5",
    "embeddings_filename": "nutrition_text_embeddings.json",
    
    # LLM settings
    "llm_model_name": "google/gemma-3-4b-it",
    "llm_context_window": 8192,
    "max_new_tokens": 1024,
    "temperature": 0.3,
    
    # Retrieval settings
    "num_chunks_to_retrieve": 10,  # Increased from 5 to get more candidates before filtering
    "minimum_score_threshold": 0.10,  # Lowered from 0.25 to catch more potentially relevant chunks
    
    # Hybrid search settings
    "use_hybrid_search": True,
    "semantic_weight": 0.65,  # Slightly increased semantic weight for better relevance
    "keyword_weight": 0.35,   # Adjusted keyword weight
    
    # Query expansion settings
    "use_query_expansion": True,
    "expansion_factor": 2,   # How aggressive the query expansion should be (higher = more terms)
    
    # Reranking settings
    "use_reranking": True,
    "rerank_model_name": "BAAI/bge-reranker-large",
    "rerank_top_k": 5,
    
    # Prompt settings
    "prompt_strategy": "cot_few_shot",  # standard, few_shot, cot, or cot_few_shot (combined approach)
    "prompt_templates": {
        "standard": """Answer the following question directly and factually based solely on the provided information. If the information doesn't contain the answer, say "I don't have enough information to answer this question."

Information:
{context}

Question: {query}

Answer:""",
        "few_shot": """Answer the following question directly and factually. DO NOT use phrases like "the context," "the information provided," or "based on the document." If you don't have enough information, say "I don't have enough information to answer this question."

Information:
{context}

Here are examples of good answers:
Question: What are macronutrients?
Answer: Macronutrients are nutrients that the body requires in large amounts, specifically carbohydrates, proteins, and fats. They provide energy and are essential for various bodily functions.

Question: What's the nutritional difference between white and brown rice?
Answer: Brown rice contains more fiber, vitamins, and minerals than white rice because it retains the bran and germ layers that are removed in white rice processing. Brown rice has a lower glycemic index and provides more B vitamins, iron, and magnesium.

Question: What are the best protein sources for longevity?
Answer: I don't have enough information to answer this question.

Question: {query}

Answer:""",
        "cot_few_shot": """Answer directly and factually without referring to sources. Present your answer as established knowledge.

Information:
{context}

Question: {query}

I'll analyze this systematically:

First, let me identify the key facts about both protein types:
- Animal proteins: [list specific nutrients, qualities, and characteristics]
- Plant proteins: [list specific nutrients, qualities, and characteristics]

Next, I'll note important differences:
- Digestibility: [compare digestibility]
- Nutrient profiles: [compare nutrients]
- Health considerations: [note any health benefits or concerns]

Examples of good answers:
Question: What's the difference between saturated and unsaturated fats?
Answer: Saturated fats have no double bonds between carbon atoms and tend to be solid at room temperature. They're found mainly in animal products and can raise LDL cholesterol levels. Unsaturated fats contain at least one double bond and are typically liquid at room temperature. They're found in plant oils, nuts, seeds, and fish, and can help improve cholesterol levels and reduce inflammation.

Question: Which cooking oil has the highest smoke point?
Answer: Refined avocado oil has the highest smoke point at approximately 520°F (270°C), making it excellent for high-heat cooking methods like deep-frying and searing. Refined safflower, sunflower, and peanut oils also have high smoke points above 450°F.

Final Answer:"""
    }
}

def setup_device():
    """
    Set up device for computation based on availability
    """
    if CONFIG["device"] != "cpu" and not (CONFIG["device"] == "cuda" or CONFIG["device"] == "mps"):
        # Auto-detect
        if torch.cuda.is_available():
            CONFIG["device"] = "cuda"
            print("[INFO] Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            CONFIG["device"] = "mps"
            print("[INFO] Using MPS device (Apple Silicon)")
        else:
            CONFIG["device"] = "cpu"
            print("[INFO] Using CPU device")
    else:
        print(f"[INFO] Using {CONFIG['device']} device")
        
def save_config(filename="config.json"):
    """
    Save configuration to a JSON file
    
    Args:
        filename: Name of the file to save to
    """
    with open(filename, "w") as f:
        json.dump(CONFIG, f, indent=4)
        
def load_config(filename="config.json"):
    """
    Load configuration from a JSON file
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        dict: Loaded configuration
    """
    if not os.path.exists(filename):
        print(f"[ERROR] Configuration file {filename} not found.")
        return {}
    
    with open(filename, "r") as f:
        config = json.load(f)
    
    return config

# Set up the device on module import
setup_device() 