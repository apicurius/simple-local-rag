"""
Configuration settings for the Simple Local RAG project
"""
import os
import json
import torch
import logging
from typing import Dict, Any, List, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple-local-rag")

# Default configuration dictionary
DEFAULT_CONFIG = {
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
    "chunk_overlap": 3,      # Number of sentences to overlap between chunks
    
    # Embedding settings
    "embedding_model_name": "BAAI/bge-large-en-v1.5",
    "embeddings_filename": "nutrition_text_embeddings.json",
    "use_embedding_cache": True,
    "use_vector_db": False,
    "vector_db_config": {
        "index_type": "flat",  # "flat" or "ivf"
        "n_clusters": 100,     # Only used for IVF index
    },
    
    # LLM settings
    "llm_model_name": "google/gemma-3-4b-it",
    "llm_context_window": 8192,
    "max_new_tokens": 1024,
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 50,
    "streaming": False,        # Whether to enable streaming generation
    
    # Retrieval settings
    "num_chunks_to_retrieve": 10,
    "minimum_score_threshold": 0.10,
    "ensure_diverse_chunks": True,  # Whether to ensure diversity in retrieved chunks
    "diversity_threshold": 0.85,    # Similarity threshold for considering chunks too similar
    
    # Hybrid search settings
    "use_hybrid_search": True,
    "semantic_weight": 0.65,
    "keyword_weight": 0.35,
    
    # Query expansion settings
    "use_query_expansion": True,
    "expansion_factor": 2,
    
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

# Create global config by copying default config
CONFIG = DEFAULT_CONFIG.copy()

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters and set defaults for missing values
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        dict: Validated configuration with defaults for missing values
    """
    # Create a validated copy starting with defaults
    validated = DEFAULT_CONFIG.copy()
    
    # Update with provided config
    validated.update(config)
    
    # Validate specific parameters
    # Device settings
    if validated["device"] not in ["cpu", "cuda", "mps"]:
        logger.warning(f"Invalid device: {validated['device']}. Using auto-detection.")
        validated["device"] = "cpu"  # Will be updated by setup_device()
    
    # Chunk settings
    validated["chunk_size"] = max(1, validated["chunk_size"])
    validated["chunk_overlap"] = max(0, min(validated["chunk_overlap"], validated["chunk_size"] - 1))
    validated["min_token_length"] = max(10, validated["min_token_length"])
    
    # LLM settings
    validated["max_new_tokens"] = max(1, min(validated["max_new_tokens"], 4096))
    validated["temperature"] = max(0.0, min(validated["temperature"], 2.0))
    validated["top_p"] = max(0.0, min(validated["top_p"], 1.0))
    validated["top_k"] = max(1, validated["top_k"])
    
    # Retrieval settings
    validated["num_chunks_to_retrieve"] = max(1, validated["num_chunks_to_retrieve"])
    validated["minimum_score_threshold"] = max(0.0, min(validated["minimum_score_threshold"], 1.0))
    
    # Weights settings
    if validated["use_hybrid_search"]:
        total_weight = validated["semantic_weight"] + validated["keyword_weight"]
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Semantic and keyword weights should sum to 1.0. Current sum: {total_weight}")
            # Normalize weights
            validated["semantic_weight"] = validated["semantic_weight"] / total_weight
            validated["keyword_weight"] = validated["keyword_weight"] / total_weight
    
    # Verify templates
    if validated["prompt_strategy"] not in validated["prompt_templates"]:
        logger.warning(f"Prompt strategy '{validated['prompt_strategy']}' not found in templates. Using 'standard' instead.")
        validated["prompt_strategy"] = "standard"
    
    return validated

def setup_device():
    """
    Set up device for computation based on availability
    """
    if CONFIG["device"] != "cpu" and not (CONFIG["device"] == "cuda" or CONFIG["device"] == "mps"):
        # Auto-detect
        if torch.cuda.is_available():
            CONFIG["device"] = "cuda"
            logger.info("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            CONFIG["device"] = "mps"
            logger.info("Using MPS device (Apple Silicon)")
        else:
            CONFIG["device"] = "cpu"
            logger.info("Using CPU device")
    else:
        logger.info(f"Using {CONFIG['device']} device")
        
def save_config(filename="config.json"):
    """
    Save configuration to a JSON file
    
    Args:
        filename: Name of the file to save to
    """
    with open(filename, "w") as f:
        json.dump(CONFIG, f, indent=4)
    logger.info(f"Configuration saved to {filename}")
        
def load_config(filename="config.json"):
    """
    Load configuration from a JSON file
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        dict: Loaded configuration
    """
    if not os.path.exists(filename):
        logger.error(f"Configuration file {filename} not found.")
        return {}
    
    with open(filename, "r") as f:
        loaded_config = json.load(f)
    
    # Validate loaded config
    validated_config = validate_config(loaded_config)
    logger.info(f"Configuration loaded and validated from {filename}")
    
    return validated_config
    
def create_config_profile(profile_name: str, config_dict: Dict[str, Any]):
    """
    Create a named configuration profile
    
    Args:
        profile_name: Name of the profile
        config_dict: Configuration dictionary
    """
    # Create profiles directory if it doesn't exist
    profiles_dir = os.path.join(os.path.dirname(__file__), "..", "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Validate config
    validated_config = validate_config(config_dict)
    
    # Save to file
    profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
    with open(profile_path, "w") as f:
        json.dump(validated_config, f, indent=4)
        
    logger.info(f"Configuration profile '{profile_name}' saved to {profile_path}")
    
def load_config_profile(profile_name: str) -> Dict[str, Any]:
    """
    Load a named configuration profile
    
    Args:
        profile_name: Name of the profile
        
    Returns:
        dict: Loaded and validated configuration profile
    """
    profiles_dir = os.path.join(os.path.dirname(__file__), "..", "profiles")
    profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
    
    if not os.path.exists(profile_path):
        logger.error(f"Configuration profile '{profile_name}' not found.")
        return {}
    
    with open(profile_path, "r") as f:
        loaded_config = json.load(f)
    
    # Validate loaded config
    validated_config = validate_config(loaded_config)
    logger.info(f"Configuration profile '{profile_name}' loaded from {profile_path}")
    
    return validated_config

# Set up the device on module import
setup_device() 