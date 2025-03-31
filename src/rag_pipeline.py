"""
Main RAG (Retrieval Augmented Generation) pipeline for the Simple Local RAG project
"""
import os
from typing import List, Dict, Any, Optional, Union, Tuple

from src.config import CONFIG
from src.utils.pdf_utils import download_pdf, open_and_read_pdf
from src.utils.text_utils import process_sentences, create_sentence_chunks, process_chunks_to_items
from src.embedding.embedding import EmbeddingManager
from src.retrieval.retrieval import Retriever
from src.generation.generator import Generator

class RAGPipeline:
    """
    Main RAG pipeline class that combines document processing, embedding, retrieval, and generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RAG pipeline
        
        Args:
            config: Configuration dictionary (default: use global CONFIG)
        """
        # Use global config if not provided
        self.config = CONFIG.copy() if config is None else config
        
        # Initialize components to None
        self.embedding_manager = None
        self.retriever = None
        self.generator = None
        
        # Initialize document data to None
        self.text_chunks = None
    
    def initialize_embedding_manager(self):
        """Initialize the embedding manager"""
        self.embedding_manager = EmbeddingManager(
            model_name=self.config["embedding_model_name"],
            device=self.config["device"]
        )
        return self.embedding_manager
    
    def initialize_retriever(self):
        """Initialize the retriever"""
        if self.embedding_manager is None:
            self.initialize_embedding_manager()
            
        self.retriever = Retriever(embedding_manager=self.embedding_manager)
        return self.retriever
    
    def initialize_generator(self):
        """Initialize the generator"""
        self.generator = Generator(
            model_name=self.config["llm_model_name"],
            device=self.config["device"],
            load_in_8bit=self.config["load_in_8bit"],
            load_in_4bit=self.config["load_in_4bit"]
        )
        return self.generator
    
    def process_document(self, pdf_path: str = None, page_offset: int = 0):
        """
        Process a PDF document
        
        Args:
            pdf_path: Path to the PDF file
            page_offset: Page offset to use
        """
        if pdf_path is None:
            pdf_path = CONFIG["pdf_path"]
        
        # Check if PDF exists, download if not
        if not os.path.exists(pdf_path):
            print(f"[INFO] PDF not found at {pdf_path}. Downloading...")
            download_pdf(pdf_path)
        
        # Open and read PDF
        print(f"[INFO] Processing PDF: {pdf_path}")
        pages_and_texts = open_and_read_pdf(pdf_path, page_offset=page_offset)
        print(f"[INFO] Read {len(pages_and_texts)} pages from PDF")
        
        # Process sentences
        pages_and_sentences = process_sentences(pages_and_texts)
        print(f"[INFO] Processed sentences from {len(pages_and_sentences)} pages")
        
        # Create sentence chunks
        chunk_size = CONFIG["chunk_size"]
        chunk_overlap = CONFIG.get("chunk_overlap", 0)
        chunks = create_sentence_chunks(pages_and_sentences, chunk_size=chunk_size, overlap=chunk_overlap)
        print(f"[INFO] Created {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}")
        
        # Process chunks to items
        min_token_length = CONFIG["min_token_length"]
        self.text_chunks = process_chunks_to_items(chunks, min_token_length=min_token_length)
        print(f"[INFO] Processed {len(self.text_chunks)} chunks with minimum token length {min_token_length}")
        
        return self.text_chunks
    
    def create_embeddings(self, model_name: str = None, batch_size: int = 32):
        """
        Create embeddings for the processed document
        
        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for embedding creation
        """
        if model_name is None:
            model_name = CONFIG["embedding_model_name"]
            
        # Initialize embedding manager if not already done
        if self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager(model_name=model_name, device=CONFIG["device"])
        
        # Make sure we have text chunks
        if not self.text_chunks:
            print("[INFO] No text chunks found. Processing document...")
            self.process_document()
        
        # Create embeddings
        self.embedding_manager.embed_chunks(self.text_chunks, batch_size=batch_size)
        
        # Save embeddings
        self.save_embeddings()
        
        return self.embedding_manager.embeddings
    
    def save_embeddings(self, filename: str = None):
        """
        Save embeddings to a file
        
        Args:
            filename: Filename to save to (if None, uses config)
        """
        if filename is None:
            filename = CONFIG["embeddings_filename"]
            
        if self.embedding_manager is None or self.embedding_manager.embeddings is None:
            raise ValueError("No embeddings available. Run create_embeddings() first.")
            
        self.embedding_manager.save_embeddings(filename)
        
        return filename
    
    def load_embeddings(self, filename: str = None):
        """
        Load embeddings from a file
        
        Args:
            filename: Filename to load from (if None, uses config)
        """
        if filename is None:
            filename = CONFIG["embeddings_filename"]
            
        # Initialize embedding manager if not done yet
        if self.embedding_manager is None:
            model_name = CONFIG["embedding_model_name"]
            self.embedding_manager = EmbeddingManager(model_name=model_name, device=CONFIG["device"])
            
        # Load embeddings
        self.embedding_manager.load_embeddings(filename)
        self.text_chunks = self.embedding_manager.text_chunks
        
        print(f"[INFO] Loaded {len(self.embedding_manager.embeddings)} embeddings from {filename}")
        
        return self.embedding_manager.embeddings
    
    def setup_retrieval(self):
        """
        Set up the retriever component
        """
        # Make sure embeddings are loaded
        if self.embedding_manager is None or self.embedding_manager.embeddings is None:
            print("[INFO] No embeddings loaded. Loading embeddings...")
            self.load_embeddings()
            
        # Create retriever
        self.retriever = Retriever(
            embedding_manager=self.embedding_manager,
            embeddings=self.embedding_manager.embeddings,
            chunks=self.text_chunks
        )
        
        return self.retriever
    
    def setup_generation(self):
        """
        Set up the generator component
        """
        # Create generator
        self.generator = Generator(
            model_name=CONFIG["llm_model_name"],
            device=CONFIG["device"],
            load_in_8bit=CONFIG.get("load_in_8bit", False),
            load_in_4bit=CONFIG.get("load_in_4bit", False)
        )
        
        return self.generator
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List[Dict]: List of retrieved chunks with similarity scores
        """
        if self.retriever is None:
            self.setup_retrieval()
            
        # Retrieve chunks
        results = self.retriever.search(query, top_k=top_k)
        
        return results
    
    def format_results(self, results: List[Dict[str, Any]], include_scores: bool = True) -> str:
        """
        Format retrieved results for display or inclusion in a prompt
        
        Args:
            results: List of dictionaries with chunks and scores
            include_scores: Whether to include similarity scores in the output
            
        Returns:
            str: Formatted results string
        """
        return self.retriever.format_results(results, include_scores=include_scores)
    
    def generate(self, query: str, context: str, streaming: bool = None) -> str:
        """
        Generate an answer for a query given a context
        
        Args:
            query: Query string
            context: Context string
            streaming: Whether to use streaming generation (overrides config)
            
        Returns:
            str: Generated answer
        """
        if self.generator is None:
            self.setup_generation()
            
        return self.generator.generate(query, context, streaming=streaming)
    
    def set_rag_strategy(self, 
                        use_reranking: bool = None, 
                        use_hybrid_search: bool = None,
                        prompt_strategy: str = None,
                        use_query_expansion: bool = None):
        """
        Set RAG strategies for future queries
        
        Args:
            use_reranking: Whether to use reranking
            use_hybrid_search: Whether to use hybrid search
            prompt_strategy: Prompt strategy to use (standard, few_shot, cot)
            use_query_expansion: Whether to use query expansion
            
        Returns:
            self: The RAG pipeline instance (for method chaining)
        """
        # Update config with specified values
        if use_reranking is not None:
            self.config["use_reranking"] = use_reranking
            
        if use_hybrid_search is not None:
            self.config["use_hybrid_search"] = use_hybrid_search
            
        if prompt_strategy is not None:
            if prompt_strategy not in ["standard", "few_shot", "cot"]:
                print(f"[WARNING] Invalid prompt strategy '{prompt_strategy}'. Using 'standard'.")
                prompt_strategy = "standard"
            self.config["prompt_strategy"] = prompt_strategy
            
        if use_query_expansion is not None:
            self.config["use_query_expansion"] = use_query_expansion
            
        print("[INFO] RAG strategy updated:")
        print(f"  - Reranking: {self.config.get('use_reranking', False)}")
        print(f"  - Hybrid search: {self.config.get('use_hybrid_search', False)}")
        print(f"  - Prompt strategy: {self.config.get('prompt_strategy', 'standard')}")
        print(f"  - Query expansion: {self.config.get('use_query_expansion', False)}")
        
        return self
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a query and suggest optimal RAG strategies
        
        Args:
            query: Query string
            
        Returns:
            Dict: Analysis results with suggested strategies
        """
        # Simple heuristic-based analysis
        word_count = len(query.split())
        char_count = len(query)
        
        # Initialize analysis result
        analysis = {
            "query": query,
            "word_count": word_count,
            "char_count": char_count,
            "complexity": "simple",
            "suggested_strategies": {}
        }
        
        # Determine query complexity
        if word_count > 15 or "why" in query.lower() or "how" in query.lower():
            analysis["complexity"] = "complex"
            analysis["suggested_strategies"]["prompt_strategy"] = "cot"
            analysis["suggested_strategies"]["use_reranking"] = True
        elif any(kw in query.lower() for kw in ["difference", "compare", "versus", "vs"]):
            analysis["complexity"] = "comparative"
            analysis["suggested_strategies"]["prompt_strategy"] = "cot"
            analysis["suggested_strategies"]["use_hybrid_search"] = True
        elif any(kw in query.lower() for kw in ["list", "benefits", "types", "examples"]):
            analysis["complexity"] = "list-based"
            analysis["suggested_strategies"]["prompt_strategy"] = "standard"
            analysis["suggested_strategies"]["use_query_expansion"] = True
        else:
            analysis["suggested_strategies"]["prompt_strategy"] = "standard"
            analysis["suggested_strategies"]["use_reranking"] = False
            
        return analysis
    
    def adaptive_query(self, query: str, top_k: int = None, include_scores: bool = False) -> str:
        """
        Adaptive RAG pipeline query that adapts strategies based on query complexity
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            include_scores: Whether to include similarity scores in context
            
        Returns:
            str: Generated answer
        """
        # Analyze query and adapt strategies
        analysis = self.analyze_query_complexity(query)
        self.set_rag_strategy(**analysis["suggested_strategies"])
        
        print(f"[INFO] Query analyzed as '{analysis['complexity']}'. Adapting RAG strategies.")
        
        # Execute the adapted query
        return self.query(query, top_k=top_k, include_scores=include_scores)
    
    def query(self, query: str, top_k: int = None, include_scores: bool = False, 
             include_context: bool = True, streaming: bool = None) -> Union[str, Tuple[str, str]]:
        """
        Process a query and generate a response
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            include_scores: Whether to include similarity scores in context
            include_context: Whether to return context along with the answer
            streaming: Whether to use streaming generation (overrides config)
            
        Returns:
            str or Tuple[str, str]: Generated answer (and context if include_context=True)
        """
        # Ensure resources are loaded
        self._ensure_resources_loaded()
        
        # Retrieve relevant chunks
        results = self.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.retriever.format_results(results, include_scores=include_scores)
        
        # Generate answer with streaming option
        answer = self.generate(query, context, streaming=streaming)
        
        if include_context:
            return answer, context
        else:
            return answer
            
    def query_adaptive(self, query: str, top_k: int = None, include_scores: bool = False,
                       include_context: bool = True, streaming: bool = None) -> Union[str, Tuple[str, str]]:
        """
        Adaptively process a query based on its complexity
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            include_scores: Whether to include similarity scores in context
            include_context: Whether to return context along with the answer
            streaming: Whether to use streaming generation (overrides config)
            
        Returns:
            str or Tuple[str, str]: Generated answer (and context if include_context=True)
        """
        # Ensure resources are loaded
        self._ensure_resources_loaded()
        
        # Analyze query complexity
        complexity = self._analyze_query_complexity(query)
        
        # Set RAG strategy based on query complexity
        self._adapt_rag_strategy(complexity, query)
        
        # Retrieve relevant chunks
        results = self.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.retriever.format_results(results, include_scores=include_scores)
        
        # Generate answer with appropriate strategy and streaming option
        answer = self.generate(query, context, streaming=streaming)
        
        if include_context:
            return answer, context
        else:
            return answer
            
    def _ensure_resources_loaded(self):
        """Ensure that all required resources (retriever, generator) are loaded"""
        if self.retriever is None:
            self.setup_retrieval()
        
        if self.generator is None:
            self.setup_generation() 