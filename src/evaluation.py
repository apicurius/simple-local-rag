"""
Evaluation framework for the RAG pipeline
"""
import json
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from src.rag_pipeline import RAGPipeline
from src.config import CONFIG, logger

class RAGEvaluator:
    """
    Evaluator for the RAG pipeline
    """
    
    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        """
        Initialize the evaluator
        
        Args:
            pipeline: RAG pipeline to evaluate (if None, a new one will be created)
        """
        self.pipeline = pipeline if pipeline is not None else RAGPipeline()
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_test_set(self, test_file: str) -> List[Dict[str, Any]]:
        """
        Load a test set from a JSON file
        
        Args:
            test_file: Path to the test file (JSON)
            
        Returns:
            List[Dict]: List of test items
        """
        if not os.path.exists(test_file):
            logger.error(f"Test file {test_file} not found")
            return []
            
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            logger.info(f"Loaded {len(test_data)} test items from {test_file}")
            return test_data
        except Exception as e:
            logger.error(f"Error loading test file {test_file}: {e}")
            return []
            
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save evaluation results to a file
        
        Args:
            results: Evaluation results dictionary
            output_file: Path to save results
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            
    def evaluate_retrieval(self, question: str, relevant_chunks: List[str], top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval for a single question
        
        Args:
            question: Question to evaluate
            relevant_chunks: List of relevant chunk identifiers
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict: Retrieval metrics
        """
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mrr": 0.0,
            "latency": 0.0
        }
        
        # Measure retrieval time
        start_time = time.time()
        retrieved_results = self.pipeline.retrieve(question, top_k=top_k)
        metrics["latency"] = time.time() - start_time
        
        # If no relevant chunks or no retrieved results, return default metrics
        if not relevant_chunks or not retrieved_results:
            return metrics
            
        # Get retrieved chunk identifiers
        # For evaluation, we'll use a mapping from page numbers to chunk IDs
        # This is because our test set uses IDs that differ from page numbers
        # Try to load mapping file
        chunk_mapping = {}
        try:
            mapping_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/chunk_mapping.json')
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    chunk_mapping = json.load(f)
                logger.debug(f"Loaded chunk mapping: {chunk_mapping}")
        except Exception as e:
            logger.warning(f"Failed to load chunk mapping: {e}")
            
        retrieved_ids = []
        for i, chunk in enumerate(retrieved_results):
            # Get page number and convert to test set ID via mapping if possible
            if "page_number" in chunk:
                page_num = str(chunk["page_number"])
                # Check mapping in both directions
                chunk_id = None
                # Direct page to ID mapping
                if page_num in chunk_mapping:
                    chunk_id = str(chunk_mapping[page_num])
                # Reverse mapping (ID to page)
                else:
                    for test_id, mapped_page in chunk_mapping.items():
                        if str(mapped_page) == page_num:
                            chunk_id = test_id
                            break
                            
                # If no mapping found, use page number
                if chunk_id is None:
                    chunk_id = page_num
                retrieved_ids.append(chunk_id)
            # Fallback fields
            elif "id" in chunk:
                chunk_id = str(chunk["id"])
                retrieved_ids.append(chunk_id)
            elif "chunk_id" in chunk:
                chunk_id = str(chunk["chunk_id"])
                retrieved_ids.append(chunk_id)
            elif "index" in chunk:
                chunk_id = str(chunk["index"])
                retrieved_ids.append(chunk_id)
            else:
                # As last resort, use the index position
                chunk_id = str(i)
                retrieved_ids.append(chunk_id)
                
        logger.debug(f"Retrieved IDs: {retrieved_ids}")
        logger.debug(f"Relevant chunks: {relevant_chunks}")
        
        # Convert relevant_chunks to set for faster lookup
        relevant_set = set(relevant_chunks)
        
        # Calculate precision, recall, and F1
        relevant_retrieved = [id for id in retrieved_ids if id in relevant_set]
        if retrieved_ids:
            metrics["precision"] = len(relevant_retrieved) / len(retrieved_ids)
        if relevant_chunks:
            metrics["recall"] = len(relevant_retrieved) / len(relevant_chunks)
        
        if metrics["precision"] > 0 or metrics["recall"] > 0:
            metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        
        # Calculate MRR (Mean Reciprocal Rank)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                metrics["mrr"] = 1.0 / (i + 1)
                break
                
        return metrics
        
    def evaluate_generation(self, question: str, reference_answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate generation for a single question
        
        Args:
            question: Question to evaluate
            reference_answer: Reference answer
            context: Retrieved context
            
        Returns:
            Dict: Generation metrics
        """
        metrics = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "latency": 0.0
        }
        
        # Measure generation time
        start_time = time.time()
        generated_answer = self.pipeline.generate(question, context)
        metrics["latency"] = time.time() - start_time
        
        # Generate our own reference answer if needed
        if not reference_answer or CONFIG.get("create_reference_answers", False):
            # Use a more verbose prompt to get a comprehensive reference answer
            logger.info(f"Creating reference answer for: {question}")
            try:
                reference_template = (
                    "Answer the following question comprehensively, accurately, and in detail, "
                    "covering all important aspects in a detailed response. Question: {query}\n\n"
                    "Context: {context}\n\nComprehensive Answer:"
                )
                
                # Create a modified version of the pipeline for reference generation
                ref_pipeline = self.pipeline
                original_prompt_strategy = ref_pipeline.config.get("prompt_strategy", "standard") 
                original_extract_answer = ref_pipeline.config.get("extract_answer", False)
                
                # Temporarily modify pipeline config
                ref_pipeline.config["prompt_strategy"] = "standard"
                ref_pipeline.config["extract_answer"] = False
                
                # Generate the reference answer using the context
                reference_answer = ref_pipeline.generate(question, context)
                
                # Restore original config
                ref_pipeline.config["prompt_strategy"] = original_prompt_strategy
                ref_pipeline.config["extract_answer"] = original_extract_answer
                
                logger.info(f"Generated reference answer: {reference_answer[:100]}...")
                
                # If there's an existing reference, create a combined version
                if reference_answer and CONFIG.get("use_multiple_references", True):
                    metrics["generated_reference"] = reference_answer
            except Exception as e:
                logger.warning(f"Error generating reference answer: {e}")
        
        # Calculate ROUGE scores with the reference answer
        if reference_answer and generated_answer:
            try:
                rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
                metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
                metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
                metrics["rougeL"] = rouge_scores["rougeL"].fmeasure
                
                # Add semantic similarity metric if possible
                try:
                    # Use the embedding manager we already have
                    if hasattr(self.pipeline, "embedding_manager") and self.pipeline.embedding_manager:
                        embed_mgr = self.pipeline.embedding_manager
                        ref_embedding = embed_mgr.get_embedding_for_text(reference_answer)
                        gen_embedding = embed_mgr.get_embedding_for_text(generated_answer)
                        
                        # Calculate cosine similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        semantic_sim = float(cosine_similarity([ref_embedding], [gen_embedding])[0][0])
                        metrics["semantic_similarity"] = semantic_sim
                except Exception as e:
                    logger.warning(f"Error calculating semantic similarity: {e}")
                    
            except Exception as e:
                logger.warning(f"Error calculating ROUGE scores: {e}")
                
        return metrics, generated_answer
        
    def evaluate_test_set(self, test_data: List[Dict[str, Any]], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline on a test set
        
        Args:
            test_data: List of test items
            output_file: Path to save results (optional)
            
        Returns:
            Dict: Evaluation results
        """
        # Initialize results structure
        results = {
            "overall": {
                "retrieval": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "mrr": 0.0,
                    "latency": 0.0
                },
                "generation": {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "semantic_similarity": 0.0,
                    "latency": 0.0
                },
                "total_latency": 0.0
            },
            "per_question": []
        }
        
        # Ensure pipeline is ready
        if not self.pipeline.retriever or not self.pipeline.embeddings:
            logger.info("Loading embeddings for evaluation")
            self.pipeline.load_embeddings()
        
        # Evaluate each question
        logger.info(f"Starting evaluation of {len(test_data)} test items")
        for i, item in enumerate(test_data):
            question = item["question"]
            reference_answer = item.get("reference_answer", "")
            relevant_chunks = item.get("relevant_chunks", [])
            
            logger.info(f"Evaluating question {i+1}/{len(test_data)}: {question[:50]}...")
            
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(question, relevant_chunks)
            
            # Get context for generation
            results_for_context = self.pipeline.retrieve(question, top_k=5)
            context = self.pipeline.retriever.format_results(results_for_context)
            
            # Evaluate generation
            generation_metrics, generated_answer = self.evaluate_generation(question, reference_answer, context)
            
            # Calculate total latency
            total_latency = retrieval_metrics["latency"] + generation_metrics["latency"]
            
            # Store per-question results
            question_result = {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "retrieval": retrieval_metrics,
                "generation": generation_metrics,
                "total_latency": total_latency
            }
            results["per_question"].append(question_result)
            
            # Update overall metrics (to be averaged later)
            for key in results["overall"]["retrieval"]:
                results["overall"]["retrieval"][key] += retrieval_metrics[key]
            
            for key in results["overall"]["generation"]:
                results["overall"]["generation"][key] += generation_metrics[key]
                
            results["overall"]["total_latency"] += total_latency
            
        # Calculate averages
        num_questions = len(test_data)
        if num_questions > 0:
            for section in ["retrieval", "generation"]:
                for key in results["overall"][section]:
                    results["overall"][section][key] /= num_questions
                    
            results["overall"]["total_latency"] /= num_questions
            
        # Save results if output file provided
        if output_file:
            self.save_results(results, output_file)
            
        return results
        
    def run_evaluation(self, test_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete evaluation using a test file
        
        Args:
            test_file: Path to the test file
            output_file: Path to save results (optional)
            
        Returns:
            Dict: Evaluation results
        """
        # Load test data
        test_data = self.load_test_set(test_file)
        if not test_data:
            return {}
            
        # Run evaluation
        results = self.evaluate_test_set(test_data, output_file)
        
        # Print summary results
        logger.info("\n=== RAG Evaluation Results ===")
        logger.info(f"Test set: {test_file}")
        logger.info(f"Number of questions: {len(test_data)}")
        
        # Retrieval metrics
        r_metrics = results["overall"]["retrieval"]
        logger.info("\nRetrieval Metrics:")
        logger.info(f"  Precision: {r_metrics['precision']:.4f}")
        logger.info(f"  Recall: {r_metrics['recall']:.4f}")
        logger.info(f"  F1: {r_metrics['f1']:.4f}")
        logger.info(f"  MRR: {r_metrics['mrr']:.4f}")
        logger.info(f"  Latency: {r_metrics['latency']:.4f} seconds")
        
        # Generation metrics
        g_metrics = results["overall"]["generation"]
        logger.info("\nGeneration Metrics:")
        logger.info(f"  ROUGE-1: {g_metrics['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {g_metrics['rouge2']:.4f}")
        logger.info(f"  ROUGE-L: {g_metrics['rougeL']:.4f}")
        logger.info(f"  Latency: {g_metrics['latency']:.4f} seconds")
        
        # Overall metrics
        logger.info("\nOverall Metrics:")
        logger.info(f"  Total Latency: {results['overall']['total_latency']:.4f} seconds")
        
        return results