#!/usr/bin/env python
"""
Run evaluation on the RAG pipeline
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime
from src.evaluation import RAGEvaluator
from src.rag_pipeline import RAGPipeline
from src.config import CONFIG, setup_device, logger

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline")
    
    parser.add_argument("--test-file", type=str, default="data/test_questions.json",
                        help="Path to test questions file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--profile", type=str, default=None,
                        help="Configuration profile to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    
    # Initialize pipeline with debug logging
    pipeline = RAGPipeline()
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(pipeline)
    
    # Run evaluation
    results = evaluator.run_evaluation(args.test_file, output_file)
    
    # Print retrieval debug info
    print("\nDEBUG: Chunk Mapping Information")
    mapping_file = os.path.join('data', 'chunk_mapping.json')
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            print(f"Mapping loaded with {len(mapping)} entries: {mapping}")
    except Exception as e:
        print(f"Error loading mapping: {e}")
        
    # Debug sample retrieval
    try:
        # Use a question from our test set
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
            if test_data:
                test_query = test_data[0]["question"]
                relevant_ids = test_data[0]["relevant_chunks"]
                print(f"\nTesting retrieval for: {test_query}")
                print(f"Expected relevant chunks: {relevant_ids}")
                
                # Get retrieval results
                retrieval_results = pipeline.retrieve(test_query, top_k=5)
                retrieved_ids = [str(r.get("page_number", "N/A")) for r in retrieval_results]
                print(f"Retrieved page numbers: {retrieved_ids}")
                
                # Print example structure
                print("\nSample retrieval result structure:")
                print(json.dumps(retrieval_results[0], default=str, indent=2))
    except Exception as e:
        print(f"Debug error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())