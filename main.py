#!/usr/bin/env python
"""
Command-line interface for the Simple Local RAG project
"""
import os
import sys
import argparse
from src.config import CONFIG, setup_device, save_config, load_config, create_config_profile, load_config_profile
from src.rag_pipeline import RAGPipeline

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Simple Local RAG Pipeline")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process document command
    process_parser = subparsers.add_parser("process", help="Process a PDF document")
    process_parser.add_argument("--pdf", type=str, help="Path to PDF file")
    process_parser.add_argument("--offset", type=int, default=41, help="Page offset")
    
    # Create embeddings command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings from processed text")
    embed_parser.add_argument("--model", type=str, help="Name of the embedding model")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("query", type=str, nargs="?", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=CONFIG.get("num_chunks_to_retrieve", 5), 
                            help="Number of chunks to retrieve")
    query_parser.add_argument("--include-scores", action="store_true", 
                            help="Include similarity scores in output")
    query_parser.add_argument("--hide-context", action="store_true", 
                            help="Hide the context in the output (context is shown by default)")
    query_parser.add_argument("--interactive", action="store_true", 
                            help="Run in interactive mode")
    query_parser.add_argument("--adaptive", action="store_true",
                           help="Use adaptive query strategy in interactive mode")
    query_parser.add_argument("--stream", action="store_true",
                           help="Enable streaming generation (token-by-token output)")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure the pipeline")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--save", type=str, help="Save configuration to file")
    config_parser.add_argument("--load", type=str, help="Load configuration from file")
    config_parser.add_argument("--clear-device", action="store_true", 
                             help="Clear device setting (use auto-detection)")
    config_parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                             help="Set device (cpu, cuda, mps)")
    config_parser.add_argument("--model", type=str, help="Set LLM model name")
    config_parser.add_argument("--embedding-model", type=str, help="Set embedding model name")
    config_parser.add_argument("--pdf", type=str, help="Set PDF path")
    config_parser.add_argument("--load-8bit", action="store_true", help="Enable 8-bit quantization")
    config_parser.add_argument("--load-4bit", action="store_true", help="Enable 4-bit quantization")
    config_parser.add_argument("--profile", type=str, help="Load a configuration profile")
    config_parser.add_argument("--save-profile", type=str, help="Save current configuration as a named profile")
    
    # RAG strategy command
    strategy_parser = subparsers.add_parser("strategy", help="Configure RAG strategy")
    strategy_parser.add_argument("--reranking", type=str, choices=["on", "off"], 
                              help="Turn reranking on or off")
    strategy_parser.add_argument("--hybrid-search", type=str, choices=["on", "off"], 
                              help="Turn hybrid search on or off")
    strategy_parser.add_argument("--query-expansion", type=str, choices=["on", "off"], 
                              help="Turn query expansion on or off")
    strategy_parser.add_argument("--prompt-strategy", type=str, 
                              choices=["standard", "few_shot", "cot", "cot_few_shot"], 
                              help="Set prompt strategy (standard, few_shot, cot, cot_few_shot)")
    
    return parser.parse_args()

def process_document(args):
    """Process a document and extract text"""
    # Create RAG pipeline
    pipeline = RAGPipeline()
    
    # Process document
    pdf_path = args.pdf if args.pdf else CONFIG["pdf_path"]
    pipeline.process_document(pdf_path=pdf_path, page_offset=args.offset)
    
    return pipeline

def create_embeddings(args):
    """Create embeddings from processed text"""
    # Create RAG pipeline
    pipeline = RAGPipeline()
    
    # Create embeddings
    model_name = args.model if args.model else CONFIG["embedding_model_name"]
    pipeline.create_embeddings(model_name=model_name)
    
    return pipeline

def run_query(args):
    """Run a query through the RAG pipeline"""
    # Create RAG pipeline
    pipeline = RAGPipeline()
    
    # Load embeddings
    pipeline.load_embeddings()
    
    # Interactive mode
    if args.interactive:
        print("Interactive mode. Type 'exit' or 'quit' to exit.")
        print("Type 'reset' to reset the pipeline.")
        print("Type 'adaptive on/off' to toggle adaptive query mode.")
        
        adaptive_mode = args.adaptive
        if adaptive_mode:
            print("[INFO] Adaptive query mode is enabled.")
        else:
            print("[INFO] Adaptive query mode is disabled.")
        
        while True:
            try:
                query = input("\nEnter your query: ")
                
                if query.lower() in ["exit", "quit"]:
                    break
                
                if query.lower() == "reset":
                    print("Resetting pipeline...")
                    pipeline = RAGPipeline()
                    pipeline.load_embeddings()
                    continue
                
                if query.lower().startswith("adaptive"):
                    parts = query.lower().split()
                    if len(parts) > 1:
                        if parts[1] == "on":
                            adaptive_mode = True
                            print("[INFO] Adaptive query mode is enabled.")
                        elif parts[1] == "off":
                            adaptive_mode = False
                            print("[INFO] Adaptive query mode is disabled.")
                    else:
                        print(f"[INFO] Adaptive query mode is {'enabled' if adaptive_mode else 'disabled'}.")
                    continue
                    
                if adaptive_mode:
                    answer, context = pipeline.query_adaptive(query, top_k=args.top_k, 
                                                  include_scores=args.include_scores, 
                                                  include_context=not args.hide_context,
                                                  streaming=args.stream)
                else:
                    answer = pipeline.query(query, top_k=args.top_k, 
                                         include_scores=args.include_scores,
                                         include_context=not args.hide_context,
                                         streaming=args.stream)
                
                print(f"\nAnswer: {answer}")
                if not args.hide_context:
                    print("\nContext:")
                    print(context)
                    
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    else:
        if not args.query:
            print("Error: Query text is required.")
            return None
        
        # Configure streaming based on arguments
        CONFIG["streaming"] = args.stream
        
        if args.hide_context:
            # Get only the answer, not the context
            answer = pipeline.query(args.query, top_k=args.top_k, 
                                   include_scores=args.include_scores, 
                                   include_context=False,
                                   streaming=args.stream)
            if not args.stream:  # Only print answer if not streaming (streaming already prints)
                print(f"{answer}")
        else:
            # Get both answer and context
            answer, context = pipeline.query(args.query, top_k=args.top_k, 
                                           include_scores=args.include_scores, 
                                           include_context=True,
                                           streaming=args.stream)
            if not args.stream:  # Only print answer if not streaming
                print(f"Answer: {answer}")
            print("\nContext used for generation:")
            print(context)
    
    return pipeline

def config_pipeline(args):
    """Configure the pipeline"""
    if args.show:
        # Show current configuration
        print("Current configuration:")
        for key, value in CONFIG.items():
            if key != "prompt_templates":  # Skip showing templates (too verbose)
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: [Template definitions available]")
    
    # Load configuration profile if requested
    if args.profile:
        profile_config = load_config_profile(args.profile)
        if profile_config:
            CONFIG.update(profile_config)
            print(f"[INFO] Configuration profile '{args.profile}' loaded")
            # Re-setup device after loading profile
            setup_device()
    
    # Update configuration based on arguments
    if args.device:
        CONFIG["device"] = args.device
        print(f"[INFO] Device set to {args.device}")
    
    if args.clear_device:
        CONFIG["device"] = "cpu"  # Will be updated by setup_device()
        print("[INFO] Device setting cleared")
    
    if args.model:
        CONFIG["llm_model_name"] = args.model
        print(f"[INFO] LLM model set to {args.model}")
    
    if args.embedding_model:
        CONFIG["embedding_model_name"] = args.embedding_model
        print(f"[INFO] Embedding model set to {args.embedding_model}")
    
    if args.pdf:
        CONFIG["pdf_path"] = args.pdf
        print(f"[INFO] PDF path set to {args.pdf}")
    
    if args.load_8bit:
        CONFIG["load_in_8bit"] = True
        CONFIG["load_in_4bit"] = False
        print("[INFO] 8-bit quantization enabled")
    
    if args.load_4bit:
        CONFIG["load_in_4bit"] = True
        CONFIG["load_in_8bit"] = False
        print("[INFO] 4-bit quantization enabled")
        
    # Setup device (auto-detect if needed)
    setup_device()
    
    # Save configuration if requested
    if args.save:
        save_config(args.save)
        print(f"[INFO] Configuration saved to {args.save}")
    
    # Load configuration if requested
    if args.load:
        loaded_config = load_config(args.load)
        CONFIG.update(loaded_config)
        print(f"[INFO] Configuration loaded from {args.load}")
        # Re-setup device after loading config
        setup_device()
        
    # Save as profile if requested
    if args.save_profile:
        create_config_profile(args.save_profile, CONFIG)
        print(f"[INFO] Current configuration saved as profile '{args.save_profile}'")

def config_strategy(args):
    """Configure RAG strategy"""
    changes = False
    
    if args.reranking:
        CONFIG["use_reranking"] = (args.reranking == "on")
        changes = True
        
    if args.hybrid_search:
        CONFIG["use_hybrid_search"] = (args.hybrid_search == "on")
        changes = True
        
    if args.query_expansion:
        CONFIG["use_query_expansion"] = (args.query_expansion == "on")
        changes = True
        
    if args.prompt_strategy:
        CONFIG["prompt_strategy"] = args.prompt_strategy
        changes = True
    
    if changes:
        print("[INFO] RAG strategy updated:")
        print(f"  - Reranking: {CONFIG.get('use_reranking', False)}")
        print(f"  - Hybrid search: {CONFIG.get('use_hybrid_search', False)}")
        print(f"  - Prompt strategy: {CONFIG.get('prompt_strategy', 'standard')}")
        print(f"  - Query expansion: {CONFIG.get('use_query_expansion', False)}")

def main():
    """Main function"""
    args = parse_args()
    
    # Setup device (auto-detect if needed)
    setup_device()
    
    # Process command
    if args.command == "process":
        pipeline = process_document(args)
    elif args.command == "embed":
        pipeline = create_embeddings(args)
    elif args.command == "query":
        pipeline = run_query(args)
    elif args.command == "config":
        config_pipeline(args)
    elif args.command == "strategy":
        config_strategy(args)
    else:
        print("Error: Command is required.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 