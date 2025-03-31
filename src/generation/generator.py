"""
Text generation functions for the Simple Local RAG project
"""
import torch
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.config import CONFIG

class Generator:
    """
    Generates responses using a language model based on a query and context
    """
    
    def __init__(self, 
                model_name: str = None, 
                device: str = None,
                load_in_8bit: bool = None,
                load_in_4bit: bool = None):
        """
        Initialize the generator with a model
        
        Args:
            model_name: Name of the language model to use
            device: Device to run the model on (cuda, cpu, mps)
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
        """
        if model_name is None:
            model_name = CONFIG["llm_model_name"]
        
        if device is None:
            device = CONFIG["device"]
            
        if load_in_8bit is None:
            load_in_8bit = CONFIG["load_in_8bit"]
            
        if load_in_4bit is None:
            load_in_4bit = CONFIG["load_in_4bit"]
        
        self.model_name = model_name
        self.device = device
        
        # Special handling for quantized models
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
                print(f"[INFO] Loading model with {'8-bit' if load_in_8bit else '4-bit'} quantization")
            except ImportError:
                print("[WARNING] bitsandbytes not available, loading model in full precision")
        
        print(f"[INFO] Loading LLM model '{model_name}' on {device}")
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=CONFIG["device_map"] if device == "cuda" else None,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Move model to device if not using device_map
        if device != "cuda" or CONFIG["device_map"] is None:
            self.model.to(device)
        
        # Create a pipeline for text generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else device
        )
    
    def _create_standard_prompt(self, query: str, context: str) -> str:
        """
        Create a standard prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        return f"""You are a helpful assistant answering questions about nutrition and human health. 
Use ONLY the provided context to answer the question, and if you don't know the answer, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    def _create_few_shot_prompt(self, query: str, context: str) -> str:
        """
        Create a few-shot prompt with examples
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            str: Formatted prompt with examples
        """
        few_shot_examples = CONFIG.get("few_shot_examples", [])
        if not few_shot_examples:
            return self._create_standard_prompt(query, context)
            
        examples_text = ""
        for i, example in enumerate(few_shot_examples):
            examples_text += f"\nExample {i+1}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n"
            
        return f"""You are a helpful assistant answering questions about nutrition and human health.
Use ONLY the provided context to answer the question, and if you don't know the answer, say so.

Here are some examples of good answers:
{examples_text}

Context:
{context}

Question: {query}

Answer:"""
    
    def _create_cot_prompt(self, query: str, context: str) -> str:
        """
        Create a chain-of-thought prompt that encourages step-by-step reasoning
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            str: Formatted chain-of-thought prompt
        """
        return f"""You are a helpful assistant answering questions about nutrition and human health.
Use ONLY the provided context to answer the question, and if you don't know the answer, say so.

To answer the question correctly, think through this step by step:
1. First, identify the key concepts in the question
2. Find relevant information from the context
3. Connect the information to answer the question
4. Provide a clear, concise answer

Context:
{context}

Question: {query}

Reasoning step by step:"""
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM based on the configured strategy
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        prompt_strategy = CONFIG.get("prompt_strategy", "standard")
        
        if prompt_strategy == "few_shot":
            return self._create_few_shot_prompt(query, context)
        elif prompt_strategy == "cot":
            return self._create_cot_prompt(query, context)
        elif prompt_strategy == "cot_few_shot":
            # Use the prompt template from config
            prompt_templates = CONFIG.get("prompt_templates", {})
            if "cot_few_shot" in prompt_templates:
                return prompt_templates["cot_few_shot"].format(context=context, query=query)
            else:
                # Fallback to standard COT if template not found
                return self._create_cot_prompt(query, context)
        else:  # default to standard
            return self._create_standard_prompt(query, context)
    
    def _extract_final_answer(self, full_response: str, prompt: str) -> str:
        """
        Extract the final answer from a chain-of-thought response
        
        Args:
            full_response: Full model response
            prompt: Original prompt
            
        Returns:
            str: Extracted final answer
        """
        # Remove the prompt from the response
        answer = full_response[len(prompt):].strip()
        
        # For chain-of-thought, try to extract just the conclusion/final answer
        prompt_strategy = CONFIG.get("prompt_strategy", "standard")
        
        if prompt_strategy in ["cot", "cot_few_shot"]:
            # Look for final answer indicators
            conclusion_indicators = [
                "Final Answer:", "Therefore,", "In conclusion,", "To summarize,", "The answer is",
                "In summary,", "Overall,", "Finally,", "To conclude,"
            ]
            
            # First try to find "Final Answer:" as it's most explicit
            if "Final Answer:" in answer:
                final_answer_idx = answer.find("Final Answer:")
                return answer[final_answer_idx + len("Final Answer:"):].strip()
            
            # Then try other conclusion indicators
            for indicator in conclusion_indicators:
                if indicator in answer:
                    # Extract everything after the indicator
                    conclusion_idx = answer.find(indicator)
                    if conclusion_idx > len(answer) // 3:  # Only if it's in the latter portion
                        return answer[conclusion_idx + len(indicator):].strip()
            
            # If no conclusion indicator found, try to get the last paragraph
            paragraphs = answer.split("\n\n")
            if len(paragraphs) > 1:
                # Return last non-empty paragraph
                for p in reversed(paragraphs):
                    if p.strip():
                        return p.strip()
        
        # Remove any "context describes" or similar phrases
        phrases_to_remove = [
            "Based on the context,", 
            "According to the context,",
            "The context describes",
            "The information provided",
            "Based on the provided information,",
            "Based on the information,",
            "From the context,"
        ]
        
        for phrase in phrases_to_remove:
            if answer.startswith(phrase):
                answer = answer[len(phrase):].strip()
                # Remove comma if it starts with one after phrase removal
                if answer.startswith(','):
                    answer = answer[1:].strip()
        
        return answer
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer for a query based on the provided context
        
        Args:
            query: Query string
            context: Context string
            
        Returns:
            str: Generated answer
        """
        # Get prompt strategy from config
        prompt_strategy = CONFIG.get("prompt_strategy", "standard")
        
        # Get prompt template based on strategy
        prompt_templates = CONFIG.get("prompt_templates", {})
        if prompt_strategy in prompt_templates:
            # Use template from config
            prompt_template = prompt_templates[prompt_strategy]
            prompt = prompt_template.format(context=context, query=query)
        else:
            # Use default prompt creation
            prompt = self.create_prompt(query, context)
        
        # Set generation parameters with fallbacks
        max_new_tokens = CONFIG.get("max_new_tokens", 512)
        temperature = CONFIG.get("temperature", 0.3)
        top_p = CONFIG.get("top_p", 0.9)
        top_k = CONFIG.get("top_k", 50)
        
        # Generate answer
        with torch.no_grad():
            output = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0
            )
        
        # Extract generated text
        answer = self._extract_final_answer(output[0]["generated_text"], prompt)
        
        return answer 