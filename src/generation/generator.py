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
        # Determine appropriate torch dtype based on device
        if device == "cuda" or device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        print(f"[INFO] Using {dtype} precision for LLM")
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=CONFIG["device_map"] if device == "cuda" else None,
            quantization_config=quantization_config,
            torch_dtype=dtype
        )
        
        # Move model to device if not using device_map
        if device != "cuda" or CONFIG["device_map"] is None:
            self.model.to(device)
        
        # Create a pipeline for text generation
        try:
            # For CUDA, use device index 0
            if device == "cuda":
                device_arg = 0
            # For MPS or CPU, pass the device string
            else:
                device_arg = device
                
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_arg
            )
            print(f"[INFO] Successfully created pipeline on {device}")
        except Exception as e:
            print(f"[WARNING] Error creating pipeline on {device}: {e}")
            print("[WARNING] Falling back to CPU for pipeline")
            # Fall back to CPU
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cpu"
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
    
    def generate(self, query: str, context: str, streaming: bool = None) -> str:
        """
        Generate an answer for a query based on the provided context
        
        Args:
            query: Query string
            context: Context string
            streaming: Whether to enable streaming generation (overrides config)
            
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
        
        # Determine streaming mode
        use_streaming = streaming if streaming is not None else CONFIG.get("streaming", False)
        
        # Generate answer
        if use_streaming:
            return self._generate_streaming(prompt, max_new_tokens, temperature, top_p, top_k)
        else:
            return self._generate_standard(prompt, max_new_tokens, temperature, top_p, top_k)
    
    def _generate_standard(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        """
        Standard non-streaming generation
        
        Args:
            prompt: The prompt to use
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            str: Generated answer
        """
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
        
        # Apply additional answer extraction if configured
        if CONFIG.get("extract_answer", False):
            answer = self._extract_answer(answer, CONFIG.get("prompt_strategy", "standard"))
        
        return answer
        
    def _generate_streaming(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        """
        Streaming generation with token-by-token output
        
        Args:
            prompt: The prompt to use
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            str: Generated answer
        """
        # Import necessary for TextIteratorStreamer
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Create a streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate in a separate thread
        generation_kwargs = {
            "input_ids": self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Collect streamed output
        generated_text = []
        for token in streamer:
            generated_text.append(token)
            # Print token in real-time (with flush to ensure immediate display)
            print(token, end="", flush=True)
        
        # Print a newline at the end
        print()
        
        # Join all tokens into the final text
        full_text = prompt + "".join(generated_text)
        
        # Extract answer using the same function as non-streaming
        answer = self._extract_final_answer(full_text, prompt)
        
        # Apply additional answer extraction if configured
        if CONFIG.get("extract_answer", False):
            answer = self._extract_answer(answer, CONFIG.get("prompt_strategy", "standard"))
        
        return answer
        
    def _extract_final_answer(self, output_text: str, prompt: str) -> str:
        """
        Extract the final answer from the generated text
        
        Args:
            output_text: Full generated text
            prompt: Original prompt
            
        Returns:
            str: Extracted answer
        """
        # For simplicity, remove the prompt portion
        answer = output_text[len(prompt):].strip()
        
        # Remove any special tokens or markers
        answer = answer.replace("<s>", "").replace("</s>", "").strip()
        
        return answer
        
    def _extract_answer(self, full_text: str, prompt_strategy: str = "standard") -> str:
        """
        Extract the final clean answer from generated text based on prompt strategy
        
        Args:
            full_text: The full generated text including reasoning
            prompt_strategy: Which prompt strategy was used
            
        Returns:
            str: Clean extracted answer
        """
        # Different extraction strategies based on prompt type
        if prompt_strategy == "cot" or prompt_strategy == "cot_few_shot":
            # For chain-of-thought, look for "Final Answer:" marker
            if "Final Answer:" in full_text:
                return full_text.split("Final Answer:")[1].strip()
                
            # Alternative markers
            for marker in ["Therefore,", "In conclusion,", "To summarize,", "In summary,"]:
                if marker in full_text:
                    parts = full_text.split(marker)
                    if len(parts) > 1:
                        return marker + parts[1].strip()
            
            # If we can't find markers, return the last paragraph
            paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
            if paragraphs:
                return paragraphs[-1].strip()
                
        # For standard and few-shot prompts, just return the full answer
        return full_text.strip()