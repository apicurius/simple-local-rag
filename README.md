# Simple Local RAG

A straightforward, robust implementation of Retrieval-Augmented Generation (RAG) that works entirely locally without requiring cloud services.

!["This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources."](images/simple-local-rag-workflow-flowchart.png)

## Features

- **Fully Local**: All processing is done on your machine - no API keys or external services required
- **PDF Document Processing**: Extract and chunk text from PDF documents
- **Embeddings**: Create and store text embeddings using state-of-the-art models
- **Advanced Retrieval**: Multiple retrieval strategies including hybrid search and reranking
- **Improved Prompting**: Combined chain-of-thought and few-shot prompting for better responses
- **Optimized Scoring**: Better normalization techniques for more accurate retrieval

## Components

- **Document Processing**: Extract text from PDFs and create chunks
- **Embedding**: Create and manage embeddings using Sentence Transformers
- **Retrieval**: Retrieve relevant context using semantic search, keyword search, or hybrid approaches
- **Generation**: Generate responses using language models with optimized prompting

## Enhanced Features

- **Score Normalization**: Improved score normalization using min-max and sigmoid techniques
- **Combined Prompting**: Merged chain-of-thought and few-shot prompting for better responses
- **Adaptive Retrieval**: System can adjust retrieval strategy based on query complexity
- **Better Reranking**: Enhanced reranking with improved score calibration

## Usage

```bash
# Process a PDF document
python main.py process --pdf your_document.pdf

# Create embeddings
python main.py embed

# Query the RAG pipeline
python main.py query "Your question here?"

# Run in interactive mode
python main.py query --interactive

# Configure RAG strategies
python main.py strategy --prompt-strategy cot_few_shot
```

## Setup

### Clone repo
```
git clone https://github.com/apicurius/simple-local-rag.git
cd simple-local-rag
```

### Create environment
```
python -m venv .venv
```

### Activate environment
Linux/macOS:
```
source .venv/bin/activate
```

Windows: 
```
.\.venv\Scripts\activate
```

### Install requirements
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Note:** For PyTorch with CUDA, see: https://pytorch.org/get-started/locally/

On Windows use:
```
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## What is RAG?

RAG stands for Retrieval Augmented Generation.

Each step can be roughly broken down to:

* **Retrieval** - Seeking relevant information from a source given a query. For example, getting relevant passages from a PDF.
* **Augmented** - Using the relevant retrieved information to modify an input to a generative model (e.g. an LLM).
* **Generation** - Generating an output given an input. For example, in the case of an LLM, generating a passage of text given an input prompt.

## Why RAG?

The main goal of RAG is to improve the generation outputs of LLMs:

1. **Preventing hallucinations** - RAG pipelines can help LLMs generate more factual outputs by providing them with factual (retrieved) inputs.
2. **Work with custom data** - RAG systems can provide LLMs with domain-specific data such as medical information or company documentation and thus customize their outputs to suit specific use cases.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Sentence Transformers
- scikit-learn
- numpy
- spaCy

## License

MIT License