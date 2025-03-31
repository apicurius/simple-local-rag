# Simple Local RAG

A lightweight and extensible Retrieval-Augmented Generation (RAG) system for local documents and private data.

\![Simple Local RAG Workflow](images/simple-local-rag-workflow-flowchart.png)

## Features

- 📄 **Local PDF Processing**: Extract, chunk, and process text from PDF documents
- 🔍 **Hybrid Search**: Combine semantic (embedding) search with keyword search (BM25)
- 🔄 **Query Expansion**: Enhance queries with domain-specific terminology and WordNet synonyms
- 📊 **Results Diversification**: Ensure a balanced set of results covering different aspects
- 🏆 **Cross-Encoder Reranking**: Improve result relevance with specialized reranking models
- 🎯 **Cutoff Threshold**: Filter out low-scoring chunks to improve precision
- 🔧 **Configurable**: Highly configurable system with JSON-based configuration

## Recent Improvements

- Implemented BM25 algorithm for improved keyword search performance
- Enhanced query expansion with domain-specific terminology and contextual terms
- Implemented cross-encoder reranking for better result relevance scoring
- Improved diversification to balance content similarity and page distance
- Added performance monitoring and comprehensive logging

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages: listed in `requirements.txt`

### Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook examples or use the Python modules directly

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline with a PDF document
rag = RAGPipeline(pdf_path="human-nutrition-text.pdf")

# Ask a question
response = rag.query("What are the main macronutrients?")
print(response)
```

## Configuration

The system is configured through `config.json`. Key settings include:

- `use_hybrid_search`: Enable hybrid semantic + keyword search
- `semantic_weight` / `keyword_weight`: Control the balance between search methods
- `use_reranking`: Enable cross-encoder reranking for better relevance
- `ensure_diverse_chunks`: Enable diversity in retrieved chunks
- `use_query_expansion`: Enable query expansion techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.
