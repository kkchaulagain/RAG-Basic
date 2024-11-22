# RAG Learning System

A Retrieval-Augmented Generation (RAG) system that automatically collects, processes, and indexes topic-specific information for interactive learning conversations. The system consists of two main components: data import (`import.py`) and interactive RAG agent (`rag.py`).

## Overview

This system enables:
- Automated web scraping and content processing for specific topics
- Intelligent filtering and organization of relevant information
- Interactive Q&A with context-aware responses
- Topic-specific knowledge base creation and maintenance

## System Components

### 1. Import System (import.py)

The import component handles data collection and processing:

```bash
python import.py --topics "topic1,topic2,topic3" --output-dir "./docs"
```

Key features:
- Automated web scraping from reliable sources
- Content relevance checking using LLM
- Topic-specific information extraction
- Intelligent content organization
- Vector embeddings generation for efficient retrieval

Configuration options:
```yaml
# config.yaml
import:
  max_articles: 5
  chunk_size: 2000
  chunk_overlap: 200
  embeddings_model: "HuggingFace"
  llm_model: "llama3.1:latest"
```

### 2. RAG Agent (rag.py)

The interactive component for knowledge retrieval and generation:

```bash
python rag.py --knowledge-base "./processed_docs" --model "llama3.1:latest"
```

Features:
- Context-aware responses
- Semantic search for relevant information
- Conversation memory
- Source attribution

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd rag-learning-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and download the required model:
```bash
# Install Ollama (MacOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.1:latest
```

## Dependencies

Required Python packages:
```
selenium==4.15.2
beautifulsoup4==4.12.2
langchain==0.1.0
langchain-ollama==0.1.1
faiss-cpu==1.7.4
transformers==4.36.2
sentence-transformers==2.2.2
webdriver-manager==4.0.1
```

## Usage

### 1. Importing Knowledge

```python
# Example usage of import.py
from import_system import scrape_articles

queries = [
    "Agile project management basics",
    "Software development lifecycle best practices",
    "How to create a project quotation",
    "FAQs for IT clients"
]

for query in queries:
    scrape_articles(query)
```

### 2. Interacting with RAG Agent

```python
# Example usage of rag.py
from rag_agent import RagAgent

agent = RagAgent(
    knowledge_base_path="./processed_docs",
    model_name="llama3.1:latest"
)

response = agent.query("Explain the key principles of Agile project management.")
print(response)
```

## File Structure

```
rag-learning-system/
├── import.py                 # Data import and processing
├── rag.py                   # Interactive RAG agent
├── config.yaml              # Configuration settings
├── requirements.txt         # Python dependencies
├── docs/                    # Raw scraped content
└── processed_docs/          # Processed and indexed content
    ├── topic1/
    ├── topic2/
    └── topic3/
```

## How It Works

1. **Data Import Process**:
   - Web scraping of relevant articles
   - Content relevance assessment
   - Topic-specific information extraction
   - Generation of embeddings
   - Organization into topic-specific folders

2. **RAG Interaction Process**:
   - Query analysis
   - Semantic search for relevant context
   - Context-aware response generation
   - Conversation history maintenance

## Configuration

The system can be configured through `config.yaml`:

```yaml
system:
  base_path: "./processed_docs"
  max_context_length: 4000

import:
  max_articles: 5
  chunk_size: 2000
  chunk_overlap: 200

rag:
  model_name: "llama3.1:latest"
  temperature: 0.7
  max_tokens: 500
```

## Error Handling

The system includes robust error handling for:
- Failed web requests
- Content processing errors
- Model availability issues
- File system operations

## Limitations

- Requires active internet connection for import
- Depends on Ollama availability
- Limited to text-based content
- Processing time depends on content volume

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Contact

For support or questions, please open an issue in the repository.