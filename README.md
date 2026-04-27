# RAG Tutorial with LangChain

A step-by-step tutorial demonstrating how to build a Retrieval Augmented Generation (RAG) system using LangChain, OpenAI, and Pinecone.

## Overview

This tutorial progressively builds a complete RAG pipeline:
1. **Document Ingestion** - Load, chunk, embed, and store documents in a vector database
2. **Naive RAG** - Implement a basic retrieval chain using manual function calls
3. **LCEL RAG** - Refactor to use LangChain Expression Language for a cleaner, more powerful approach

## Tutorial Progression

Follow the commits in order to learn RAG concepts incrementally:

| Step | Commit | Description |
|------|--------|-------------|
| 1 | `598dee4` | **Initial Setup** - Project structure with dependencies (LangChain, OpenAI, Pinecone), sample data, and basic configuration |
| 2 | `2e34caf` | **Add Imports** - Import LangChain components for document processing: TextLoader, CharacterTextSplitter, OpenAIEmbeddings, PineconeVectorStore |
| 3 | `9066596` | **Document Ingestion Pipeline** - Complete ingestion: load text documents, split into chunks, generate embeddings, store in Pinecone |
| 4 | `5b0c33f` | **Naive RAG Implementation** - Manual step-by-step retrieval chain demonstrating core RAG concepts without LCEL |
| 5 | `5e4d009` | **LCEL-Based RAG** - Declarative retrieval chain using LangChain Expression Language with streaming, async, and composability |

## Key Components

### Ingestion (`ingestion.py`)
- **TextLoader**: Load documents from text files
- **CharacterTextSplitter**: Split documents into manageable chunks (1000 chars)
- **OpenAIEmbeddings**: Convert text chunks to vector embeddings
- **PineconeVectorStore**: Store and index vectors for similarity search

### Retrieval (`main.py`)
- **Raw LLM**: Direct query to LLM (no context) - baseline comparison
- **Naive RAG**: Manual retrieval → format → prompt → LLM pipeline
- **LCEL RAG**: Declarative chain using `|` operator with built-in streaming/async

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_index_name
```

3. Run ingestion to populate the vector store:
```bash
python ingestion.py
```

4. Run the RAG pipeline:
```bash
python main.py
```

## How the LCEL Retrieval Chain Works

The core of the LCEL approach is this chain:

```python
retrieval_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
)
```

The `|` operator pipes output from one step as input to the next.

### Component Breakdown

| Component | Role | Input → Output |
|-----------|------|----------------|
| `RunnablePassthrough.assign(context=...)` | Passes input dict through unchanged, adds a `context` key | `dict` → `dict` (with `context` added) |
| `itemgetter("question")` | Extracts the question string from the dict | `dict` → `str` |
| `retriever` | Similarity search against the vector store | `str` → `list[Document]` |
| `format_docs` | Joins document contents into a single string | `list[Document]` → `str` |
| `prompt_template` | Renders `{question}` and `{context}` into a prompt | `dict` → `PromptValue` |
| `llm` | Generates a response from the prompt | `PromptValue` → `AIMessage` |
| `StrOutputParser()` | Strips the `AIMessage` wrapper | `AIMessage` → `str` |

### Dry Run Example

**Input:**
```python
retrieval_chain.invoke({"question": "What is prompt injection?"})
```

**Step 1 — `RunnablePassthrough.assign`**

The sub-chain `itemgetter("question") | retriever | format_docs` runs on the input:
```
"What is prompt injection?"          # itemgetter("question")
        ↓
[Document(page_content="Prompt injection is an attack..."),
 Document(page_content="Attackers embed malicious instructions...")]  # retriever
        ↓
"Prompt injection is an attack...\n\nAttackers embed malicious instructions..."  # format_docs
```

The result is merged back into the original dict:
```python
{
  "question": "What is prompt injection?",
  "context": "Prompt injection is an attack...\n\nAttackers embed malicious instructions..."
}
```

**Step 2 — `prompt_template`**

Renders the template (e.g. `"Answer using context:\n{context}\n\nQuestion: {question}"`):
```
Answer using context:
Prompt injection is an attack...

Attackers embed malicious instructions...

Question: What is prompt injection?
```

**Step 3 — `llm`**

Returns an `AIMessage`:
```python
AIMessage(content="Prompt injection is a technique where attackers embed instructions...")
```

**Step 4 — `StrOutputParser`**

Returns a plain string:
```python
"Prompt injection is a technique where attackers embed instructions..."
```

> **Key insight:** The `assign` step is what makes this a retrieval chain — it runs the retriever in parallel with passing through the original input, so both `question` and `context` are available when the prompt template runs.

## Why LCEL?

The tutorial demonstrates two approaches to building RAG:

| Feature | Naive Approach | LCEL Approach |
|---------|----------------|---------------|
| Code style | Imperative | Declarative |
| Streaming | Manual implementation | Built-in `.stream()` |
| Async | Manual implementation | Built-in `.ainvoke()` |
| Composability | Limited | Pipe operator `\|` |
| Batch processing | Manual loops | Built-in `.batch()` |

## Technologies

- **LangChain** - Framework for building LLM applications
- **OpenAI** - Embeddings and chat completions
- **Pinecone** - Vector database for similarity search
- **Python** - 3.12+
