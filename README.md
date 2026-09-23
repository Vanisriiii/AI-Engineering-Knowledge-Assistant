# 🤖 AI Engineering Knowledge & Code Review Assistant

Understand Your Project. Ask Questions. Review Code.

AI Engineering Knowledge & Code Review Assistant is an AI-powered application built with Python, Streamlit, LangChain, Gemini, and FAISS.

It helps users understand project documentation using Retrieval-Augmented Generation (RAG) and review Python source code for potential issues, improvements, and test cases.

🔗 Live Demo: [https://ai-engineering-knowledge-assistant-ew5tjt2efwgmi7o9tvshcw.streamlit.app/](https://ai-engineering-knowledge-assistant-ew5tjt2efwgmi7o9tvshcw.streamlit.app/)

---

## ✨ Features

### 📚 Project Knowledge Base

The Project Knowledge Base allows users to upload project documentation and create an AI-powered searchable knowledge base.

### 📤 Upload Project Documentation

Upload project documentation in:

- PDF format
- TXT format

The application extracts the text from uploaded files and prepares it for semantic search.

### ✂️ Document Processing

Uploaded documents are processed through:

- Text extraction
- Text chunking
- Document splitting
- Metadata creation

This allows large documents to be divided into smaller, meaningful sections for efficient retrieval.

### 🧠 Semantic Search with RAG

The application uses Gemini Embeddings and FAISS to create a searchable vector database.

When a user asks a question, the system:

1. Searches the knowledge base
2. Finds relevant document sections
3. Retrieves the most relevant context
4. Sends the retrieved context to Gemini
5. Generates an answer based on the retrieved information
6. Displays the relevant source files

### 💬 AI Knowledge Assistant

Users can ask questions such as:

- What is this project about?
- Which API is responsible for authentication?
- Explain this functionality.
- Where is customer validation mentioned?
- What could cause this API error?
- Find the relevant section for this requirement.

The assistant uses the uploaded project documentation to generate grounded responses.

---

## 🔍 Retrieval-Augmented Generation

The application uses a RAG pipeline to connect project documentation with the Gemini language model.

```text
Upload PDF / TXT
        ↓
Extract Text
        ↓
Split Text into Chunks
        ↓
Generate Gemini Embeddings
        ↓
Store Embeddings in FAISS
        ↓
User Question
        ↓
Semantic Similarity Search
        ↓
Retrieve Relevant Context
        ↓
Gemini LLM
        ↓
AI Answer + Sources
