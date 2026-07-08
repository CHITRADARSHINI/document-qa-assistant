# 📄 DocChat — AI Document Q&A Assistant

Chat with your PDF documents using RAG (Retrieval-Augmented Generation).

## 🚀 Features
- Upload one or multiple PDFs into a single chat
- Ask questions and get accurate answers from your documents
- Sources shown for every answer
- Document history in sidebar
- Add more PDFs to an existing chat

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| LLM | Groq (Llama 3.1) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| Orchestration | LangChain |
| UI | Streamlit |

## ⚙️ Run Locally

```bash
git clone https://github.com/CHITRADARSHINI/document-qa-assistant
cd document-qa-assistant
pip install -r requirements.txt
# Add your GROQ_API_KEY to .env
streamlit run app/main.py
```

## 🏗️ Architecture
```
PDF → chunks → embeddings → FAISS vector store
                                    ↓
User question → embed → find similar chunks → Groq LLM → answer
```