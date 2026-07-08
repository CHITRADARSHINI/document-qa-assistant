# DocChat 📄

Chat with one or more PDFs at once. Upload your documents, ask questions in
plain English, and get answers grounded in the source text — with the
supporting passages shown alongside every response.

## Features

- **Multi-document chat** — upload several PDFs into a single conversation
  and ask questions across all of them at once
- **Add documents mid-conversation** — drop in more PDFs without starting a
  new chat
- **Source citations** — every answer comes with an expandable panel showing
  the exact passages it was drawn from
- **Multiple saved chats** — switch between different document sets from the
  sidebar
- **Retrieval-Augmented Generation (RAG)** — powered by FAISS similarity
  search over chunked, embedded document text

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://www.langchain.com/) — RAG pipeline orchestration
- [Groq](https://groq.com/) (`llama-3.1-8b-instant`) — LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — vector store
- [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) — embeddings

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/docchat.git
   cd docchat
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your Groq API key**

   Copy `.env.example` to `.env` and fill in your key
   (get one free at [console.groq.com](https://console.groq.com/keys)):
   ```bash
   cp .env.example .env
   ```

4. **Run the app**
   ```bash
   streamlit run main.py
   ```

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with
   GitHub, and click **New app**.
3. Select this repo and set the main file path to `main.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy — you'll get a public URL to link from your portfolio.

## Project structure

```
docchat/
├── main.py            # App entry point (UI + RAG pipeline)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for required environment variables
├── .gitignore
└── README.md
```

## License

MIT