import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #21262d !important;
    }
    [data-testid="stSidebarContent"] { padding: 8px 12px !important; }
    [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
    [data-testid="stSidebarHeader"] { padding: 0 !important; min-height: 0 !important; }

    .stApp, p, label { color: #c9d1d9 !important; }
    h1, h2, h3, h4 { color: #e6edf3 !important; }

    /* ── Sidebar buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #8b949e !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 8px 10px !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }

    /* ── "New chat" styled as its own box ── */
    [data-testid="stSidebar"] .stButton:first-of-type > button {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        font-weight: 600 !important;
        padding: 10px 12px !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] .stButton:first-of-type > button:hover {
        border-color: #58a6ff !important;
        background-color: #21262d !important;
    }

    /* ── Main buttons ── */
    .stButton > button {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        border-color: #58a6ff !important;
        background-color: #21262d !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        color: #e6edf3 !important;
    }
    /* Push chat text right to make room for + on left */
    div[data-testid="stChatInput"] > div {
        padding-left: 44px !important;
    }
    [data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
    }

    iframe {
        display: block !important;
        border: none !important;
        height: 0 !important;
    }

    /* ── Upload screen file uploader ── */
    [data-testid="stFileUploader"] {
        background: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 12px !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: #58a6ff !important; }
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 28px 24px !important;
        background: transparent !important;
        border: none !important;
        min-height: 110px;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #1f6feb !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 24px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        display: flex !important;
    }
    [data-testid="stFileUploaderDropzone"] button span { display: inline !important; }
    [data-testid="stFileUploaderDropzone"] button:hover { background: #388bfd !important; }

    /* ── Example cards ── */
    .ex-cards-wrap {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 16px;
    }
    .ex-cards-row {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 16px;
    }
    .ex-card {
        flex: 1 1 220px;
        min-width: 200px;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 18px;
        min-height: 120px;
        transition: border-color 0.15s;
        box-sizing: border-box;
    }
    .ex-card:hover { border-color: #388bfd; }
    .ex-card-icon { font-size: 26px; margin-bottom: 10px; }
    .ex-card-title { font-size: 14px; font-weight: 600; color: #e6edf3; margin-bottom: 6px; }
    .ex-card-desc { font-size: 13px; color: #8b949e; line-height: 1.5; }

    /* ── Responsive tweaks ── */
    @media (max-width: 700px) {
        .ex-card { flex: 1 1 100%; }
        div[data-testid="stChatInput"] > div { padding-left: 40px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
for key, default in {
    "qa_chain": None,
    "chat_history": [],
    "doc_history": [],
    "active_chat_idx": None,
    "screen": "upload",
    "add_doc_key": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────
def display_name(name):
    return name[:-4] if name.lower().endswith(".pdf") else name

def truncate(name, n=28):
    return name if len(name) <= n else name[:n-3] + "..."

def build_combined_chain(files_data):
    all_chunks = []
    for filename, data in files_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return chain, vectorstore

def switch_to_chat(idx):
    chat = st.session_state.doc_history[idx]
    st.session_state.active_chat_idx = idx
    st.session_state.qa_chain = chat["chain"]
    st.session_state.chat_history = chat["history"]
    st.session_state.screen = "chat"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;
                padding:0 2px 12px 2px;
                border-bottom:1px solid #21262d;
                margin-bottom:12px'>
        <span style='font-size:20px'>📄</span>
        <span style='font-size:18px;font-weight:700;color:#e6edf3'>DocChat</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New chat", key="new_chat_btn", use_container_width=True):
        st.session_state.screen = "upload"
        st.rerun()

    if st.session_state.doc_history:
        st.markdown("""
        <p style='font-size:11px;color:#484f58;text-transform:uppercase;
                  letter-spacing:0.8px;padding:16px 2px 6px 2px;margin:0'>
            Chats
        </p>""", unsafe_allow_html=True)
        for i, chat in enumerate(st.session_state.doc_history):
            is_active = i == st.session_state.active_chat_idx
            names = " + ".join([display_name(n) for n in chat["docs"]])
            icon = "📖" if is_active else "📄"
            if st.button(
                f"{icon}  {truncate(names)}",
                key=f"chat_{i}",
                use_container_width=True
            ):
                switch_to_chat(i)
                st.rerun()

# ══════════════════════════════════════════════════════════
# SCREEN 1 — Upload
# ══════════════════════════════════════════════════════════
if st.session_state.screen == "upload":
    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div style='text-align:center;margin-bottom:28px'>
            <h2 style='font-size:34px;font-weight:700;
                       color:#e6edf3;margin-bottom:8px'>
                Start a New Chat
            </h2>
            <p style='color:#8b949e;font-size:15px;margin:0'>
                Upload one or more PDFs — all in one chat
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload PDFs", type="pdf",
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files:
            files_data = [(f.name, f.read()) for f in uploaded_files]
            doc_names = [f[0] for f in files_data]
            with st.spinner("Building your chat..."):
                chain, vectorstore = build_combined_chain(files_data)
            st.session_state.doc_history.append({
                "name": doc_names[0],
                "docs": doc_names,
                "chain": chain,
                "vectorstore": vectorstore,
                "history": []
            })
            switch_to_chat(len(st.session_state.doc_history) - 1)
            st.rerun()

        if st.session_state.doc_history:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to chat", use_container_width=True):
                st.session_state.screen = "chat"
                st.rerun()

    # ── Example cards: full-width responsive flex row, kept out of the
    # narrow "mid" column so it never gets forced to stack vertically ──
    st.markdown("""
    <div style='text-align:center;margin:36px 0 16px 0'>
        <p style='color:#484f58;font-size:12px;
                  text-transform:uppercase;
                  letter-spacing:1px;margin:0'>
            Get Started with Examples
        </p>
    </div>
    """, unsafe_allow_html=True)

    example_cards = [
        ("📊", "Analyze Sales Data",
         "Quickly summarize and query your quarterly reports."),
        ("⚖️", "Review Legal Contracts",
         "Ask questions about clauses and obligations in your PDFs."),
        ("🔬", "Summarize Research Papers",
         "Extract key insights and findings from academic papers."),
    ]
    cards_html = "".join(
        f"""<div class='ex-card'>
            <div class='ex-card-icon'>{icon}</div>
            <div class='ex-card-title'>{title}</div>
            <div class='ex-card-desc'>{desc}</div>
        </div>"""
        for icon, title, desc in example_cards
    )
    st.markdown(
        f"<div class='ex-cards-wrap'><div class='ex-cards-row'>{cards_html}</div></div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
# SCREEN 2 — Chat
# ══════════════════════════════════════════════════════════
elif st.session_state.screen == "chat":
    idx = st.session_state.active_chat_idx
    if idx is None:
        st.session_state.screen = "upload"
        st.rerun()

    chat = st.session_state.doc_history[idx]
    doc_labels = " + ".join([display_name(n) for n in chat["docs"]])

    st.markdown(
        f"<h4 style='color:#e6edf3;margin-bottom:4px'>📖 {doc_labels}</h4>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<hr style='border:none;border-top:1px solid #21262d;margin:4px 0 16px 0'>",
        unsafe_allow_html=True
    )

    # Chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(msg["sources"]):
                        st.caption(f"**{i+1}.** {src[:300]}")

    # ── Completely invisible file uploader ────────────────
    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] {
        position: fixed !important;
        top: -99999px !important;
        left: -99999px !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        visibility: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)

    extra_files = st.file_uploader(
        "add", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"plus_{st.session_state.add_doc_key}"
    )

    # ── + button injected directly inside the chat input bar ─
    # Instead of floating a separate iframe on top of the page (which drifts
    # away from the chat bar), we reach into the parent document and insert
    # a real <button> as a child of the chat input's own flex container —
    # the same way Streamlit renders its send button on the right.
    components.html("""
    <script>
    (function() {
        function addPlusButton() {
            var doc = window.parent.document;
            var container = doc.querySelector('div[data-testid="stChatInput"] > div');
            if (!container) return;
            if (container.querySelector('#customPlusBtn')) return;

            container.style.position = 'relative';

            var btn = doc.createElement('button');
            btn.id = 'customPlusBtn';
            btn.type = 'button';
            btn.innerHTML = '＋';
            btn.style.cssText = [
                'position:absolute',
                'left:10px',
                'top:50%',
                'transform:translateY(-50%)',
                'background:transparent',
                'border:none',
                'color:#6e7681',
                'font-size:22px',
                'line-height:1',
                'cursor:pointer',
                'width:28px',
                'height:28px',
                'display:flex',
                'align-items:center',
                'justify-content:center',
                'padding:0',
                'z-index:1000'
            ].join(';');

            btn.addEventListener('mouseover', function() { this.style.color = '#58a6ff'; });
            btn.addEventListener('mouseout', function() { this.style.color = '#6e7681'; });
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var inputs = doc.querySelectorAll('input[type="file"]');
                if (inputs.length > 0) {
                    inputs[inputs.length - 1].click();
                }
            });

            container.insertBefore(btn, container.firstChild);
        }

        addPlusButton();

        // Streamlit re-renders the DOM on every rerun, which removes our
        // injected button — reattach it whenever the parent DOM changes.
        var observer = new MutationObserver(function() { addPlusButton(); });
        observer.observe(window.parent.document.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0, scrolling=False)

    # ── Chat input ────────────────────────────────────────
    question = st.chat_input("Ask anything about your document...")

    # ── Handle new files added ────────────────────────────
    if extra_files:
        new_files = [
            (f.name, f.read()) for f in extra_files
            if f.name not in chat["docs"]
        ]
        if new_files:
            with st.spinner("Adding documents..."):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                for fname, fdata in new_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(fdata)
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50
                    )
                    chunks = splitter.split_documents(docs)
                    chat["vectorstore"].add_documents(chunks)
                    chat["docs"].append(fname)
                retriever = chat["vectorstore"].as_retriever(
                    search_kwargs={"k": 3}
                )
                llm = ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY")
                )
                new_chain = RetrievalQA.from_chain_type(
                    llm=llm, retriever=retriever,
                    return_source_documents=True
                )
                chat["chain"] = new_chain
                st.session_state.qa_chain = new_chain
            st.session_state.add_doc_key += 1
            st.rerun()

    # ── Handle question ───────────────────────────────────
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner(""):
                response = st.session_state.qa_chain.invoke(
                    {"query": question}
                )
                answer = response["result"]
                sources = [
                    d.page_content for d in response["source_documents"]
                ]
            st.write(answer)
            with st.expander("📚 Sources"):
                for i, src in enumerate(sources):
                    st.caption(f"**{i+1}.** {src[:300]}")

        st.session_state.chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "sources": sources}
        ])
        chat["history"] = st.session_state.chat_history