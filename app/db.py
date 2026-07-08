import sqlite3
import json
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "chats.db"
STORE_DIR = Path(__file__).parent / "stores"
STORE_DIR.mkdir(exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            docs TEXT,
            history TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn

def create_chat(docs):
    chat_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO chats (id, docs, history) VALUES (?, ?, ?)",
        (chat_id, json.dumps(docs), json.dumps([]))
    )
    conn.commit()
    conn.close()
    return chat_id

def list_chats():
    conn = get_conn()
    rows = conn.execute("SELECT id, docs FROM chats ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": r[0], "docs": json.loads(r[1])} for r in rows]

def get_chat(chat_id):
    conn = get_conn()
    row = conn.execute("SELECT docs, history FROM chats WHERE id=?", (chat_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"docs": json.loads(row[0]), "history": json.loads(row[1])}

def update_chat(chat_id, docs=None, history=None):
    conn = get_conn()
    if docs is not None:
        conn.execute("UPDATE chats SET docs=? WHERE id=?", (json.dumps(docs), chat_id))
    if history is not None:
        conn.execute("UPDATE chats SET history=? WHERE id=?", (json.dumps(history), chat_id))
    conn.commit()
    conn.close()

def store_path(chat_id):
    return str(STORE_DIR / chat_id)