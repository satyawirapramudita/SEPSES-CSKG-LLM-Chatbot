"""
SEPSES CSKG LLM Chatbot - Streamlit Frontend Application
=========================================================
Tanggung Jawab  : Muhammad Dhafin Alfeizar Gandhan (Full-Stack UI Dev)
Branch          : feature/frontend-ui
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Entry point multi-page Streamlit application dengan navigasi sidebar.
    Halaman yang tersedia:
    - 💬 Chat Interface  : Chat dengan LLM + KG context
    - 🔍 KG Explorer     : Browse Knowledge Graph secara interaktif
    - 📋 Log Analyzer    : Upload & analisis security log
    - 📊 Evaluation      : Dashboard perbandingan LLM
    - ⚙️  Settings       : Konfigurasi LLM dan endpoint

Desain:
    Dark cybersecurity theme dengan accent hijau neon.
    Semua state disimpan di st.session_state.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

# ── Page imports ────────────────────────────────────────────
from frontend.components.chat_window import render_chat_page
from frontend.components.eval_dashboard import render_eval_page
from frontend.components.graph_visualizer import render_kg_explorer_page
from frontend.components.log_uploader import render_log_analyzer_page

# ============================================================
# Page Config (HARUS sebelum semua st.* calls)
# ============================================================
st.set_page_config(
    page_title="SEPSES CSKG Chatbot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SEPSES CSKG LLM Chatbot — Cybersecurity Analysis via Knowledge Graph + LLM",
    }
)

# ============================================================
# Global CSS — Dark Cybersecurity Theme
# ============================================================
GLOBAL_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ───────────────────────────────────────── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #0f1629;
    --bg-card:       #131d35;
    --bg-card-hover: #1a2545;
    --accent-green:  #00ff88;
    --accent-blue:   #00b4ff;
    --accent-red:    #ff4d6d;
    --accent-yellow: #ffd60a;
    --text-primary:  #e8eaf0;
    --text-muted:    #6b7a99;
    --border-color:  #1e2d4a;
    --border-accent: #00ff8840;
    --gradient-main: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a1628 100%);
}

/* ── App Background ───────────────────────────────────────── */
.stApp {
    background: var(--gradient-main);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* ── Hide Streamlit Branding ──────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c18 0%, #0c1526 100%);
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Sidebar Logo Area ────────────────────────────────────── */
.sidebar-logo {
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-green) !important;
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px rgba(0,255,136,0.4);
}
.sidebar-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Nav Button (Radio as Nav) ────────────────────────────── */
[data-testid="stSidebar"] .stRadio label {
    cursor: pointer;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 2px 0;
    transition: all 0.2s ease;
    display: block;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,255,136,0.08) !important;
    color: var(--accent-green) !important;
}

/* ── Cards ────────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
}
.metric-card:hover {
    border-color: var(--border-accent);
    box-shadow: 0 0 25px rgba(0,255,136,0.1);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-green);
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ── Chat Bubbles ─────────────────────────────────────────── */
.chat-bubble-user {
    background: linear-gradient(135deg, #1a3a5c, #1e4a7a);
    border: 1px solid #2a5a8a;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.92rem;
    box-shadow: 0 2px 12px rgba(0,180,255,0.15);
}
.chat-bubble-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 85%;
    font-size: 0.92rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    position: relative;
}
.chat-bubble-assistant::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, var(--accent-green), var(--accent-blue));
    border-radius: 4px 0 0 4px;
}

/* ── Source Citation Box ──────────────────────────────────── */
.source-box {
    background: rgba(0,255,136,0.05);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-green);
}

/* ── Status Badges ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(255,77,109,0.2); color: #ff4d6d; border: 1px solid #ff4d6d40; }
.badge-high     { background: rgba(255,140,0,0.2);  color: #ff8c00; border: 1px solid #ff8c0040; }
.badge-medium   { background: rgba(255,214,10,0.2); color: #ffd60a; border: 1px solid #ffd60a40; }
.badge-low      { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid #00ff8840; }
.badge-info     { background: rgba(0,180,255,0.15); color: #00b4ff; border: 1px solid #00b4ff40; }

/* ── Section Headers ──────────────────────────────────────── */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
}

/* ── Input Fields ─────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0,255,136,0.15) !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00b4ff) !important;
    color: #0a0e1a !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(0,255,136,0.4) !important;
}

/* ── Select Box ───────────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Progress / Spinner ───────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--accent-green) !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-green) !important;
}

/* ── File Uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-green) !important;
}

/* ── Expander ─────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Animated Glow ────────────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(0,255,136,0.2); }
    50%       { box-shadow: 0 0 25px rgba(0,255,136,0.5); }
}
.glow-pulse { animation: pulse-glow 3s ease-in-out infinite; }

/* ── Typing Indicator ─────────────────────────────────────── */
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
.typing-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: blink 1.2s infinite;
    margin: 0 2px;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
</style>
"""


def _init_session_state() -> None:
    """
    Inisialisasi semua session state yang diperlukan aplikasi.
    Dipanggil sekali saat startup.
    """
    defaults = {
        # Navigasi
        "current_page": "💬 Chat",
        # Chat
        "chat_history": [],
        "selected_llm": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "chat_mode": "Security Analysis",
        # Log Analysis
        "ingested_logs": [],
        "log_stats": {},
        # KG Explorer
        "kg_query": "",
        "kg_results": None,
        "kg_graph_html": None,
        # Evaluation
        "eval_results": None,
        "eval_running": False,
        # Settings
        "sparql_endpoint": os.getenv("SPARQL_ENDPOINT", "https://w3id.org/sepses/sparql"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "mistral"),
        "top_k": int(os.getenv("TOP_K_RETRIEVAL", "5")),
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def _render_sidebar() -> str:
    """
    Render sidebar navigation dengan logo dan status indicators.

    Returns:
        str: Nama halaman yang dipilih user.
    """
    with st.sidebar:
        # ── Logo & Title ─────────────────────────────────────
        st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size:2rem; margin-bottom:0.3rem;">🛡️</div>
            <div class="sidebar-title">SEPSES CSKG</div>
            <div class="sidebar-subtitle">Cybersecurity AI Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.72rem; color:#6b7a99; text-transform:uppercase; "
            "letter-spacing:0.1em; margin: 0.5rem 0 0.3rem; padding-left:0.5rem;'>"
            "Navigation</p>",
            unsafe_allow_html=True
        )

        pages = [
            "💬 Chat",
            "🔍 KG Explorer",
            "📋 Log Analyzer",
            "📊 Evaluation",
            "⚙️ Settings",
        ]
        selected = st.radio(
            label="nav",
            options=pages,
            index=pages.index(st.session_state.current_page),
            label_visibility="collapsed",
        )
        st.session_state.current_page = selected

        # ── LLM Quick Selector ────────────────────────────────
        st.markdown("<hr style='border-color:#1e2d4a; margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem; color:#6b7a99; text-transform:uppercase; "
            "letter-spacing:0.1em; margin-bottom:0.5rem;'>Active LLM</p>",
            unsafe_allow_html=True
        )
        LLM_OPTIONS = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gpt-4o-mini",
            "mistral",
            "gemma4:latest",
            "minimax-m2.7:cloud",
        ]
        current = st.session_state.get("selected_llm", "gemini-2.0-flash")
        default_idx = LLM_OPTIONS.index(current) if current in LLM_OPTIONS else 0
        st.session_state.selected_llm = st.selectbox(
            label="llm_select",
            options=LLM_OPTIONS,
            index=default_idx,
            format_func=lambda m: {
                "gemini-2.0-flash":   "✨ Gemini 2.0 Flash (Google)",
                "gemini-1.5-flash":   "✨ Gemini 1.5 Flash (Google)",
                "gpt-4o-mini":        "🤖 GPT-4o-mini (OpenAI)",
                "mistral":            "🦙 Mistral-7B (Ollama)",
                "gemma4:latest":      "🦙 Gemma4 (Ollama)",
                "minimax-m2.7:cloud": "🦙 MiniMax (Ollama)",
            }.get(m, m),
            label_visibility="collapsed",
        )

        # ── Status Indicators ─────────────────────────────────
        st.markdown("<hr style='border-color:#1e2d4a; margin: 1rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem; color:#6b7a99; text-transform:uppercase; "
            "letter-spacing:0.1em; margin-bottom:0.5rem;'>System Status</p>",
            unsafe_allow_html=True
        )

        # KG Endpoint status
        kg_status = "🟢" if _check_kg_available() else "🟡"
        st.markdown(
            f"<div style='font-size:0.82rem; padding:0.2rem 0;'>"
            f"{kg_status} Knowledge Graph</div>",
            unsafe_allow_html=True
        )

        # LLM status
        has_gemini = bool(os.getenv("GEMINI_API_KEY")) and "GANTI" not in os.getenv("GEMINI_API_KEY", "")
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        llm_status = "🟢" if (has_gemini or has_openai) else "🔴"
        active_llm = st.session_state.get("selected_llm", "gemini-2.0-flash")
        llm_short = {"gemini-2.0-flash": "Gemini", "gpt-4o-mini": "GPT-4o"}.get(active_llm, active_llm[:8])
        st.markdown(
            f"<div style='font-size:0.82rem; padding:0.2rem 0;'>"
            f"{llm_status} LLM ({llm_short})</div>",
            unsafe_allow_html=True
        )

        # Log DB status
        log_count = len(st.session_state.ingested_logs)
        log_status = "🟢" if log_count > 0 else "⚪"
        st.markdown(
            f"<div style='font-size:0.82rem; padding:0.2rem 0;'>"
            f"{log_status} Log Vector DB "
            f"<span style='color:#6b7a99'>({log_count} entries)</span></div>",
            unsafe_allow_html=True
        )

        # ── Version Info ──────────────────────────────────────
        st.markdown(
            "<div style='position:absolute; bottom:1rem; left:1rem; right:1rem;"
            "font-size:0.68rem; color:#3a4a6a; text-align:center;'>"
            "SEPSES CSKG v1.0.0<br>"
            "Topic 4 · Cybersecurity AI</div>",
            unsafe_allow_html=True
        )

    return selected


def _check_kg_available() -> bool:
    """
    Cek apakah SPARQL endpoint atau RAG pipeline tersedia.

    Returns:
        bool: True jika tersedia.
    """
    try:
        from rag_logic.rag_pipeline import RagPipeline  # type: ignore  # noqa: F401
        return True
    except ImportError:
        pass
    # Coba ping SPARQL endpoint
    try:
        import requests
        endpoint = os.getenv("SPARQL_ENDPOINT", "https://w3id.org/sepses/sparql")
        r = requests.get(endpoint, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


# ============================================================
# Main Entry Point
# ============================================================
def main() -> None:
    """Entry point utama aplikasi Streamlit."""

    # Inject global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Init session state
    _init_session_state()

    # Render sidebar & get selected page
    selected_page = _render_sidebar()

    # ── Route ke halaman yang sesuai ─────────────────────────
    if selected_page == "💬 Chat":
        render_chat_page()
    elif selected_page == "🔍 KG Explorer":
        render_kg_explorer_page()
    elif selected_page == "📋 Log Analyzer":
        render_log_analyzer_page()
    elif selected_page == "📊 Evaluation":
        render_eval_page()
    elif selected_page == "⚙️ Settings":
        _render_settings_page()


def _render_settings_page() -> None:
    """Render halaman Settings untuk konfigurasi endpoint dan model."""
    st.markdown(
        "<div class='section-header'>⚙️ Settings</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔌 Endpoint Configuration")
        new_endpoint = st.text_input(
            "SPARQL Endpoint",
            value=st.session_state.sparql_endpoint,
            help="SEPSES KG SPARQL endpoint URL",
        )
        if new_endpoint != st.session_state.sparql_endpoint:
            st.session_state.sparql_endpoint = new_endpoint

        new_ollama = st.text_input(
            "Ollama Model",
            value=st.session_state.ollama_model,
            help="Model name untuk Ollama (e.g. mistral, llama3)",
        )
        if new_ollama != st.session_state.ollama_model:
            st.session_state.ollama_model = new_ollama

        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1, max_value=20,
            value=st.session_state.top_k,
            help="Jumlah dokumen/triples yang diambil per query",
        )
        st.session_state.top_k = top_k

    with col2:
        st.subheader("🔑 API Key Status")
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY")) and "GANTI" not in os.getenv("GEMINI_API_KEY", "")

        st.markdown(f"**Google Gemini API Key**: {'✅ Configured' if has_gemini else '❌ Not set'}")
        if not has_gemini:
            st.markdown(
                "🔗 Dapatkan **gratis** di: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)"
            )

        st.markdown(f"**OpenAI API Key**: {'✅ Configured' if has_openai else '⚠️ Quota may be exhausted'}")

        st.info(
            "API keys dikelola via file `.env`. Isi `GEMINI_API_KEY` untuk "
            "menggunakan Gemini 2.0 Flash secara **gratis**. Jangan pernah commit `.env`.",
            icon="🔐"
        )

        st.subheader("📊 Session Stats")
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Log Entries Ingested", len(st.session_state.ingested_logs))

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
            st.rerun()

        if st.button("🗑️ Clear Ingested Logs", use_container_width=True):
            st.session_state.ingested_logs = []
            st.session_state.log_stats = {}
            st.success("Log data cleared.")
            st.rerun()


if __name__ == "__main__":
    main()
