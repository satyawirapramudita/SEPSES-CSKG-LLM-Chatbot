"""
SEPSES CSKG LLM Chatbot - Chat Window Component
================================================
Tanggung Jawab  : Muhammad Dhafin Alfeizar Gandhan (Full-Stack UI Dev)
Branch          : feature/frontend-ui

Deskripsi:
    Komponen chat interface utama dengan:
    - Message history dengan role-based bubbles (user/assistant)
    - Mode selector: Security Analysis | Log Analysis | KG QA
    - Source citation display (SPARQL query + KG nodes)
    - Streaming-like response display
    - Graceful fallback ke mock response jika RAG pipeline belum tersedia
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st


# ============================================================
# Mock RAG Pipeline (fallback jika rag_logic belum tersedia)
# ============================================================
def _get_rag_pipeline(llm_name: str):
    """
    Load RAG pipeline nyata, atau fallback ke mock jika belum tersedia.

    Args:
        llm_name: Nama LLM yang dipilih user.

    Returns:
        Callable: query(question, mode) -> (answer, context, sparql_used)
    """
    try:
        from rag_logic.rag_pipeline import RagPipeline  # type: ignore
        pipeline = RagPipeline(llm_name=llm_name)

        def real_query(question: str, mode: str) -> Tuple[str, str, str]:
            result = pipeline.query(question, mode=mode)
            return (
                result.get("answer", ""),
                result.get("context", ""),
                result.get("sparql_used", ""),
            )
        return real_query

    except ImportError:
        return _mock_query_fn(llm_name)


def _mock_query_fn(llm_name: str):
    """
    Mock query function untuk demo/testing sebelum RAG pipeline siap.

    Args:
        llm_name: Nama LLM.

    Returns:
        Callable: Mock query function.
    """
    MOCK_RESPONSES = {
        "Security Analysis": {
            "answer": (
                "**[Demo Mode — RAG pipeline belum terhubung]**\n\n"
                "Berdasarkan SEPSES Knowledge Graph, CVE-2021-44228 (Log4Shell) adalah "
                "kerentanan kritis dengan **CVSS Score 10.0** yang mempengaruhi Apache Log4j 2.x. "
                "Kerentanan ini terhubung ke:\n"
                "- 🔴 **CWE-917**: Improper Neutralization of Special Elements\n"
                "- ⚔️ **CAPEC-88**: OS Command Injection via HTTP Query Strings\n"
                "- 🎯 **ATT&CK T1190**: Exploit Public-Facing Application\n\n"
                "Rekomendasi: Upgrade ke Log4j 2.17.1+, atau gunakan mitigasi "
                "`-Dlog4j2.formatMsgNoLookups=true`."
            ),
            "context": (
                "cve:CVE-2021-44228 cve:hasCVSS cvss:BaseMetric_10.0 ;\n"
                "  cve:hasCWE cwe:CWE-917 ;\n"
                "  cve:hasCPE cpe:apache:log4j:2.0 .\n"
                "cwe:CWE-917 cwe:hasCAPEC capec:CAPEC-88 ."
            ),
            "sparql": (
                "SELECT ?cwe ?capec ?score WHERE {\n"
                "  <http://w3id.org/sepses/resource/cve/CVE-2021-44228>\n"
                "    cve:hasCVSS ?cvssNode ;\n"
                "    cve:hasCWE ?cwe .\n"
                "  ?cwe cwe:hasCAPEC ?capec .\n"
                "  ?cvssNode cvss:baseScore ?score .\n"
                "}"
            ),
        },
        "Log Analysis": {
            "answer": (
                "**[Demo Mode — Log Analysis belum terhubung]**\n\n"
                "Analisis log menunjukkan **3 anomali** terdeteksi:\n\n"
                "1. 🔴 **CRITICAL** — SQL Injection attempt dari IP `192.168.1.100` "
                "ke database port 3306. Terkait CWE-89.\n"
                "2. 🟠 **HIGH** — Brute force SSH dari `198.51.100.22` (50+ attempts). "
                "CAPEC-49 berlaku.\n"
                "3. 🟡 **MEDIUM** — Port scan terdeteksi pada subnet 10.0.0.0/24.\n\n"
                "**Rekomendasi segera**: Blokir IP penyerang, aktifkan account lockout policy."
            ),
            "context": "Log entries: 13 alerts parsed. Severity distribution: 2 Critical, 3 High, 5 Medium, 3 Low",
            "sparql": "N/A — Log analysis uses ChromaDB vector search",
        },
        "KG Question Answering": {
            "answer": (
                "**[Demo Mode — KG QA belum terhubung]**\n\n"
                "SEPSES Knowledge Graph mengintegrasikan data dari:\n"
                "- **CVE** (Common Vulnerabilities & Exposures) — ~200K+ entries\n"
                "- **CWE** (Common Weakness Enumeration) — 900+ weakness types\n"
                "- **CAPEC** (Attack Patterns) — 500+ attack patterns\n"
                "- **CPE** (Platform Enumeration) — product-vulnerability mapping\n"
                "- **CVSS** (Scoring) — severity metrics\n"
                "- **ATT&CK** Enterprise + ICS — tactics & techniques\n\n"
                "Total: **36+ juta triples** accessible via SPARQL endpoint."
            ),
            "context": "SPARQL endpoint: https://w3id.org/sepses/sparql — Status: Public",
            "sparql": "SELECT (COUNT(?s) AS ?total) WHERE { ?s a cve:CVE . }",
        },
    }

    def mock_query(question: str, mode: str) -> Tuple[str, str, str]:
        resp = MOCK_RESPONSES.get(mode, MOCK_RESPONSES["Security Analysis"])
        # Simulasi latency
        time.sleep(0.8)
        return resp["answer"], resp["context"], resp["sparql"]

    return mock_query


# ============================================================
# Main Chat Page Renderer
# ============================================================
def render_chat_page() -> None:
    """Render halaman chat interface utama."""

    # ── Page Header ───────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size:1.8rem; font-weight:700; color:#e8eaf0; margin:0;">
            💬 Security Analysis Chat
        </h1>
        <p style="color:#6b7a99; font-size:0.88rem; margin:0.3rem 0 0;">
            Query the SEPSES Cybersecurity Knowledge Graph via natural language
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls Row ──────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

    with ctrl_col1:
        mode = st.selectbox(
            "Analysis Mode",
            options=["Security Analysis", "Log Analysis", "KG Question Answering"],
            index=["Security Analysis", "Log Analysis", "KG Question Answering"].index(
                st.session_state.get("chat_mode", "Security Analysis")
            ),
            help=(
                "Security Analysis: threat/vuln analysis via KG\n"
                "Log Analysis: query against ingested security logs\n"
                "KG Question Answering: general KG queries"
            ),
        )
        st.session_state.chat_mode = mode

    with ctrl_col2:
        llm_label = {
            "gpt-4o-mini": "🤖 GPT-4o-mini (OpenAI)",
            "mistral": "🦙 Mistral-7B (Ollama)",
        }
        current_llm = st.session_state.selected_llm
        st.markdown(
            f"<div style='padding-top:0.3rem;'>"
            f"<span style='font-size:0.8rem;color:#6b7a99;'>Active LLM:</span><br>"
            f"<span style='font-weight:600;color:#00ff88;'>{llm_label.get(current_llm, current_llm)}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    with ctrl_col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<hr style='border-color:#1e2d4a; margin: 0.5rem 0 1rem;'>", unsafe_allow_html=True)

    # ── Mode Info Banner ──────────────────────────────────────
    mode_config = {
        "Security Analysis": {
            "icon": "🔐",
            "desc": "Analisis CVE, CWE, CAPEC, threat actors, dan vulnerability chains via SEPSES KG",
            "examples": [
                "What attack patterns exploit CVE-2021-44228?",
                "Explain the attack chain for Log4Shell",
                "Which CVEs affect Apache products with CVSS > 9.0?",
            ],
        },
        "Log Analysis": {
            "icon": "📋",
            "desc": "Analisis security log yang telah diupload. Upload log terlebih dahulu di halaman Log Analyzer.",
            "examples": [
                "Identify the top threats in the uploaded logs",
                "What SQL injection attempts were detected?",
                "Summarize all high-priority alerts",
            ],
        },
        "KG Question Answering": {
            "icon": "🔍",
            "desc": "Tanya jawab umum atas struktur dan konten SEPSES Knowledge Graph",
            "examples": [
                "How many CVEs are in the SEPSES KG?",
                "What vocabularies does SEPSES CSKG use?",
                "Show me CVEs linked to CWE-79",
            ],
        },
    }

    cfg = mode_config[mode]

    # Tampilkan contoh pertanyaan jika chat kosong
    if not st.session_state.chat_history:
        st.markdown(
            f"<div style='background:rgba(0,255,136,0.05); border:1px solid rgba(0,255,136,0.15); "
            f"border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;'>"
            f"<div style='font-weight:600; color:#00ff88; margin-bottom:0.6rem;'>"
            f"{cfg['icon']} {mode}</div>"
            f"<div style='font-size:0.85rem; color:#8899b8; margin-bottom:0.8rem;'>{cfg['desc']}</div>"
            f"<div style='font-size:0.78rem; color:#6b7a99; margin-bottom:0.4rem;'>Quick examples:</div>",
            unsafe_allow_html=True
        )
        ex_cols = st.columns(len(cfg["examples"]))
        for i, (col, example) in enumerate(zip(ex_cols, cfg["examples"])):
            with col:
                if st.button(
                    f'"{example}"',
                    key=f"example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": example,
                        "timestamp": datetime.now().strftime("%H:%M"),
                        "mode": mode,
                    })
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat History Display ──────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            _render_message(msg)

    # ── Process Unanswered User Messages ─────────────────────
    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "user"
    ):
        _process_and_respond(mode)

    # ── Input Area ────────────────────────────────────────────
    _render_input_area(mode)


def _render_message(msg: Dict) -> None:
    """
    Render satu pesan chat bubble.

    Args:
        msg: Dict dengan keys: role, content, timestamp, mode,
             dan opsional: sources, sparql_used.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", "")
    sources = msg.get("sources", "")
    sparql_used = msg.get("sparql_used", "")

    if role == "user":
        st.markdown(
            f"<div style='display:flex; justify-content:flex-end; margin:0.5rem 0;'>"
            f"<div class='chat-bubble-user'>"
            f"<div style='font-size:0.78rem; color:#6b9fd4; margin-bottom:0.3rem;'>"
            f"👤 You · {timestamp}</div>"
            f"{content}"
            f"</div></div>",
            unsafe_allow_html=True
        )
    else:
        # Assistant bubble
        llm_label = {
            "gpt-4o-mini": "🤖 GPT-4o-mini",
            "mistral": "🦙 Mistral-7B",
        }.get(msg.get("llm", "gpt-4o-mini"), "🤖 AI")

        st.markdown(
            f"<div style='margin:0.5rem 0;'>"
            f"<div class='chat-bubble-assistant'>",
            unsafe_allow_html=True
        )

        # Header
        st.markdown(
            f"<div style='font-size:0.78rem; color:#6b9fd4; margin-bottom:0.6rem;'>"
            f"🛡️ SEPSES Assistant ({llm_label}) · {timestamp}</div>",
            unsafe_allow_html=True
        )

        # Content (support markdown)
        st.markdown(content)

        # Source citation (collapsible)
        if sources or sparql_used:
            with st.expander("📎 View Sources & SPARQL Query", expanded=False):
                if sparql_used and sparql_used != "N/A — Log analysis uses ChromaDB vector search":
                    st.markdown("**SPARQL Query Used:**")
                    st.code(sparql_used, language="sparql")
                if sources:
                    st.markdown("**Retrieved Context (KG Triples / Log Entries):**")
                    st.markdown(
                        f"<div class='source-box'>{sources}</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("</div></div>", unsafe_allow_html=True)


def _process_and_respond(mode: str) -> None:
    """
    Proses pertanyaan terakhir dari user dan tambahkan respons ke history.

    Args:
        mode: Mode analisis yang aktif.
    """
    last_msg = st.session_state.chat_history[-1]
    question = last_msg["content"]
    llm_name = st.session_state.selected_llm

    with st.spinner("🔍 Querying SEPSES Knowledge Graph..."):
        try:
            query_fn = _get_rag_pipeline(llm_name)
            answer, context, sparql_used = query_fn(question, mode)
        except Exception as exc:
            answer = (
                f"⚠️ **Error**: Tidak dapat memproses pertanyaan.\n\n"
                f"Detail: `{str(exc)}`\n\n"
                f"Pastikan `.env` sudah dikonfigurasi dan RAG pipeline tersedia."
            )
            context = ""
            sparql_used = ""

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().strftime("%H:%M"),
        "mode": mode,
        "llm": llm_name,
        "sources": context,
        "sparql_used": sparql_used,
    })
    st.rerun()


def _render_input_area(mode: str) -> None:
    """
    Render area input chat di bagian bawah halaman.

    Args:
        mode: Mode analisis yang aktif.
    """
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    placeholder_map = {
        "Security Analysis": "e.g., What attack patterns exploit CVE-2021-44228?",
        "Log Analysis": "e.g., What are the top threats in the uploaded logs?",
        "KG Question Answering": "e.g., How many CVEs are linked to CWE-89?",
    }

    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([9, 1])
        with col_input:
            user_input = st.text_input(
                label="chat_input",
                placeholder=placeholder_map.get(mode, "Type your security question..."),
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("➤", use_container_width=True)

    if submitted and user_input and user_input.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().strftime("%H:%M"),
            "mode": mode,
        })
        st.rerun()
