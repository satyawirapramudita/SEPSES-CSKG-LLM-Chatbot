"""
SEPSES CSKG LLM Chatbot - RAG Pipeline (Orchestrator)
=======================================================
Tanggung Jawab  : Fahmi Abdillah Zain (RAG Logic Dev)
Branch          : feature/rag-logic
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Orchestrator utama yang mengintegrasikan seluruh komponen:
    - KG Engine (SPARQL) → structured context
    - Log Vector Store (ChromaDB) → log context
    - NL2SPARQL → query generation
    - Multi-Hop Traversal → attack chain
    - LLM Connector → answer generation

    Alur:
    1. user_query → NL2SPARQL → SPARQL query
    2. SPARQL query → SparqlClient → KG context
    3. user_query → HybridRetriever → Log context (jika mode Log Analysis)
    4. [KG context + Log context] → LLM → answer
    5. return {answer, context, sparql_used, latency_ms}

    Graceful Degradation:
    - Jika SPARQL endpoint down → skip KG context, tetap answer dengan LLM
    - Jika ChromaDB kosong → skip log context
    - Setiap error dikatchup dan di-log, tidak pernah crash ke user
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)


# ============================================================
# RAG Pipeline
# ============================================================
class RagPipeline:
    """
    Main RAG + GraphRAG orchestrator.

    Mode yang didukung:
    - "Security Analysis"     : KG context (CVE/CWE/CAPEC chain) + LLM
    - "Log Analysis"          : Log context (ChromaDB) + KG enrichment + LLM
    - "KG Question Answering" : KG context + LLM (SPARQL-grounded)
    """

    SUPPORTED_MODES = {"Security Analysis", "Log Analysis", "KG Question Answering"}

    def __init__(
        self,
        llm_name: Optional[str] = None,
        sparql_client=None,
        retriever=None,
    ) -> None:
        """
        Inisialisasi pipeline dengan lazy-loading semua komponen.

        Args:
            llm_name      : "gpt-4o-mini" | "mistral" | dsb.
            sparql_client : Pre-initialized SparqlClient (opsional).
            retriever     : Pre-initialized HybridRetriever (opsional).
        """
        self._llm_name = llm_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._llm      = None   # lazy init
        self._client   = sparql_client  # lazy init
        self._retriever = retriever     # lazy init
        self._nl2sparql = None  # lazy init
        self._multi_hop = None  # lazy init
        logger.info("rag_pipeline_created", llm=self._llm_name)

    # ============================================================
    # Public API
    # ============================================================
    def query(
        self,
        question: str,
        mode: str = "Security Analysis",
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Entry point utama pipeline.

        Args:
            question     : Pertanyaan natural language dari user.
            mode         : Analysis mode (lihat SUPPORTED_MODES).
            chat_history : Riwayat percakapan sebelumnya (opsional).

        Returns:
            Dict dengan keys:
                - answer      (str)  : Respons LLM
                - context     (str)  : Context yang diambil dari KG/log
                - sparql_used (str)  : SPARQL query yang digunakan
                - latency_ms  (float): Total waktu pemrosesan
                - mode        (str)  : Mode yang digunakan
                - llm         (str)  : Nama LLM yang digunakan
                - error       (str)  : Error message jika ada

        Raises:
            ValueError: Jika mode tidak valid.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Mode '{mode}' tidak valid. Gunakan salah satu: {self.SUPPORTED_MODES}"
            )

        start_time = time.time()
        logger.info("rag_query_start", mode=mode, question=question[:80])

        try:
            if mode == "Security Analysis":
                result = self._security_analysis_flow(question, chat_history)
            elif mode == "Log Analysis":
                result = self._log_analysis_flow(question, chat_history)
            else:  # KG Question Answering
                result = self._kg_qa_flow(question, chat_history)

        except Exception as exc:
            logger.error("rag_pipeline_error", error=str(exc), mode=mode)
            result = {
                "answer": (
                    f"⚠️ **Pipeline Error**: {str(exc)}\n\n"
                    f"Pastikan:\n"
                    f"- File `.env` sudah dikonfigurasi dengan API key\n"
                    f"- Ollama berjalan (jika menggunakan Mistral)\n"
                    f"- Koneksi internet tersedia untuk SPARQL endpoint"
                ),
                "context":     "",
                "sparql_used": "",
                "error":       str(exc),
            }

        latency_ms = round((time.time() - start_time) * 1000, 2)
        result["latency_ms"] = latency_ms
        result["mode"]       = mode
        result["llm"]        = self._llm_name

        logger.info("rag_query_done", latency_ms=latency_ms, mode=mode)
        return result

    # ============================================================
    # Flow per Mode
    # ============================================================
    def _security_analysis_flow(
        self, question: str, chat_history: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Flow: Security Analysis.
        CVE ID detected → multi-hop chain → LLM answer.
        """
        from rag_logic.prompt_templates import (
            SYSTEM_SECURITY_ANALYSIS,
            security_analysis_prompt,
        )

        # 1. Deteksi CVE ID dalam pertanyaan
        cve_ids = self._extract_cve_ids(question)
        kg_context = ""
        sparql_used = ""

        if cve_ids:
            # 2. Bangun attack chain dari CVE
            chain = self._get_multi_hop().build_attack_chain(cve_ids[0])
            kg_context  = chain.get("context_str", "")
            sparql_used = f"Multi-hop traversal: {cve_ids[0]} → CWE → CAPEC → ATT&CK"
        else:
            # 3. Fallback: cari berdasarkan keyword
            kg_context  = self._get_multi_hop().build_threat_context(question)
            sparql_used = "Keyword-based product/threat search"

            # Tambahan: coba NL2SPARQL untuk context lebih kaya
            try:
                sparql_q, method = self._get_nl2sparql().convert(question)
                rows = self._get_sparql_client().execute_query(sparql_q)
                if rows:
                    from rag_logic.prompt_templates import format_kg_context
                    extra = format_kg_context(rows[:8])
                    kg_context = f"{kg_context}\n\n{extra}"
                    sparql_used = f"{sparql_used} + NL2SPARQL ({method})"
            except Exception as exc:
                logger.warning("nl2sparql_failed_in_flow", error=str(exc))

        # 4. Generate answer
        messages = self._build_messages(
            system_prompt=SYSTEM_SECURITY_ANALYSIS,
            user_prompt=security_analysis_prompt(question, kg_context),
            chat_history=chat_history,
        )
        answer = self._get_llm().generate(messages)

        return {"answer": answer, "context": kg_context, "sparql_used": sparql_used}

    def _log_analysis_flow(
        self, question: str, chat_history: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Flow: Log Analysis.
        Retriever → log context → KG enrichment → LLM answer.
        """
        from rag_logic.prompt_templates import SYSTEM_LOG_ANALYSIS, log_analysis_prompt

        # 1. Retrieve relevant log entries
        log_context = ""
        try:
            retriever = self._get_retriever()
            results = retriever.search(question, top_k=5)
            if results:
                log_lines = []
                for r in results:
                    meta = r.get("metadata", {})
                    log_lines.append(
                        f"[{meta.get('severity','?').upper()}] "
                        f"Src: {meta.get('source_ip','?')} → "
                        f"Dst: {meta.get('dest_ip','?')} | "
                        f"{r.get('document','')}"
                    )
                log_context = "\n".join(log_lines)
        except Exception as exc:
            logger.warning("log_retrieval_failed", error=str(exc))
            log_context = "No log data available (ChromaDB not initialized or empty)."

        # 2. KG enrichment: deteksi CVE dalam query/log
        kg_context = ""
        cve_ids = self._extract_cve_ids(question + " " + log_context)
        if cve_ids:
            try:
                chain = self._get_multi_hop().build_attack_chain(cve_ids[0])
                kg_context = chain.get("context_str", "")
            except Exception as exc:
                logger.warning("kg_enrichment_failed", error=str(exc))

        # 3. Generate answer
        messages = self._build_messages(
            system_prompt=SYSTEM_LOG_ANALYSIS,
            user_prompt=log_analysis_prompt(question, log_context, kg_context),
            chat_history=chat_history,
        )
        answer = self._get_llm().generate(messages)

        return {
            "answer":      answer,
            "context":     log_context,
            "sparql_used": "N/A — ChromaDB hybrid search",
        }

    def _kg_qa_flow(
        self, question: str, chat_history: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Flow: KG Question Answering.
        NL2SPARQL → SPARQL execution → LLM answer.
        """
        from rag_logic.prompt_templates import (
            SYSTEM_KG_QA,
            kg_qa_prompt,
            format_kg_context,
        )

        # 1. Convert NL → SPARQL
        sparql_q = ""
        sparql_method = "fallback"
        kg_context = ""

        try:
            sparql_q, sparql_method = self._get_nl2sparql().convert(question)
        except Exception as exc:
            logger.warning("nl2sparql_failed", error=str(exc))
            sparql_q = ""

        # 2. Execute SPARQL
        if sparql_q:
            try:
                rows = self._get_sparql_client().execute_query(sparql_q)
                kg_context = format_kg_context(rows[:10]) if rows else "No results returned."
            except Exception as exc:
                logger.warning("sparql_execution_failed", error=str(exc))
                kg_context = f"SPARQL execution failed: {exc}"
        else:
            kg_context = "Could not generate SPARQL query for this question."

        # 3. Generate answer
        messages = self._build_messages(
            system_prompt=SYSTEM_KG_QA,
            user_prompt=kg_qa_prompt(question, kg_context),
            chat_history=chat_history,
        )
        answer = self._get_llm().generate(messages)

        return {
            "answer":      answer,
            "context":     kg_context,
            "sparql_used": f"{sparql_q[:300]}..." if len(sparql_q) > 300 else sparql_q,
        }

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _extract_cve_ids(text: str) -> List[str]:
        """Extract semua CVE ID yang ditemukan dalam teks."""
        return list(dict.fromkeys(
            m.upper() for m in re.findall(r"CVE-\d{4}-\d{4,7}", text, re.IGNORECASE)
        ))

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        chat_history: Optional[List[Dict]] = None,
    ):
        """
        Build list Message objects untuk LLM.

        Args:
            system_prompt: System prompt string.
            user_prompt  : User prompt string.
            chat_history : List of {"role": ..., "content": ...} dicts.

        Returns:
            List[Message]
        """
        from rag_logic.llm_connector import Message

        messages = [Message("system", system_prompt)]

        # Append recent history (max 6 turns = 3 pairs)
        if chat_history:
            for msg in chat_history[-6:]:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append(Message(role, content))

        messages.append(Message("user", user_prompt))
        return messages

    # ── Lazy Component Loaders ───────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from rag_logic.llm_connector import get_llm_connector
            self._llm = get_llm_connector(self._llm_name)
        return self._llm

    def _get_sparql_client(self):
        if self._client is None:
            from kg_engine.sparql_client import SparqlClient
            self._client = SparqlClient()
        return self._client

    def _get_retriever(self):
        if self._retriever is None:
            from log_analysis.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever

    def _get_nl2sparql(self):
        if self._nl2sparql is None:
            from rag_logic.nl2sparql import NL2SPARQL
            self._nl2sparql = NL2SPARQL(llm=self._get_llm())
        return self._nl2sparql

    def _get_multi_hop(self):
        if self._multi_hop is None:
            from rag_logic.multi_hop import MultiHopTraversal
            self._multi_hop = MultiHopTraversal(sparql_client=self._get_sparql_client())
        return self._multi_hop
