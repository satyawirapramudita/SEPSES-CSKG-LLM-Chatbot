"""
SEPSES CSKG LLM Chatbot - Hybrid Log Retriever
===============================================
Tanggung Jawab  : Satya Wira Pramudita (Evaluator & Log Dev)
Branch          : feature/eval-log-dev
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Hybrid retriever yang menggabungkan:
    1. BM25 (keyword/sparse retrieval) via rank-bm25
    2. Semantic search (dense retrieval) via ChromaDB
    3. Reciprocal Rank Fusion (RRF) untuk re-ranking hasil

    Pendekatan hybrid ini lebih robust dibanding single-method:
    - BM25 bagus untuk exact match (CVE ID, IP address)
    - Dense retrieval bagus untuk semantic similarity
    - RRF menggabungkan keduanya tanpa memerlukan threshold tuning
"""

import os
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from log_analysis.log_parser import LogEntry, LogParser, LogType
from log_analysis.vector_store import VectorStore

load_dotenv()

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """
    Hybrid BM25 + Semantic retrieval dengan Reciprocal Rank Fusion (RRF).

    Alur Kerja:
    1. ingest_logs() → parse file → store ke ChromaDB + build BM25 index
    2. search(query) → parallel BM25 + semantic search → RRF fusion → return top-k

    RRF Formula: score(d) = Σ 1/(k + rank_i(d))
    di mana k=60 (konstanta RRF) dan rank_i adalah peringkat di retriever ke-i.
    """

    RRF_K = 60  # Konstanta RRF standar (Robertson & Zaragoza, 2009)

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """
        Inisialisasi HybridRetriever.

        Args:
            vector_store: Instance VectorStore (ChromaDB). Jika None, dibuat baru.
            top_k       : Jumlah default hasil retrieval.
                          Default dari env TOP_K_RETRIEVAL (5).
        """
        self._vector_store = vector_store or VectorStore()
        self._top_k = top_k or int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self._parser = LogParser()

        # BM25 index (di-build saat ingest)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []       # Teks dokumen per-index
        self._bm25_entries: List[LogEntry] = []  # LogEntry per-index (untuk metadata)

        logger.info("hybrid_retriever_init", top_k=self._top_k)

    # ============================================================
    # Public API
    # ============================================================

    def ingest_logs(
        self,
        file_path: Optional[str] = None,
        log_text: Optional[str] = None,
        log_type: LogType = LogType.UNKNOWN,
    ) -> int:
        """
        Ingest log dari file atau string teks ke VectorStore + BM25 index.

        Args:
            file_path : Path ke file log (opsional, exclusive dengan log_text).
            log_text  : Konten log sebagai string (opsional).
            log_type  : Tipe log (digunakan jika log_text diberikan).

        Returns:
            int: Jumlah log entry yang berhasil diingest.

        Raises:
            ValueError: Jika keduanya file_path dan log_text kosong, atau keduanya diisi.
        """
        if not file_path and not log_text:
            raise ValueError("Harus menyediakan file_path atau log_text.")
        if file_path and log_text:
            raise ValueError("Hanya boleh salah satu: file_path atau log_text.")

        # Parse log
        if file_path:
            entries = self._parser.parse_file(file_path)
        else:
            entries = self._parser.parse_text(log_text, log_type)

        if not entries:
            logger.warning("no_entries_after_parsing")
            return 0

        logger.info("ingesting_entries", count=len(entries))

        # 1. Simpan ke ChromaDB (dense retrieval)
        ingested = self._vector_store.ingest(entries)

        # 2. Build/update BM25 index (sparse retrieval)
        self._build_bm25_index(entries)

        logger.info("ingest_complete", ingested=ingested)
        return ingested

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        severity_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: BM25 + Semantic, di-fuse dengan RRF.

        Args:
            query          : Query dalam natural language atau keyword.
            top_k          : Jumlah hasil (default: self._top_k).
            severity_filter: Filter berdasarkan severity level (opsional).
                             Nilai: "critical", "high", "medium", "low", "info".

        Returns:
            List[Dict]: Daftar hasil dengan key:
                        - document  : Teks log entry
                        - metadata  : Metadata entry
                        - rrf_score : Skor gabungan RRF
                        - id        : Document ID
                        - sources   : ["bm25", "semantic"] yang berkontribusi

        Raises:
            ValueError: Jika query kosong.
            RuntimeError: Jika belum ada data yang diingest.
        """
        if not query or not query.strip():
            raise ValueError("Query tidak boleh kosong.")

        k = top_k or self._top_k

        # Cek ketersediaan data
        stats = self._vector_store.get_stats()
        if stats["document_count"] == 0:
            raise RuntimeError(
                "Belum ada data log. Silakan ingest log terlebih dahulu."
            )

        logger.info(
            "hybrid_search_start",
            query=query[:100],
            top_k=k,
            severity_filter=severity_filter
        )

        # Siapkan filter ChromaDB jika ada severity filter
        where_filter = {"severity": severity_filter} if severity_filter else None

        # 1. Semantic search (ChromaDB)
        semantic_candidates = self._semantic_search(query, top_k=k * 2, where_filter=where_filter)

        # 2. BM25 keyword search
        bm25_candidates = self._bm25_search(query, top_k=k * 2)

        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_candidates, bm25_candidates, top_k=k
        )

        logger.info(
            "hybrid_search_complete",
            semantic_hits=len(semantic_candidates),
            bm25_hits=len(bm25_candidates),
            fused_results=len(fused_results)
        )

        return fused_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Ambil statistik retriever saat ini.

        Returns:
            Dict dengan info vector store dan BM25 index.
        """
        vs_stats = self._vector_store.get_stats()
        return {
            **vs_stats,
            "bm25_corpus_size": len(self._bm25_corpus),
            "bm25_index_ready": self._bm25_index is not None,
        }

    # ============================================================
    # Private Methods
    # ============================================================

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Jalankan semantic search via ChromaDB.

        Args:
            query       : Query teks.
            top_k       : Jumlah kandidat.
            where_filter: Filter metadata ChromaDB.

        Returns:
            List[Dict]: Hasil semantic search.
        """
        try:
            results = self._vector_store.search(query, top_k=top_k, where_filter=where_filter)
            return results
        except RuntimeError:
            # Collection kosong, kembalikan hasil kosong
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Jalankan BM25 keyword search.

        Args:
            query : Query teks.
            top_k : Jumlah kandidat.

        Returns:
            List[Dict]: Hasil BM25 search dengan skor.
        """
        if not self._bm25_index or not self._bm25_corpus:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25_index.get_scores(tokenized_query)

            # Ambil top-k dengan skor > 0
            indexed_scores = [
                (i, score) for i, score in enumerate(scores) if score > 0
            ]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = indexed_scores[:top_k]

            results = []
            for idx, score in top_results:
                entry = self._bm25_entries[idx]
                results.append({
                    "id": entry.doc_id,
                    "document": entry.to_document_text(),
                    "metadata": entry.to_metadata(),
                    "bm25_score": score,
                })
            return results

        except Exception as exc:
            logger.error("bm25_search_failed", error=str(exc))
            return []

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Gabungkan hasil semantic dan BM25 menggunakan Reciprocal Rank Fusion.

        RRF Score: score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))
        Konstanta k=60 mencegah dominasi hasil dengan ranking sangat tinggi.

        Args:
            semantic_results: Hasil dari ChromaDB (semantic search).
            bm25_results    : Hasil dari BM25 (keyword search).
            top_k           : Jumlah hasil akhir.

        Returns:
            List[Dict]: Hasil gabungan yang sudah di-sort berdasarkan RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        doc_sources: Dict[str, List[str]] = {}

        # Proses semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (self.RRF_K + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
            doc_data[doc_id] = result
            doc_sources.setdefault(doc_id, []).append("semantic")

        # Proses BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (self.RRF_K + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result
            doc_sources.setdefault(doc_id, []).append("bm25")

        # Sort berdasarkan RRF score (descending)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids[:top_k]:
            result = doc_data[doc_id].copy()
            result["rrf_score"] = round(rrf_scores[doc_id], 6)
            result["sources"] = doc_sources.get(doc_id, [])
            fused.append(result)

        return fused

    def _build_bm25_index(self, new_entries: List[LogEntry]) -> None:
        """
        Build atau update BM25 index dengan entries baru.

        BM25 index di-build in-memory dari seluruh corpus (append-only).

        Args:
            new_entries: LogEntry baru yang akan ditambahkan ke index.
        """
        # Tambahkan ke corpus yang ada
        for entry in new_entries:
            self._bm25_corpus.append(entry.to_document_text())
            self._bm25_entries.append(entry)

        # Re-build BM25 index dari seluruh corpus
        if self._bm25_corpus:
            tokenized_corpus = [doc.lower().split() for doc in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("bm25_index_built", corpus_size=len(self._bm25_corpus))
