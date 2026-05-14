"""
SEPSES CSKG LLM Chatbot - ChromaDB Vector Store Wrapper
========================================================
Tanggung Jawab  : Satya Wira Pramudita (Evaluator & Log Dev)
Branch          : feature/eval-log-dev
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Wrapper ChromaDB untuk persistent storage dan semantic search
    atas security log entries yang telah diproses oleh log_parser.py.

    Fitur:
    - Persistent storage di path yang dikonfigurasi via .env
    - Embedding menggunakan sentence-transformers (lokal, tanpa API key)
    - Collection management per sumber log
    - Batch upsert dengan deduplication via doc_id
"""

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
import structlog
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from log_analysis.log_parser import LogEntry

load_dotenv()

# ============================================================
# Structured Logging
# ============================================================
logger = structlog.get_logger(__name__)


# ============================================================
# VectorStore Implementation
# ============================================================
class VectorStore:
    """
    ChromaDB wrapper untuk security log embeddings.

    Menyediakan operasi:
    - ingest      : Simpan batch LogEntry ke ChromaDB
    - search      : Semantic similarity search
    - delete_collection : Hapus seluruh collection
    - get_stats   : Info jumlah document per collection
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Inisialisasi ChromaDB client dan embedding model.

        Args:
            db_path         : Path direktori persistent ChromaDB.
                              Default dari env CHROMA_DB_PATH.
            embedding_model : Nama model sentence-transformers.
                              Default dari env EMBEDDING_MODEL.
            collection_name : Nama collection ChromaDB.
                              Default dari env CHROMA_COLLECTION_LOGS.

        Raises:
            OSError: Jika direktori db_path tidak dapat dibuat.
            RuntimeError: Jika embedding model gagal diload.
        """
        self._db_path = db_path or os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self._model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_LOGS", "security_logs"
        )

        logger.info(
            "vector_store_init",
            db_path=self._db_path,
            model=self._model_name,
            collection=self._collection_name
        )

        # Buat direktori jika belum ada
        try:
            os.makedirs(self._db_path, exist_ok=True)
        except OSError as exc:
            logger.error("chroma_dir_creation_failed", path=self._db_path, error=str(exc))
            raise

        # Init ChromaDB persistent client
        self._client = chromadb.PersistentClient(
            path=self._db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Load embedding model
        try:
            self._encoder = SentenceTransformer(self._model_name)
            logger.info("embedding_model_loaded", model=self._model_name)
        except Exception as exc:
            logger.error("embedding_model_load_failed", model=self._model_name, error=str(exc))
            raise RuntimeError(f"Gagal load embedding model '{self._model_name}': {exc}") from exc

        # Get or create collection
        self._collection = self._get_or_create_collection(self._collection_name)

    # ============================================================
    # Public API
    # ============================================================

    def ingest(self, entries: List[LogEntry], batch_size: int = 100) -> int:
        """
        Ingest batch LogEntry ke ChromaDB dengan embedding.

        Menggunakan doc_id sebagai ChromaDB ID untuk deduplication
        (upsert behavior: jika ID sama, data akan diupdate).

        Args:
            entries    : Daftar LogEntry dari log_parser.
            batch_size : Jumlah dokumen per batch untuk menghindari OOM.

        Returns:
            int: Jumlah dokumen yang berhasil di-ingest.

        Raises:
            ValueError: Jika entries kosong.
        """
        if not entries:
            logger.warning("ingest_empty_entries")
            raise ValueError("List entries tidak boleh kosong.")

        total_ingested = 0
        batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]

        for batch_idx, batch in enumerate(batches):
            logger.info(
                "ingesting_batch",
                batch_index=batch_idx + 1,
                total_batches=len(batches),
                batch_size=len(batch)
            )

            try:
                doc_texts = [entry.to_document_text() for entry in batch]
                doc_ids = [entry.doc_id for entry in batch]
                metadatas = [entry.to_metadata() for entry in batch]

                # Generate embeddings
                embeddings = self._encoder.encode(
                    doc_texts, batch_size=32, show_progress_bar=False
                ).tolist()

                # Upsert ke ChromaDB
                self._collection.upsert(
                    ids=doc_ids,
                    embeddings=embeddings,
                    documents=doc_texts,
                    metadatas=metadatas
                )
                total_ingested += len(batch)

            except Exception as exc:
                logger.error(
                    "batch_ingest_failed",
                    batch_index=batch_idx + 1,
                    error=str(exc)
                )
                raise

        logger.info("ingest_complete", total_ingested=total_ingested)
        return total_ingested

    def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search atas log entries.

        Args:
            query       : Query teks dalam natural language.
            top_k       : Jumlah hasil teratas yang dikembalikan.
            where_filter: Filter metadata ChromaDB (opsional).
                          Contoh: {"severity": "high"}

        Returns:
            List[Dict]: Daftar hasil dengan key:
                        - document  : Teks dokumen
                        - metadata  : Metadata log entry
                        - distance  : Cosine distance (makin kecil makin relevan)
                        - id        : Document ID

        Raises:
            ValueError: Jika query kosong.
            RuntimeError: Jika collection kosong.
        """
        if not query or not query.strip():
            raise ValueError("Query tidak boleh kosong.")

        collection_count = self._collection.count()
        if collection_count == 0:
            logger.warning("search_on_empty_collection", collection=self._collection_name)
            raise RuntimeError(
                "Collection kosong. Silakan ingest log terlebih dahulu."
            )

        logger.info(
            "vector_search",
            query=query[:100],
            top_k=top_k,
            collection_size=collection_count
        )

        try:
            query_embedding = self._encoder.encode([query]).tolist()

            kwargs: Dict[str, Any] = {
                "query_embeddings": query_embedding,
                "n_results": min(top_k, collection_count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            results = self._collection.query(**kwargs)

        except Exception as exc:
            logger.error("vector_search_failed", query=query[:100], error=str(exc))
            raise RuntimeError(f"Vector search gagal: {exc}") from exc

        # Flatten hasil ChromaDB
        formatted_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        logger.info("vector_search_complete", results_count=len(formatted_results))
        return formatted_results

    def delete_collection(self, confirm: bool = False) -> None:
        """
        Hapus seluruh collection dari ChromaDB.

        Args:
            confirm: Harus True untuk konfirmasi penghapusan (safety guard).

        Raises:
            ValueError: Jika confirm tidak True.
        """
        if not confirm:
            raise ValueError(
                "Operasi delete_collection memerlukan confirm=True sebagai safety guard."
            )
        logger.warning("deleting_collection", collection=self._collection_name)
        self._client.delete_collection(name=self._collection_name)
        # Re-create kosong
        self._collection = self._get_or_create_collection(self._collection_name)
        logger.info("collection_deleted_and_recreated", collection=self._collection_name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Ambil statistik collection saat ini.

        Returns:
            Dict dengan:
            - collection_name : Nama collection
            - document_count  : Jumlah dokumen
            - db_path         : Path ChromaDB
        """
        return {
            "collection_name": self._collection_name,
            "document_count": self._collection.count(),
            "db_path": self._db_path,
        }

    def switch_collection(self, collection_name: str) -> None:
        """
        Pindah ke collection lain (berguna untuk isolasi per sumber log).

        Args:
            collection_name: Nama collection baru.
        """
        logger.info(
            "switching_collection",
            from_collection=self._collection_name,
            to_collection=collection_name
        )
        self._collection_name = collection_name
        self._collection = self._get_or_create_collection(collection_name)

    # ============================================================
    # Private Methods
    # ============================================================

    def _get_or_create_collection(self, name: str):
        """
        Get existing collection atau create baru jika belum ada.

        Args:
            name: Nama collection ChromaDB.

        Returns:
            chromadb.Collection: Collection object.
        """
        try:
            collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Gunakan cosine similarity
            )
            logger.info(
                "collection_ready",
                collection=name,
                document_count=collection.count()
            )
            return collection
        except Exception as exc:
            logger.error("collection_init_failed", name=name, error=str(exc))
            raise
