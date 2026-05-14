"""
SEPSES CSKG LLM Chatbot - SPARQL Client
========================================
Tanggung Jawab  : Ajie Armansyah Sunaryo (Knowledge Architect)
Branch          : feature/kg-engine
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Client untuk SEPSES SPARQL endpoint publik dengan:
    - Auto-fallback ke Apache Jena Fuseki lokal jika endpoint publik down
    - Query builder menggunakan template file .rq
    - Caching hasil query (TTL 5 menit) untuk mengurangi beban endpoint
    - Semua metode public mengembalikan dict/list yang clean (bukan raw JSON)
"""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv
from SPARQLWrapper import JSON, SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException

load_dotenv()

logger = structlog.get_logger(__name__)

# ── Namespace Prefixes ──────────────────────────────────────
PREFIXES = """
PREFIX cve:   <http://w3id.org/sepses/vocab/ref/cve#>
PREFIX cwe:   <http://w3id.org/sepses/vocab/ref/cwe#>
PREFIX capec: <http://w3id.org/sepses/vocab/ref/capec#>
PREFIX cpe:   <http://w3id.org/sepses/vocab/ref/cpe#>
PREFIX cvss:  <http://w3id.org/sepses/vocab/ref/cvss#>
PREFIX res:   <http://w3id.org/sepses/resource/>
PREFIX rescve: <http://w3id.org/sepses/resource/cve/>
PREFIX rescwe: <http://w3id.org/sepses/resource/cwe/>
"""

# ── Resource URI Builder ────────────────────────────────────
CVE_URI_BASE   = "http://w3id.org/sepses/resource/cve/"
CWE_URI_BASE   = "http://w3id.org/sepses/resource/cwe/"
CAPEC_URI_BASE = "http://w3id.org/sepses/resource/capec/"

# ── Query Cache (simple in-memory, TTL-based) ───────────────
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 300  # seconds


class SparqlClient:
    """
    Client SPARQL untuk SEPSES Knowledge Graph.

    Strategi endpoint:
    1. Coba endpoint publik (SPARQL_ENDPOINT dari .env)
    2. Jika timeout/error → fallback ke Fuseki lokal (FUSEKI_ENDPOINT)
    3. Jika keduanya gagal → raise RuntimeError dengan pesan jelas
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        fallback_endpoint: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Inisialisasi SPARQL client.

        Args:
            endpoint         : URL endpoint publik SEPSES.
            fallback_endpoint: URL Fuseki lokal sebagai fallback.
            timeout          : Request timeout dalam detik.
        """
        self._endpoint = endpoint or os.getenv(
            "SPARQL_ENDPOINT", "https://w3id.org/sepses/sparql"
        )
        self._fallback = fallback_endpoint or os.getenv(
            "FUSEKI_ENDPOINT", "http://localhost:3030/sepses/sparql"
        )
        self._timeout = timeout or int(os.getenv("SPARQL_TIMEOUT_SECONDS", "30"))
        self._queries_dir = Path(__file__).parent / "queries"

        logger.info(
            "sparql_client_init",
            endpoint=self._endpoint,
            fallback=self._fallback,
            timeout=self._timeout,
        )

    # ============================================================
    # Core Query Executor
    # ============================================================
    def execute_query(self, sparql_str: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Eksekusi SPARQL SELECT query dan kembalikan hasil sebagai list of dicts.

        Args:
            sparql_str: String SPARQL query lengkap.
            use_cache : Gunakan cache jika True (default True).

        Returns:
            List[Dict]: Setiap dict merepresentasikan satu baris hasil.
                        Literal values sudah di-extract dari SPARQLWrapper format.

        Raises:
            RuntimeError: Jika kedua endpoint (publik + fallback) gagal.
        """
        # Cache check
        cache_key = sparql_str.strip()
        if use_cache and cache_key in _QUERY_CACHE:
            cached = _QUERY_CACHE[cache_key]
            if time.time() - cached["timestamp"] < _CACHE_TTL:
                logger.debug("cache_hit", query_preview=sparql_str[:80])
                return cached["results"]

        # Coba endpoint publik dulu
        result = self._try_endpoint(self._endpoint, sparql_str)
        if result is None:
            logger.warning("public_endpoint_failed_trying_fallback", endpoint=self._endpoint)
            result = self._try_endpoint(self._fallback, sparql_str)

        if result is None:
            raise RuntimeError(
                f"Kedua SPARQL endpoint tidak tersedia.\n"
                f"Publik: {self._endpoint}\n"
                f"Fallback: {self._fallback}\n"
                f"Jalankan Fuseki lokal atau periksa koneksi internet."
            )

        # Flatten hasil
        rows = self._flatten_results(result)

        # Simpan ke cache
        if use_cache:
            _QUERY_CACHE[cache_key] = {"timestamp": time.time(), "results": rows}

        logger.info("query_executed", rows_returned=len(rows), query_preview=sparql_str[:80])
        return rows

    # ============================================================
    # High-Level API Methods
    # ============================================================
    def get_cve_details(self, cve_id: str) -> Dict[str, Any]:
        """
        Ambil detail lengkap satu CVE: deskripsi, CWE, CAPEC, CVSS, CPE.

        Args:
            cve_id: CVE identifier string, e.g. "CVE-2021-44228".

        Returns:
            Dict dengan keys: cve_id, description, issued, cwes, capecs,
                              cvss_score, attack_vector, cpe_products.

        Raises:
            ValueError : Jika format cve_id tidak valid.
            RuntimeError: Jika endpoint tidak tersedia.
        """
        cve_id = cve_id.strip().upper()
        if not cve_id.startswith("CVE-"):
            raise ValueError(f"Format CVE ID tidak valid: '{cve_id}'. Gunakan format CVE-YYYY-NNNN.")

        cve_uri = f"<{CVE_URI_BASE}{cve_id}>"

        sparql = f"""
{PREFIXES}
SELECT DISTINCT
    ?description ?issued
    ?cweId ?cweName
    ?capecId ?capecName
    ?score ?attackVector ?cpeProduct
WHERE {{
    {cve_uri} cve:cveId "{cve_id}" .
    OPTIONAL {{ {cve_uri} cve:description ?description . }}
    OPTIONAL {{ {cve_uri} cve:issued     ?issued . }}
    OPTIONAL {{
        {cve_uri} cve:hasCWE ?cwe .
        OPTIONAL {{ ?cwe cwe:cweId ?cweId . }}
        OPTIONAL {{ ?cwe cwe:name  ?cweName . }}
        OPTIONAL {{
            ?cwe cwe:hasCAPEC ?capec .
            OPTIONAL {{ ?capec capec:capecId ?capecId . }}
            OPTIONAL {{ ?capec capec:name    ?capecName . }}
        }}
    }}
    OPTIONAL {{
        {cve_uri} cve:hasCVSS ?cvssNode .
        OPTIONAL {{ ?cvssNode cvss:baseScore    ?score . }}
        OPTIONAL {{ ?cvssNode cvss:attackVector ?attackVector . }}
    }}
    OPTIONAL {{
        {cve_uri} cve:hasCPE ?cpe .
        OPTIONAL {{ ?cpe cpe:productId ?cpeProduct . }}
    }}
}}
LIMIT 50
"""
        rows = self.execute_query(sparql)

        if not rows:
            logger.warning("cve_not_found", cve_id=cve_id)
            return {
                "cve_id": cve_id,
                "description": f"CVE {cve_id} tidak ditemukan di SEPSES KG.",
                "issued": None,
                "cwes": [],
                "capecs": [],
                "cvss_score": None,
                "attack_vector": None,
                "cpe_products": [],
                "found": False,
            }

        # Aggregate multi-value columns
        cwes, capecs, products = set(), set(), set()
        description = rows[0].get("description", "")
        issued = rows[0].get("issued", "")
        score = rows[0].get("score")
        attack_vector = rows[0].get("attackVector")

        for row in rows:
            if row.get("cweId"):
                cwes.add(f"{row['cweId']}: {row.get('cweName', '')}")
            if row.get("capecId"):
                capecs.add(f"{row['capecId']}: {row.get('capecName', '')}")
            if row.get("cpeProduct"):
                products.add(row["cpeProduct"])

        logger.info(
            "cve_details_retrieved",
            cve_id=cve_id,
            cwes=len(cwes),
            capecs=len(capecs),
            score=score,
        )

        return {
            "cve_id": cve_id,
            "description": description,
            "issued": issued,
            "cwes": sorted(cwes),
            "capecs": sorted(capecs),
            "cvss_score": float(score) if score else None,
            "attack_vector": attack_vector,
            "cpe_products": sorted(products),
            "found": True,
        }

    def get_capec_from_cve(self, cve_id: str) -> List[Dict[str, str]]:
        """
        Multi-hop traversal: CVE → CWE → CAPEC.

        Args:
            cve_id: CVE identifier, e.g. "CVE-2021-44228".

        Returns:
            List[Dict]: Setiap dict memiliki keys: cwe_id, cwe_name, capec_id, capec_name, capec_desc.
        """
        cve_id = cve_id.strip().upper()
        cve_uri = f"<{CVE_URI_BASE}{cve_id}>"

        sparql = f"""
{PREFIXES}
SELECT DISTINCT ?cweId ?cweName ?capecId ?capecName ?capecDescription
WHERE {{
    {cve_uri} cve:hasCWE ?cwe .
    OPTIONAL {{ ?cwe cwe:cweId ?cweId . }}
    OPTIONAL {{ ?cwe cwe:name  ?cweName . }}
    OPTIONAL {{
        ?cwe cwe:hasCAPEC ?capec .
        OPTIONAL {{ ?capec capec:capecId    ?capecId . }}
        OPTIONAL {{ ?capec capec:name       ?capecName . }}
        OPTIONAL {{ ?capec capec:description ?capecDescription . }}
    }}
}}
ORDER BY ?cweId ?capecId
"""
        rows = self.execute_query(sparql)
        results = []
        for row in rows:
            results.append({
                "cwe_id":     row.get("cweId", ""),
                "cwe_name":   row.get("cweName", ""),
                "capec_id":   row.get("capecId", ""),
                "capec_name": row.get("capecName", ""),
                "capec_desc": row.get("capecDescription", ""),
            })
        logger.info("capec_chain_retrieved", cve_id=cve_id, chain_length=len(results))
        return results

    def search_vulnerabilities_by_product(
        self,
        product_name: str,
        min_score: float = 0.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Cari CVE yang mempengaruhi produk tertentu, diurutkan CVSS descending.

        Args:
            product_name: Nama produk/vendor, e.g. "apache", "windows_10".
            min_score   : Filter CVSS score minimum (0.0–10.0).
            limit       : Maksimum jumlah hasil.

        Returns:
            List[Dict]: cve_id, description, score, cpe_product per item.
        """
        sparql = f"""
{PREFIXES}
SELECT DISTINCT ?cveId ?description ?score ?cpeProduct
WHERE {{
    ?cve cve:cveId  ?cveId ;
         cve:hasCPE ?cpe .
    ?cpe cpe:productId ?cpeProduct .
    OPTIONAL {{ ?cve cve:description ?description . }}
    OPTIONAL {{
        ?cve cve:hasCVSS ?cvssNode .
        ?cvssNode cvss:baseScore ?score .
    }}
    FILTER(CONTAINS(LCASE(STR(?cpeProduct)), LCASE("{product_name}")))
    FILTER(!BOUND(?score) || ?score >= {min_score})
}}
ORDER BY DESC(?score)
LIMIT {limit}
"""
        rows = self.execute_query(sparql)
        logger.info(
            "product_search_done",
            product=product_name,
            min_score=min_score,
            results=len(rows),
        )
        return [
            {
                "cve_id":      r.get("cveId", ""),
                "description": r.get("description", ""),
                "score":       float(r["score"]) if r.get("score") else None,
                "product":     r.get("cpeProduct", ""),
            }
            for r in rows
        ]

    def get_cwes_by_capec(self, capec_id: str) -> List[Dict[str, str]]:
        """
        Reverse lookup: CAPEC → CWE → CVE (untuk analisis threat actor).

        Args:
            capec_id: CAPEC identifier, e.g. "CAPEC-88".

        Returns:
            List[Dict]: cwe_id, cwe_name, cve_id per item.
        """
        capec_uri = f"<{CAPEC_URI_BASE}{capec_id}>"

        sparql = f"""
{PREFIXES}
SELECT DISTINCT ?cweId ?cweName ?cveId
WHERE {{
    ?cwe cwe:hasCAPEC {capec_uri} .
    OPTIONAL {{ ?cwe cwe:cweId ?cweId . }}
    OPTIONAL {{ ?cwe cwe:name  ?cweName . }}
    OPTIONAL {{
        ?cve cve:hasCWE ?cwe .
        ?cve cve:cveId  ?cveId .
    }}
}}
LIMIT 30
"""
        rows = self.execute_query(sparql)
        return [
            {
                "cwe_id":   r.get("cweId", ""),
                "cwe_name": r.get("cweName", ""),
                "cve_id":   r.get("cveId", ""),
            }
            for r in rows
        ]

    def get_top_cvss_cves(self, min_score: float = 9.0, limit: int = 10) -> List[Dict]:
        """
        Ambil CVE dengan skor CVSS tertinggi.

        Args:
            min_score: Minimum CVSS score.
            limit    : Jumlah maksimum hasil.

        Returns:
            List[Dict]: cve_id, score per item, sorted descending.
        """
        sparql = f"""
{PREFIXES}
SELECT DISTINCT ?cveId ?score
WHERE {{
    ?cve cve:cveId  ?cveId ;
         cve:hasCVSS ?cvssNode .
    ?cvssNode cvss:baseScore ?score .
    FILTER(?score >= {min_score})
}}
ORDER BY DESC(?score)
LIMIT {limit}
"""
        rows = self.execute_query(sparql)
        return [
            {"cve_id": r.get("cveId", ""), "score": float(r.get("score", 0))}
            for r in rows
        ]

    def ping(self) -> bool:
        """
        Cek apakah SPARQL endpoint aktif.

        Returns:
            bool: True jika endpoint merespons dengan benar.
        """
        try:
            rows = self.execute_query(
                f"{PREFIXES}\nSELECT (COUNT(?s) AS ?n) WHERE {{ ?s a cve:CVE . }} LIMIT 1",
                use_cache=False,
            )
            return len(rows) > 0
        except Exception:
            return False

    # ============================================================
    # Private Helpers
    # ============================================================
    def _try_endpoint(self, endpoint_url: str, sparql_str: str) -> Optional[Dict]:
        """
        Coba eksekusi query ke satu endpoint.

        Args:
            endpoint_url: URL SPARQL endpoint.
            sparql_str  : Query string.

        Returns:
            Raw SPARQLWrapper JSON response dict, atau None jika gagal.
        """
        try:
            wrapper = SPARQLWrapper(endpoint_url)
            wrapper.setQuery(sparql_str)
            wrapper.setReturnFormat(JSON)
            wrapper.setTimeout(self._timeout)
            response = wrapper.query().convert()
            logger.debug("endpoint_success", endpoint=endpoint_url)
            return response
        except SPARQLWrapperException as exc:
            logger.warning("sparql_wrapper_error", endpoint=endpoint_url, error=str(exc))
            return None
        except Exception as exc:
            logger.warning("endpoint_unavailable", endpoint=endpoint_url, error=str(exc))
            return None

    @staticmethod
    def _flatten_results(raw: Dict) -> List[Dict[str, Any]]:
        """
        Flatten SPARQLWrapper JSON result ke list of plain dicts.

        SPARQLWrapper mengembalikan:
        {
          "results": {
            "bindings": [
              {"var": {"type": "literal", "value": "..."}, ...},
              ...
            ]
          }
        }

        Args:
            raw: Raw response dari SPARQLWrapper.

        Returns:
            List[Dict]: Setiap dict berisi {varName: value} (plain Python values).
        """
        bindings = raw.get("results", {}).get("bindings", [])
        rows = []
        for binding in bindings:
            row = {}
            for var_name, val_obj in binding.items():
                row[var_name] = val_obj.get("value", "")
            rows.append(row)
        return rows
