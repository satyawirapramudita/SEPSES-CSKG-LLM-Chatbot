"""
SEPSES CSKG LLM Chatbot - NL2SPARQL Converter
===============================================
Tanggung Jawab  : Fahmi Abdillah Zain (RAG Logic Dev)
Branch          : feature/rag-logic

Deskripsi:
    Mengkonversi pertanyaan natural language ke SPARQL query
    menggunakan LLM (few-shot prompting) + regex-based fallback
    untuk pertanyaan yang pola umumnya sudah diketahui.

    Strategi:
    1. Regex pattern matching untuk pola umum (fast, no API call)
    2. LLM-based generation dengan few-shot prompt untuk kasus kompleks
    3. Validasi hasil query sebelum dieksekusi
"""

import re
from typing import Optional, Tuple

import structlog

from rag_logic.llm_connector import BaseLLMConnector, Message
from rag_logic.prompt_templates import SYSTEM_NL2SPARQL, nl2sparql_prompt

logger = structlog.get_logger(__name__)

# ── Namespace Prefixes ──────────────────────────────────────
SPARQL_PREFIXES = """PREFIX cve:   <http://w3id.org/sepses/vocab/ref/cve#>
PREFIX cwe:   <http://w3id.org/sepses/vocab/ref/cwe#>
PREFIX capec: <http://w3id.org/sepses/vocab/ref/capec#>
PREFIX cpe:   <http://w3id.org/sepses/vocab/ref/cpe#>
PREFIX cvss:  <http://w3id.org/sepses/vocab/ref/cvss#>"""

CVE_URI_BASE   = "http://w3id.org/sepses/resource/cve/"
CWE_URI_BASE   = "http://w3id.org/sepses/resource/cwe/"
CAPEC_URI_BASE = "http://w3id.org/sepses/resource/capec/"


# ============================================================
# Regex Pattern Matcher (fast path — no LLM needed)
# ============================================================
class RegexSparqlMatcher:
    """
    Cepat generate SPARQL untuk pola pertanyaan yang umum
    tanpa memanggil LLM (hemat biaya API).
    """

    # Pattern: pertanyaan tentang CVE spesifik
    CVE_ID_PATTERN = re.compile(
        r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE
    )
    # Pattern: pertanyaan tentang produk
    PRODUCT_PATTERN = re.compile(
        r"(?:vulnerabilities?|CVEs?)\s+(?:affecting|affect|in|for)\s+([a-z0-9_\-\.]+)",
        re.IGNORECASE,
    )
    # Pattern: CAPEC/attack pattern
    CAPEC_PATTERN = re.compile(
        r"\b(CAPEC-\d+)\b", re.IGNORECASE
    )
    # Pattern: top/critical CVEs
    TOP_CVES_PATTERN = re.compile(
        r"(?:top|critical|highest|most)\s+(?:CVEs?|vulnerabilities?)",
        re.IGNORECASE,
    )
    # Pattern: CWE specific
    CWE_PATTERN = re.compile(
        r"\b(CWE-\d+)\b", re.IGNORECASE
    )

    def match(self, question: str) -> Optional[str]:
        """
        Coba match question ke pola yang diketahui.

        Args:
            question: Natural language question.

        Returns:
            str: SPARQL query jika match, None jika tidak ada pola.
        """
        question_clean = question.strip()

        # ── CVE ID Lookup ─────────────────────────────────
        cve_match = self.CVE_ID_PATTERN.search(question_clean)
        if cve_match:
            cve_id = cve_match.group(1).upper()
            if any(kw in question_clean.lower() for kw in ["attack", "capec", "exploit", "pattern"]):
                return self._cve_to_capec_query(cve_id)
            if any(kw in question_clean.lower() for kw in ["cvss", "score", "severity", "rating"]):
                return self._cve_cvss_query(cve_id)
            if any(kw in question_clean.lower() for kw in ["product", "affect", "cpe", "software"]):
                return self._cve_products_query(cve_id)
            # Default: full details
            return self._cve_full_query(cve_id)

        # ── CWE Lookup ────────────────────────────────────
        cwe_match = self.CWE_PATTERN.search(question_clean)
        if cwe_match:
            cwe_id = cwe_match.group(1).upper()
            return self._cves_by_cwe_query(cwe_id)

        # ── Product Search ────────────────────────────────
        prod_match = self.PRODUCT_PATTERN.search(question_clean)
        if prod_match:
            product = prod_match.group(1).lower()
            return self._product_search_query(product)

        # ── Top CVEs by CVSS ──────────────────────────────
        if self.TOP_CVES_PATTERN.search(question_clean):
            min_score = 9.0
            if "high" in question_clean.lower():
                min_score = 7.0
            if "medium" in question_clean.lower():
                min_score = 4.0
            return self._top_cvss_query(min_score)

        return None  # No regex match — use LLM

    # ── Query Templates ────────────────────────────────────

    @staticmethod
    def _cve_full_query(cve_id: str) -> str:
        uri = f"<{CVE_URI_BASE}{cve_id}>"
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?description ?issued ?cweId ?cweName ?capecId ?capecName ?score ?attackVector ?cpeProduct
WHERE {{
  {uri} cve:cveId "{cve_id}" .
  OPTIONAL {{ {uri} cve:description ?description . }}
  OPTIONAL {{ {uri} cve:issued ?issued . }}
  OPTIONAL {{
    {uri} cve:hasCWE ?cwe .
    OPTIONAL {{ ?cwe cwe:cweId ?cweId ; cwe:name ?cweName . }}
    OPTIONAL {{
      ?cwe cwe:hasCAPEC ?capec .
      OPTIONAL {{ ?capec capec:capecId ?capecId ; capec:name ?capecName . }}
    }}
  }}
  OPTIONAL {{
    {uri} cve:hasCVSS ?cvssNode .
    OPTIONAL {{ ?cvssNode cvss:baseScore ?score ; cvss:attackVector ?attackVector . }}
  }}
  OPTIONAL {{ {uri} cve:hasCPE ?cpe . ?cpe cpe:productId ?cpeProduct . }}
}} LIMIT 30"""

    @staticmethod
    def _cve_to_capec_query(cve_id: str) -> str:
        uri = f"<{CVE_URI_BASE}{cve_id}>"
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?cweId ?cweName ?capecId ?capecName ?capecDescription
WHERE {{
  {uri} cve:hasCWE ?cwe .
  OPTIONAL {{ ?cwe cwe:cweId ?cweId ; cwe:name ?cweName . }}
  OPTIONAL {{
    ?cwe cwe:hasCAPEC ?capec .
    OPTIONAL {{ ?capec capec:capecId ?capecId ; capec:name ?capecName . }}
    OPTIONAL {{ ?capec capec:description ?capecDescription . }}
  }}
}} ORDER BY ?cweId ?capecId"""

    @staticmethod
    def _cve_cvss_query(cve_id: str) -> str:
        uri = f"<{CVE_URI_BASE}{cve_id}>"
        return f"""{SPARQL_PREFIXES}
SELECT ?score ?attackVector ?attackComplexity ?privilegesRequired ?confidentialityImpact ?integrityImpact ?availabilityImpact
WHERE {{
  {uri} cve:hasCVSS ?cvssNode .
  OPTIONAL {{ ?cvssNode cvss:baseScore    ?score . }}
  OPTIONAL {{ ?cvssNode cvss:attackVector ?attackVector . }}
  OPTIONAL {{ ?cvssNode cvss:attackComplexity ?attackComplexity . }}
  OPTIONAL {{ ?cvssNode cvss:privilegesRequired ?privilegesRequired . }}
  OPTIONAL {{ ?cvssNode cvss:confidentialityImpact ?confidentialityImpact . }}
  OPTIONAL {{ ?cvssNode cvss:integrityImpact ?integrityImpact . }}
  OPTIONAL {{ ?cvssNode cvss:availabilityImpact ?availabilityImpact . }}
}}"""

    @staticmethod
    def _cve_products_query(cve_id: str) -> str:
        uri = f"<{CVE_URI_BASE}{cve_id}>"
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?cpeProduct
WHERE {{
  {uri} cve:hasCPE ?cpe .
  ?cpe cpe:productId ?cpeProduct .
}} LIMIT 20"""

    @staticmethod
    def _cves_by_cwe_query(cwe_id: str) -> str:
        uri = f"<{CWE_URI_BASE}{cwe_id}>"
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?cveId ?score
WHERE {{
  ?cve cve:hasCWE {uri} ;
       cve:cveId  ?cveId .
  OPTIONAL {{
    ?cve cve:hasCVSS ?cvssNode .
    ?cvssNode cvss:baseScore ?score .
  }}
}} ORDER BY DESC(?score) LIMIT 20"""

    @staticmethod
    def _product_search_query(product: str) -> str:
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?cveId ?score ?cpeProduct
WHERE {{
  ?cve cve:cveId  ?cveId ;
       cve:hasCPE ?cpe .
  ?cpe cpe:productId ?cpeProduct .
  OPTIONAL {{ ?cve cve:hasCVSS ?cvssNode . ?cvssNode cvss:baseScore ?score . }}
  FILTER(CONTAINS(LCASE(STR(?cpeProduct)), "{product}"))
}} ORDER BY DESC(?score) LIMIT 20"""

    @staticmethod
    def _top_cvss_query(min_score: float = 9.0) -> str:
        return f"""{SPARQL_PREFIXES}
SELECT DISTINCT ?cveId ?score ?description
WHERE {{
  ?cve cve:cveId  ?cveId ;
       cve:hasCVSS ?cvssNode .
  ?cvssNode cvss:baseScore ?score .
  OPTIONAL {{ ?cve cve:description ?description . }}
  FILTER(?score >= {min_score})
}} ORDER BY DESC(?score) LIMIT 10"""


# ============================================================
# LLM-based NL2SPARQL Converter
# ============================================================
class NL2SPARQL:
    """
    Natural Language → SPARQL converter.

    Strategi dua tahap:
    1. Regex matcher untuk pola umum (tidak perlu API call)
    2. LLM few-shot generation untuk kasus kompleks
    """

    def __init__(self, llm: BaseLLMConnector) -> None:
        """
        Args:
            llm: LLM connector untuk fallback generation.
        """
        self._llm     = llm
        self._matcher = RegexSparqlMatcher()

    def convert(self, question: str) -> Tuple[str, str]:
        """
        Convert natural language question ke SPARQL query.

        Args:
            question: Natural language question dari user.

        Returns:
            Tuple[str, str]: (sparql_query, method_used)
                - method_used: "regex" | "llm" | "fallback"

        Raises:
            RuntimeError: Jika LLM gagal dan tidak ada fallback.
        """
        # ── Step 1: Regex fast path ────────────────────────
        regex_result = self._matcher.match(question)
        if regex_result:
            logger.info("nl2sparql_regex_match", question=question[:80])
            return regex_result, "regex"

        # ── Step 2: LLM generation ─────────────────────────
        try:
            messages = [
                Message("system", SYSTEM_NL2SPARQL),
                Message("user", nl2sparql_prompt(question)),
            ]
            raw_output = self._llm.generate(messages, temperature=0.1, max_tokens=512)
            sparql = self._extract_sparql(raw_output)

            if sparql and self._validate_sparql(sparql):
                logger.info("nl2sparql_llm_success", question=question[:80])
                return sparql, "llm"
            else:
                raise ValueError("LLM generated invalid SPARQL.")

        except Exception as exc:
            logger.warning("nl2sparql_llm_failed", error=str(exc))
            # ── Step 3: Fallback — cari CVE ID apapun ─────
            cve_match = RegexSparqlMatcher.CVE_ID_PATTERN.search(question)
            if cve_match:
                cve_id = cve_match.group(1).upper()
                fallback = RegexSparqlMatcher._cve_full_query(cve_id)
                return fallback, "fallback"

            # Generic fallback: top critical CVEs
            return RegexSparqlMatcher._top_cvss_query(9.0), "fallback"

    @staticmethod
    def _extract_sparql(llm_output: str) -> str:
        """
        Ekstrak SPARQL query dari output LLM (biasanya ada dalam code block).

        Args:
            llm_output: Raw output string dari LLM.

        Returns:
            str: SPARQL query string yang diekstrak.
        """
        # Coba extract dari ```sparql ... ``` atau ``` ... ```
        patterns = [
            r"```sparql\s*([\s\S]+?)```",
            r"```\s*(PREFIX[\s\S]+?)```",
            r"```\s*(SELECT[\s\S]+?)```",
        ]
        for pat in patterns:
            m = re.search(pat, llm_output, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Jika tidak ada code block, ambil dari PREFIX/SELECT
        lines = llm_output.strip().splitlines()
        sparql_lines = []
        started = False
        for line in lines:
            if re.match(r"^\s*(PREFIX|SELECT)", line, re.IGNORECASE):
                started = True
            if started:
                sparql_lines.append(line)

        return "\n".join(sparql_lines).strip()

    @staticmethod
    def _validate_sparql(sparql: str) -> bool:
        """
        Validasi dasar apakah string merupakan SPARQL SELECT yang valid.

        Args:
            sparql: Query string.

        Returns:
            bool: True jika tampaknya valid.
        """
        sparql_upper = sparql.upper()
        has_select = "SELECT" in sparql_upper
        has_where  = "WHERE" in sparql_upper
        has_braces = "{" in sparql and "}" in sparql
        return has_select and has_where and has_braces
