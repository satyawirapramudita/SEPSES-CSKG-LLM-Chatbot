"""
SEPSES CSKG LLM Chatbot - Multi-Hop Graph Traversal
=====================================================
Tanggung Jawab  : Fahmi Abdillah Zain (RAG Logic Dev)
Branch          : feature/rag-logic

Deskripsi:
    Implementasi multi-hop reasoning pada SEPSES KG:
    CVE → CWE → CAPEC → ATT&CK (simulasi via CAPEC name mapping)
"""

from typing import Any, Dict, List, Optional
import structlog
from kg_engine.sparql_client import SparqlClient

logger = structlog.get_logger(__name__)

# ── CAPEC → ATT&CK Mapping ─────────────────────────────────
CAPEC_TO_ATTCK: Dict[str, List[Dict[str, str]]] = {
    "CAPEC-88":  [{"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access"}],
    "CAPEC-1":   [{"id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "Execution"}],
    "CAPEC-7":   [{"id": "T1548", "name": "Abuse Elevation Control Mechanism", "tactic": "Privilege Escalation"}],
    "CAPEC-100": [{"id": "T1210", "name": "Exploitation of Remote Services", "tactic": "Lateral Movement"}],
    "CAPEC-115": [{"id": "T1189", "name": "Drive-by Compromise", "tactic": "Initial Access"}],
    "CAPEC-209": [{"id": "T1059.007", "name": "JavaScript", "tactic": "Execution"}],
    "CAPEC-49":  [{"id": "T1110", "name": "Brute Force", "tactic": "Credential Access"}],
    "CAPEC-60":  [{"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access"}],
    "CAPEC-198": [{"id": "T1566", "name": "Phishing", "tactic": "Initial Access"}],
    "CAPEC-17":  [{"id": "T1055", "name": "Process Injection", "tactic": "Defense Evasion"}],
    "CAPEC-94":  [{"id": "T1059.001", "name": "PowerShell", "tactic": "Execution"}],
}


class MultiHopTraversal:
    """Multi-hop graph traversal: CVE → CWE → CAPEC → ATT&CK."""

    def __init__(self, sparql_client: Optional[SparqlClient] = None) -> None:
        """
        Args:
            sparql_client: SPARQL client. Jika None, dibuat baru.
        """
        self._client = sparql_client or SparqlClient()

    def build_attack_chain(self, cve_id: str) -> Dict[str, Any]:
        """
        Build full attack chain: CVE → CWE → CAPEC → ATT&CK.

        Args:
            cve_id: CVE identifier, e.g. "CVE-2021-44228".

        Returns:
            Dict dengan keys: cve_id, description, cvss_score, cwes, capecs,
                              products, found, context_str.
        """
        cve_id = cve_id.strip().upper()
        logger.info("building_attack_chain", cve_id=cve_id)

        try:
            details = self._client.get_cve_details(cve_id)
            chain   = self._client.get_capec_from_cve(cve_id)
        except RuntimeError as exc:
            logger.warning("attack_chain_kg_unavailable", error=str(exc))
            return self._build_unavailable_response(cve_id, str(exc))

        # Proses CWE
        cwes: List[Dict[str, str]] = []
        seen_cwe: set = set()
        for item in chain:
            cid = item.get("cwe_id", "")
            if cid and cid not in seen_cwe:
                cwes.append({"id": cid, "name": item.get("cwe_name", "")})
                seen_cwe.add(cid)

        # Proses CAPEC → ATT&CK
        capecs: List[Dict[str, Any]] = []
        seen_capec: set = set()
        for item in chain:
            cid = item.get("capec_id", "")
            if cid and cid not in seen_capec:
                capecs.append({
                    "id":    cid,
                    "name":  item.get("capec_name", ""),
                    "desc":  item.get("capec_desc", "")[:200],
                    "attck": CAPEC_TO_ATTCK.get(cid, []),
                })
                seen_capec.add(cid)

        result: Dict[str, Any] = {
            "cve_id":       cve_id,
            "description":  details.get("description", ""),
            "cvss_score":   details.get("cvss_score"),
            "attack_vector": details.get("attack_vector"),
            "cwes":         cwes,
            "capecs":       capecs,
            "products":     details.get("cpe_products", []),
            "found":        details.get("found", False),
        }
        result["context_str"] = self._format_chain(result)

        logger.info("attack_chain_built", cve_id=cve_id, cwes=len(cwes), capecs=len(capecs))
        return result

    def build_threat_context(self, query: str, top_k: int = 5) -> str:
        """
        Build context untuk query yang tidak merujuk CVE spesifik.

        Args:
            query: User query.
            top_k: Jumlah CVE yang diambil.

        Returns:
            str: Formatted context string.
        """
        import re
        tech_keywords = re.findall(
            r"\b(apache|nginx|log4j|windows|linux|openssh|openssl|wordpress|spring|struts)\b",
            query.lower(),
        )
        if tech_keywords:
            product = tech_keywords[0]
            try:
                cves = self._client.search_vulnerabilities_by_product(product, min_score=7.0, limit=top_k)
                if cves:
                    lines = [f"Top CVEs affecting '{product}' (CVSS >= 7.0):"]
                    for c in cves:
                        score_s = f"CVSS {c['score']:.1f}" if c.get("score") else "N/A"
                        lines.append(f"  - {c['cve_id']} ({score_s}): {c.get('description','')[:100]}")
                    return "\n".join(lines)
            except Exception as exc:
                logger.warning("product_search_failed", error=str(exc))

        try:
            top_cves = self._client.get_top_cvss_cves(min_score=9.0, limit=top_k)
            if top_cves:
                lines = ["Top Critical CVEs (CVSS >= 9.0):"]
                for c in top_cves:
                    lines.append(f"  - {c['cve_id']} (CVSS {c['score']:.1f})")
                return "\n".join(lines)
        except Exception as exc:
            logger.warning("top_cves_failed", error=str(exc))

        return "SEPSES Knowledge Graph is currently unavailable."

    @staticmethod
    def _format_chain(chain: Dict[str, Any]) -> str:
        """Format attack chain dict → readable string untuk LLM."""
        lines = []
        score = chain.get("cvss_score")
        score_str = f"CVSS {score:.1f}" if score else "CVSS N/A"
        av = chain.get("attack_vector", "")
        lines.append(f"=== {chain['cve_id']} | {score_str} | Attack Vector: {av} ===")
        if chain.get("description"):
            lines.append(f"Description: {chain['description'][:300]}")
        if chain.get("products"):
            lines.append(f"Affected Products: {', '.join(chain['products'][:5])}")
        if chain.get("cwes"):
            lines.append("\n[Weakness (CWE)]")
            for cwe in chain["cwes"]:
                lines.append(f"  → {cwe['id']}: {cwe['name']}")
        if chain.get("capecs"):
            lines.append("\n[Attack Patterns (CAPEC) → ATT&CK]")
            for capec in chain["capecs"]:
                lines.append(f"  → {capec['id']}: {capec['name']}")
                for tech in capec.get("attck", []):
                    lines.append(f"     ATT&CK {tech['id']} ({tech['tactic']}): {tech['name']}")
        return "\n".join(lines)

    @staticmethod
    def _build_unavailable_response(cve_id: str, error: str) -> Dict[str, Any]:
        """Fallback response ketika KG tidak tersedia."""
        return {
            "cve_id": cve_id, "description": "", "cvss_score": None,
            "cwes": [], "capecs": [], "products": [], "found": False,
            "context_str": (
                f"Note: SEPSES KG unavailable ({error}). "
                f"Answering based on general knowledge about {cve_id}."
            ),
        }
