"""
SEPSES CSKG LLM Chatbot - Graph Builder
=========================================
Tanggung Jawab  : Ajie Armansyah Sunaryo (Knowledge Architect)
Branch          : feature/kg-engine

Deskripsi:
    Membangun NetworkX graph dari hasil SPARQL query untuk divisualisasikan
    di frontend (pyvis). Output berupa dict {"nodes": [...], "edges": [...]}
    yang kompatibel dengan format yang diharapkan graph_visualizer.py.
"""

import os
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv

from kg_engine.sparql_client import SparqlClient

load_dotenv()
logger = structlog.get_logger(__name__)

# ── Node type untuk pewarnaan di frontend ──────────────────
NODE_TYPE_MAP = {
    "CVE":    "CVE",
    "CWE":    "CWE",
    "CAPEC":  "CAPEC",
    "CPE":    "CPE",
    "CVSS":   "CVSS",
    "ATT&CK": "ATT&CK",
}


class GraphBuilder:
    """
    Build NetworkX-style graph dict dari SEPSES SPARQL query results.

    Output format (kompatibel dengan frontend/components/graph_visualizer.py):
    {
        "nodes": [
            {"id": "CVE-2021-44228", "type": "CVE", "label": "...", "tooltip": "..."},
            ...
        ],
        "edges": [
            {"from": "CVE-2021-44228", "to": "CWE-917", "label": "hasCWE", "color": "#ff8c00"},
            ...
        ]
    }
    """

    # Warna edge per relasi — sinkron dengan frontend/components/graph_visualizer.py
    EDGE_COLORS = {
        "hasCWE":   "#ff8c00",
        "hasCAPEC": "#ffd60a",
        "hasCPE":   "#00b4ff",
        "hasCVSS":  "#00ff88",
        "uses":     "#b967ff",
    }

    def __init__(self, sparql_client: Optional[SparqlClient] = None) -> None:
        """
        Inisialisasi GraphBuilder.

        Args:
            sparql_client: Instance SparqlClient. Jika None, dibuat baru.
        """
        self._client = sparql_client or SparqlClient()

    def build_cve_graph(self, cve_id: str) -> Dict[str, List[Dict]]:
        """
        Build graph lengkap untuk satu CVE:
        CVE → CWE → CAPEC, CVE → CPE, CVE → CVSS

        Args:
            cve_id: CVE identifier, e.g. "CVE-2021-44228".

        Returns:
            Dict {"nodes": [...], "edges": [...]} siap untuk pyvis.

        Raises:
            ValueError   : Jika CVE ID tidak valid.
            RuntimeError : Jika SPARQL endpoint tidak tersedia.
        """
        cve_id = cve_id.strip().upper()
        logger.info("building_cve_graph", cve_id=cve_id)

        # Ambil data dari KG
        details = self._client.get_cve_details(cve_id)
        chain   = self._client.get_capec_from_cve(cve_id)

        nodes: List[Dict] = []
        edges: List[Dict] = []
        seen_nodes = set()

        def add_node(node_id: str, node_type: str, label: str, tooltip: str = "") -> None:
            if node_id not in seen_nodes:
                nodes.append({
                    "id":      node_id,
                    "type":    node_type,
                    "label":   label,
                    "tooltip": tooltip or node_id,
                })
                seen_nodes.add(node_id)

        def add_edge(from_id: str, to_id: str, relation: str) -> None:
            edges.append({
                "from":  from_id,
                "to":    to_id,
                "label": relation,
                "color": self.EDGE_COLORS.get(relation, "#3a4a6a"),
            })

        # ── CVE Node ───────────────────────────────────────
        score_label = f"CVSS {details['cvss_score']:.1f}" if details.get("cvss_score") else "CVSS N/A"
        cve_tooltip = (
            f"{score_label} | "
            f"{details.get('description', '')[:120]}..."
        )
        add_node(
            cve_id, "CVE",
            label=f"{cve_id}\n({score_label})",
            tooltip=cve_tooltip,
        )

        # ── CVSS Node ──────────────────────────────────────
        if details.get("cvss_score"):
            score = details["cvss_score"]
            severity = (
                "CRITICAL" if score >= 9.0 else
                "HIGH"     if score >= 7.0 else
                "MEDIUM"   if score >= 4.0 else "LOW"
            )
            cvss_id = f"CVSS-{score}"
            add_node(
                cvss_id, "CVSS",
                label=f"CVSS\n{score} {severity}",
                tooltip=f"Base Score: {score} | Attack Vector: {details.get('attack_vector', 'N/A')}",
            )
            add_edge(cve_id, cvss_id, "hasCVSS")

        # ── CWE + CAPEC Nodes (dari chain) ─────────────────
        seen_cwes:   Dict[str, str] = {}  # cwe_id → node_id used
        seen_capecs: Dict[str, str] = {}

        for item in chain:
            cwe_id_raw   = item.get("cwe_id", "")
            cwe_name     = item.get("cwe_name", "")
            capec_id_raw = item.get("capec_id", "")
            capec_name   = item.get("capec_name", "")
            capec_desc   = item.get("capec_desc", "")

            # CWE Node
            if cwe_id_raw and cwe_id_raw not in seen_cwes:
                node_id = cwe_id_raw
                add_node(
                    node_id, "CWE",
                    label=f"{cwe_id_raw}\n{cwe_name[:20]}" if cwe_name else cwe_id_raw,
                    tooltip=f"{cwe_id_raw}: {cwe_name}",
                )
                add_edge(cve_id, node_id, "hasCWE")
                seen_cwes[cwe_id_raw] = node_id

            # CAPEC Node
            if capec_id_raw and capec_id_raw not in seen_capecs:
                node_id = capec_id_raw
                add_node(
                    node_id, "CAPEC",
                    label=f"{capec_id_raw}\n{capec_name[:20]}" if capec_name else capec_id_raw,
                    tooltip=f"{capec_id_raw}: {capec_name}\n{capec_desc[:100]}",
                )
                # Edge dari CWE ke CAPEC
                if cwe_id_raw in seen_cwes:
                    add_edge(seen_cwes[cwe_id_raw], node_id, "hasCAPEC")
                seen_capecs[capec_id_raw] = node_id

        # ── CPE Nodes (max 3 untuk keterbacaan) ────────────
        for i, product in enumerate(details.get("cpe_products", [])[:3]):
            # Ambil nama pendek dari CPE URI
            short_name = product.split(":")[-1] if ":" in product else product
            cpe_node_id = f"CPE-{short_name[:20]}-{i}"
            add_node(
                cpe_node_id, "CPE",
                label=short_name[:20],
                tooltip=f"Affected: {product}",
            )
            add_edge(cve_id, cpe_node_id, "hasCPE")

        logger.info(
            "cve_graph_built",
            cve_id=cve_id,
            nodes=len(nodes),
            edges=len(edges),
        )
        return {"nodes": nodes, "edges": edges}

    def build_product_graph(self, product_name: str) -> Dict[str, List[Dict]]:
        """
        Build graph CVE-centric berdasarkan produk yang terdampak.

        Args:
            product_name: Nama produk (partial match), e.g. "apache".

        Returns:
            Dict {"nodes": [...], "edges": [...]}
        """
        cves = self._client.search_vulnerabilities_by_product(
            product_name, min_score=7.0, limit=8
        )

        nodes: List[Dict] = []
        edges: List[Dict] = []
        seen  = set()

        # Product center node
        prod_node_id = f"PRODUCT-{product_name}"
        nodes.append({
            "id":      prod_node_id,
            "type":    "CPE",
            "label":   product_name,
            "tooltip": f"Product: {product_name} — click to see related CVEs",
        })
        seen.add(prod_node_id)

        for item in cves:
            cve_id = item.get("cve_id", "")
            score  = item.get("score")
            if not cve_id or cve_id in seen:
                continue

            score_str = f"CVSS {score:.1f}" if score else "CVSS N/A"
            nodes.append({
                "id":      cve_id,
                "type":    "CVE",
                "label":   f"{cve_id}\n{score_str}",
                "tooltip": item.get("description", "")[:120],
            })
            edges.append({
                "from":  prod_node_id,
                "to":    cve_id,
                "label": "affects",
                "color": "#ff4d6d",
            })
            seen.add(cve_id)

        logger.info("product_graph_built", product=product_name, nodes=len(nodes))
        return {"nodes": nodes, "edges": edges}
