"""
SEPSES CSKG LLM Chatbot - KG Graph Visualizer Component
=========================================================
Tanggung Jawab  : Muhammad Dhafin Alfeizar Gandhan (Full-Stack UI Dev)
Branch          : feature/frontend-ui

Deskripsi:
    Interactive Knowledge Graph explorer menggunakan pyvis.
    Memvisualisasikan relasi CVE→CWE→CAPEC→ATT&CK dalam bentuk
    network graph interaktif yang dapat di-embed di Streamlit.

    Fitur:
    - Node coloring per tipe entitas (CVE, CWE, CAPEC, CPE, ATT&CK)
    - Click-to-expand (via hover tooltip)
    - SPARQL query input untuk custom graph
    - Mock graph untuk demo
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Node Style Configuration
# ============================================================
NODE_STYLES: Dict[str, Dict] = {
    "CVE": {
        "color": "#ff4d6d",
        "shape": "dot",
        "size": 20,
        "font_color": "#ffffff",
        "border": "#ff1744",
    },
    "CWE": {
        "color": "#ff8c00",
        "shape": "diamond",
        "size": 18,
        "font_color": "#ffffff",
        "border": "#e65100",
    },
    "CAPEC": {
        "color": "#ffd60a",
        "shape": "triangle",
        "size": 18,
        "font_color": "#0a0e1a",
        "border": "#f9a825",
    },
    "CPE": {
        "color": "#00b4ff",
        "shape": "square",
        "size": 16,
        "font_color": "#ffffff",
        "border": "#0077cc",
    },
    "ATT&CK": {
        "color": "#b967ff",
        "shape": "ellipse",
        "size": 18,
        "font_color": "#ffffff",
        "border": "#7b1fa2",
    },
    "CVSS": {
        "color": "#00ff88",
        "shape": "dot",
        "size": 14,
        "font_color": "#0a0e1a",
        "border": "#00c853",
    },
    "DEFAULT": {
        "color": "#4a6080",
        "shape": "dot",
        "size": 12,
        "font_color": "#e8eaf0",
        "border": "#2a4060",
    },
}

EDGE_STYLES: Dict[str, str] = {
    "hasCWE":   "#ff8c00",
    "hasCAPEC": "#ffd60a",
    "hasCPE":   "#00b4ff",
    "hasCVSS":  "#00ff88",
    "uses":     "#b967ff",
    "DEFAULT":  "#3a4a6a",
}


# ============================================================
# Mock Graph Data
# ============================================================
MOCK_GRAPH_LOG4SHELL = {
    "nodes": [
        {"id": "CVE-2021-44228", "type": "CVE", "label": "CVE-2021-44228\n(Log4Shell)",
         "tooltip": "CVSS: 10.0 | Apache Log4j RCE | Critical"},
        {"id": "CWE-917", "type": "CWE", "label": "CWE-917\nExpr. Language Inj.",
         "tooltip": "Improper Neutralization of Special Elements used in an Expression Language Statement"},
        {"id": "CWE-502", "type": "CWE", "label": "CWE-502\nDeserialization",
         "tooltip": "Deserialization of Untrusted Data"},
        {"id": "CAPEC-88", "type": "CAPEC", "label": "CAPEC-88\nOS Cmd Injection",
         "tooltip": "OS Command Injection via HTTP Query Strings"},
        {"id": "CAPEC-209", "type": "CAPEC", "label": "CAPEC-209\nXSS MIME Type",
         "tooltip": "XSS Using MIME Type Mismatch"},
        {"id": "CPE-log4j-2.0", "type": "CPE", "label": "apache:log4j:2.0-2.14",
         "tooltip": "Affected product: Apache Log4j 2.0 - 2.14.1"},
        {"id": "CVSS-10.0", "type": "CVSS", "label": "CVSS\n10.0 CRITICAL",
         "tooltip": "Base Score: 10.0 | Vector: AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"},
        {"id": "ATTACK-T1190", "type": "ATT&CK", "label": "T1190\nExploit Public App",
         "tooltip": "MITRE ATT&CK T1190: Exploit Public-Facing Application"},
        {"id": "ATTACK-T1059", "type": "ATT&CK", "label": "T1059\nCmd & Scripting",
         "tooltip": "MITRE ATT&CK T1059: Command and Scripting Interpreter"},
    ],
    "edges": [
        {"from": "CVE-2021-44228", "to": "CWE-917",      "label": "hasCWE",   "color": "#ff8c00"},
        {"from": "CVE-2021-44228", "to": "CWE-502",      "label": "hasCWE",   "color": "#ff8c00"},
        {"from": "CVE-2021-44228", "to": "CPE-log4j-2.0","label": "hasCPE",   "color": "#00b4ff"},
        {"from": "CVE-2021-44228", "to": "CVSS-10.0",    "label": "hasCVSS",  "color": "#00ff88"},
        {"from": "CWE-917",        "to": "CAPEC-88",     "label": "hasCAPEC", "color": "#ffd60a"},
        {"from": "CWE-502",        "to": "CAPEC-209",    "label": "hasCAPEC", "color": "#ffd60a"},
        {"from": "CAPEC-88",       "to": "ATTACK-T1190", "label": "uses",     "color": "#b967ff"},
        {"from": "CAPEC-88",       "to": "ATTACK-T1059", "label": "uses",     "color": "#b967ff"},
    ],
}

MOCK_GRAPH_ETERNALBLUE = {
    "nodes": [
        {"id": "CVE-2017-0144", "type": "CVE", "label": "CVE-2017-0144\n(EternalBlue)",
         "tooltip": "CVSS: 8.1 | Windows SMBv1 RCE | WannaCry/NotPetya"},
        {"id": "CWE-119", "type": "CWE", "label": "CWE-119\nBuffer Errors",
         "tooltip": "Improper Restriction of Operations within Bounds of Memory Buffer"},
        {"id": "CAPEC-100", "type": "CAPEC", "label": "CAPEC-100\nOverflow Buffers",
         "tooltip": "Overflow Buffers attack pattern"},
        {"id": "CPE-win-smb", "type": "CPE", "label": "microsoft:windows\nSMBv1",
         "tooltip": "Affected: Microsoft Windows SMBv1 implementations"},
        {"id": "CVSS-8.1", "type": "CVSS", "label": "CVSS\n8.1 HIGH",
         "tooltip": "Base Score: 8.1 | AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H"},
        {"id": "ATTACK-T1210", "type": "ATT&CK", "label": "T1210\nExploit Remote Svcs",
         "tooltip": "MITRE ATT&CK T1210: Exploitation of Remote Services"},
    ],
    "edges": [
        {"from": "CVE-2017-0144", "to": "CWE-119",     "label": "hasCWE",   "color": "#ff8c00"},
        {"from": "CVE-2017-0144", "to": "CPE-win-smb", "label": "hasCPE",   "color": "#00b4ff"},
        {"from": "CVE-2017-0144", "to": "CVSS-8.1",    "label": "hasCVSS",  "color": "#00ff88"},
        {"from": "CWE-119",       "to": "CAPEC-100",   "label": "hasCAPEC", "color": "#ffd60a"},
        {"from": "CAPEC-100",     "to": "ATTACK-T1210","label": "uses",     "color": "#b967ff"},
    ],
}


# ============================================================
# Graph Builder
# ============================================================
def _build_pyvis_graph(graph_data: Dict, height: int = 550) -> str:
    """
    Build pyvis Network graph dari dict data dan return HTML string.

    Args:
        graph_data : Dict dengan keys "nodes" dan "edges".
        height     : Tinggi canvas dalam pixel.

    Returns:
        str: HTML string yang berisi pyvis network.

    Raises:
        ImportError: Jika pyvis tidak terinstall.
    """
    try:
        from pyvis.network import Network  # lazy import
    except ImportError as exc:
        raise ImportError(
            "pyvis tidak terinstall. Jalankan: pip install pyvis"
        ) from exc

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#0f1629",
        font_color="#e8eaf0",
        directed=True,
        notebook=False,
    )

    # Konfigurasi physics
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.3,
          "springLength": 140,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": {"iterations": 250}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "font": {"size": 11, "color": "#8899b8", "strokeWidth": 2, "strokeColor": "#0f1629"}
      },
      "nodes": {
        "font": {"size": 12, "face": "Inter, sans-serif"},
        "borderWidth": 2,
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)", "size": 8}
      }
    }
    """)

    # Add nodes
    for node in graph_data.get("nodes", []):
        style = NODE_STYLES.get(node["type"], NODE_STYLES["DEFAULT"])
        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("tooltip", node["id"]),
            color={
                "background": style["color"],
                "border": style["border"],
                "highlight": {"background": style["color"], "border": "#ffffff"},
                "hover": {"background": style["color"], "border": "#ffffff"},
            },
            shape=style["shape"],
            size=style["size"],
            font={"color": style["font_color"], "size": 12},
        )

    # Add edges
    for edge in graph_data.get("edges", []):
        edge_color = edge.get("color", EDGE_STYLES["DEFAULT"])
        net.add_edge(
            edge["from"],
            edge["to"],
            label=edge.get("label", ""),
            color={"color": edge_color, "highlight": "#ffffff", "hover": "#ffffff"},
            width=2,
        )

    # Simpan ke temp file, tutup, lalu baca kembali
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        tmp_path = f.name

    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as rf:
            html_content = rf.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return html_content


def _get_kg_engine():
    """
    Load KG engine nyata, atau fallback ke mock.

    Returns:
        Callable | None: Function(cve_id) -> graph_data dict, atau None jika tidak tersedia.
    """
    try:
        from kg_engine.graph_builder import GraphBuilder  # type: ignore
        builder = GraphBuilder()
        return builder.build_cve_graph
    except ImportError:
        return None


# ============================================================
# Main KG Explorer Page
# ============================================================
def render_kg_explorer_page() -> None:
    """Render halaman KG Explorer dengan graph visualisasi interaktif."""

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size:1.8rem; font-weight:700; color:#e8eaf0; margin:0;">
            🔍 Knowledge Graph Explorer
        </h1>
        <p style="color:#6b7a99; font-size:0.88rem; margin:0.3rem 0 0;">
            Visualize CVE→CWE→CAPEC→ATT&CK relationship chains from SEPSES CSKG
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_preset, tab_custom, tab_legend = st.tabs([
        "⚡ Quick Presets", "🔎 Custom Query", "🎨 Legend"
    ])

    with tab_preset:
        _render_preset_graphs()

    with tab_custom:
        _render_custom_sparql_graph()

    with tab_legend:
        _render_legend()


def _render_preset_graphs() -> None:
    """Render preset graph examples untuk demo cepat."""

    st.markdown(
        "<div style='color:#8899b8; font-size:0.88rem; margin-bottom:1rem;'>"
        "Select a preset CVE to visualize its knowledge graph chain:"
        "</div>",
        unsafe_allow_html=True
    )

    presets = {
        "🔴 Log4Shell (CVE-2021-44228)": MOCK_GRAPH_LOG4SHELL,
        "🔵 EternalBlue (CVE-2017-0144)": MOCK_GRAPH_ETERNALBLUE,
    }

    selected_preset = st.radio(
        "Preset",
        options=list(presets.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    graph_data = presets[selected_preset]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    node_types = [n["type"] for n in graph_data["nodes"]]
    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Total Nodes</div>"
            f"<div class='metric-value'>{len(graph_data['nodes'])}</div></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Total Edges</div>"
            f"<div class='metric-value'>{len(graph_data['edges'])}</div></div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>CVE Nodes</div>"
            f"<div class='metric-value' style='color:#ff4d6d'>{node_types.count('CVE')}</div></div>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>ATT&CK Nodes</div>"
            f"<div class='metric-value' style='color:#b967ff'>{node_types.count('ATT&CK')}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Render graph
    try:
        graph_html = _build_pyvis_graph(graph_data, height=520)
        components.html(graph_html, height=540, scrolling=False)
    except ImportError:
        st.warning(
            "⚠️ **pyvis tidak terinstall** di environment Streamlit aktif. "
            "Jalankan: `pip install pyvis` di terminal Anaconda, lalu restart Streamlit.",
            icon="📦"
        )

    # Node details table
    st.markdown(
        "<div style='font-size:0.82rem; color:#6b7a99; margin-top:0.5rem;'>"
        "💡 Hover over nodes for details • Drag to rearrange • Scroll to zoom"
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("📋 Node Details Table"):
        import pandas as pd
        node_rows = [
            {
                "ID": n["id"],
                "Type": n["type"],
                "Description": n.get("tooltip", ""),
            }
            for n in graph_data["nodes"]
        ]
        st.dataframe(
            pd.DataFrame(node_rows),
            use_container_width=True,
            hide_index=True,
        )


def _render_custom_sparql_graph() -> None:
    """Render custom SPARQL input untuk query KG secara bebas."""

    st.markdown(
        "<div style='color:#8899b8; font-size:0.88rem; margin-bottom:1rem;'>"
        "Enter a CVE ID to retrieve its knowledge graph from SEPSES:"
        "</div>",
        unsafe_allow_html=True
    )

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        cve_input = st.text_input(
            "CVE ID",
            placeholder="e.g., CVE-2021-44228",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Search", use_container_width=True)

    if search_clicked and cve_input:
        cve_id = cve_input.strip().upper()
        if not cve_id.startswith("CVE-"):
            st.error("Format CVE ID tidak valid. Gunakan format: CVE-YYYY-NNNNN")
            return

        with st.spinner(f"🔍 Fetching graph for {cve_id}..."):
            kg_fn = _get_kg_engine()

            if kg_fn:
                try:
                    graph_data = kg_fn(cve_id)
                    st.session_state.kg_graph_html = _build_pyvis_graph(graph_data)
                    st.session_state.kg_results = graph_data
                except Exception as exc:
                    st.error(f"Error fetching KG data: {exc}")
                    return
            else:
                # Mock: tampilkan pesan dan gunakan preset Log4Shell
                st.info(
                    f"⚠️ **KG Engine belum terhubung** (feature/kg-engine oleh Ajie). "
                    f"Menampilkan demo graph untuk {cve_id}.",
                    icon="🔌"
                )
                graph_data = MOCK_GRAPH_LOG4SHELL
                st.session_state.kg_graph_html = _build_pyvis_graph(graph_data)
                st.session_state.kg_results = graph_data

    if st.session_state.get("kg_graph_html"):
        components.html(st.session_state.kg_graph_html, height=540, scrolling=False)


def _render_legend() -> None:
    """Render legenda node dan edge types."""

    st.markdown(
        "<div style='font-size:0.88rem; color:#8899b8; margin-bottom:1rem;'>"
        "Node and edge color coding used in the knowledge graph visualization:"
        "</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    legend_items = [
        ("CVE", "#ff4d6d", "Common Vulnerabilities & Exposures\n(specific vulnerability instances)"),
        ("CWE", "#ff8c00", "Common Weakness Enumeration\n(vulnerability root cause types)"),
        ("CAPEC", "#ffd60a", "Common Attack Pattern Enumeration\n(how vulnerabilities are exploited)"),
        ("CPE", "#00b4ff", "Common Platform Enumeration\n(affected products/platforms)"),
        ("ATT&CK", "#b967ff", "MITRE ATT&CK Techniques\n(adversary tactics & techniques)"),
        ("CVSS", "#00ff88", "Common Vulnerability Scoring System\n(severity metrics)"),
    ]

    for i, (node_type, color, desc) in enumerate(legend_items):
        with cols[i % 3]:
            st.markdown(
                f"<div class='metric-card' style='margin-bottom:0.8rem;'>"
                f"<div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;'>"
                f"<div style='width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0;'></div>"
                f"<span style='font-weight:600; color:{color};'>{node_type}</span>"
                f"</div>"
                f"<div style='font-size:0.78rem; color:#6b7a99;white-space:pre-line;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
