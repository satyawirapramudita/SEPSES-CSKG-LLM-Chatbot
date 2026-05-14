"""
SEPSES CSKG LLM Chatbot - Evaluation Dashboard Component
=========================================================
Tanggung Jawab  : Muhammad Dhafin Alfeizar Gandhan (Full-Stack UI Dev)
Branch          : feature/frontend-ui

Deskripsi:
    Dashboard visualisasi hasil perbandingan LLM evaluation.
    Menampilkan hasil dari evaluation/grader.py (Satya's module)
    dalam bentuk:
    - Bar chart: rata-rata metrik per LLM
    - Radar chart: multi-dimensional performance comparison
    - Per-kategori breakdown table
    - Detailed results tabel per pertanyaan
    - Mock results jika evaluasi belum dijalankan
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Mock Evaluation Results (sebelum evaluasi nyata dijalankan)
# ============================================================
MOCK_EVAL_RESULTS = {
    "metadata": {
        "timestamp": "20240624_091233",
        "judge_model": "gpt-4o-mini",
        "benchmark_size": 30,
        "llms_evaluated": ["gpt-4o-mini", "mistral"],
    },
    "summaries": [
        {
            "llm_name": "gpt-4o-mini",
            "total_questions": 30,
            "avg_faithfulness": 0.84,
            "avg_answer_relevancy": 0.87,
            "avg_context_precision": 0.79,
            "avg_overall_score": 0.834,
            "avg_latency_ms": 1240.5,
            "error_count": 0,
            "results_by_category": {
                "security_analysis": {
                    "avg_faithfulness": 0.86, "avg_answer_relevancy": 0.89,
                    "avg_context_precision": 0.81, "avg_overall": 0.853, "count": 10
                },
                "log_analysis": {
                    "avg_faithfulness": 0.81, "avg_answer_relevancy": 0.85,
                    "avg_context_precision": 0.76, "avg_overall": 0.812, "count": 10
                },
                "kg_qa": {
                    "avg_faithfulness": 0.85, "avg_answer_relevancy": 0.87,
                    "avg_context_precision": 0.80, "avg_overall": 0.840, "count": 10
                },
            },
        },
        {
            "llm_name": "mistral",
            "total_questions": 30,
            "avg_faithfulness": 0.76,
            "avg_answer_relevancy": 0.79,
            "avg_context_precision": 0.71,
            "avg_overall_score": 0.754,
            "avg_latency_ms": 2830.2,
            "error_count": 1,
            "results_by_category": {
                "security_analysis": {
                    "avg_faithfulness": 0.78, "avg_answer_relevancy": 0.80,
                    "avg_context_precision": 0.73, "avg_overall": 0.770, "count": 10
                },
                "log_analysis": {
                    "avg_faithfulness": 0.73, "avg_answer_relevancy": 0.77,
                    "avg_context_precision": 0.68, "avg_overall": 0.728, "count": 10
                },
                "kg_qa": {
                    "avg_faithfulness": 0.77, "avg_answer_relevancy": 0.80,
                    "avg_context_precision": 0.72, "avg_overall": 0.763, "count": 10
                },
            },
        },
    ],
    "detailed_results": {
        "gpt-4o-mini": [
            {"question_id": "SA-001", "category": "security_analysis",
             "question": "What are the attack patterns for CVE-2021-44228?",
             "faithfulness": 0.92, "answer_relevancy": 0.95, "context_precision": 0.88,
             "latency_ms": 1150, "error": None},
            {"question_id": "SA-002", "category": "security_analysis",
             "question": "CVSS score of CVE-2017-0144?",
             "faithfulness": 0.90, "answer_relevancy": 0.93, "context_precision": 0.87,
             "latency_ms": 980, "error": None},
            {"question_id": "LOG-001", "category": "log_analysis",
             "question": "Identify suspicious Snort alert activities",
             "faithfulness": 0.82, "answer_relevancy": 0.86, "context_precision": 0.78,
             "latency_ms": 1320, "error": None},
            {"question_id": "KG-001", "category": "kg_qa",
             "question": "Which CVEs are linked to CWE-79?",
             "faithfulness": 0.88, "answer_relevancy": 0.90, "context_precision": 0.83,
             "latency_ms": 1100, "error": None},
        ],
        "mistral": [
            {"question_id": "SA-001", "category": "security_analysis",
             "question": "What are the attack patterns for CVE-2021-44228?",
             "faithfulness": 0.78, "answer_relevancy": 0.82, "context_precision": 0.74,
             "latency_ms": 2650, "error": None},
            {"question_id": "SA-002", "category": "security_analysis",
             "question": "CVSS score of CVE-2017-0144?",
             "faithfulness": 0.80, "answer_relevancy": 0.83, "context_precision": 0.76,
             "latency_ms": 2420, "error": None},
            {"question_id": "LOG-001", "category": "log_analysis",
             "question": "Identify suspicious Snort alert activities",
             "faithfulness": 0.72, "answer_relevancy": 0.76, "context_precision": 0.66,
             "latency_ms": 3100, "error": None},
            {"question_id": "KG-001", "category": "kg_qa",
             "question": "Which CVEs are linked to CWE-79?",
             "faithfulness": 0.75, "answer_relevancy": 0.79, "context_precision": 0.70,
             "latency_ms": 2950, "error": "Partial context retrieval"},
        ],
    }
}

LLM_COLORS = {
    "gpt-4o-mini": "#00b4ff",
    "mistral":     "#b967ff",
}


# ============================================================
# Data Loader
# ============================================================
def _load_eval_results() -> Optional[Dict]:
    """
    Load hasil evaluasi dari results directory, atau gunakan mock.

    Returns:
        Dict dengan eval results, atau MOCK_EVAL_RESULTS jika belum ada.
    """
    results_dir = Path(os.getenv("EVAL_RESULTS_DIR", "./evaluation/results"))

    if results_dir.exists():
        json_files = sorted(results_dir.glob("eval_results_*.json"), reverse=True)
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    return None


# ============================================================
# Main Evaluation Dashboard Page
# ============================================================
def render_eval_page() -> None:
    """Render halaman Evaluation Dashboard."""

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size:1.8rem; font-weight:700; color:#e8eaf0; margin:0;">
            📊 LLM Evaluation Dashboard
        </h1>
        <p style="color:#6b7a99; font-size:0.88rem; margin:0.3rem 0 0;">
            Compare GPT-4o-mini vs Mistral-7B performance on cybersecurity QA benchmarks
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Results ──────────────────────────────────────────
    real_results = _load_eval_results()
    is_mock = real_results is None
    eval_data = real_results if real_results else MOCK_EVAL_RESULTS

    if is_mock:
        st.info(
            "⚠️ **Demo Mode** — Menampilkan hasil evaluasi simulasi. "
            "Jalankan `python evaluation/run_eval.py --llm gpt4o-mini mistral` "
            "untuk mendapatkan hasil nyata.",
            icon="📊"
        )

    summaries = eval_data.get("summaries", [])
    if not summaries:
        st.error("Tidak ada data evaluasi.")
        return

    # ── Metadata Row ──────────────────────────────────────────
    meta = eval_data.get("metadata", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Benchmark Size</div>"
            f"<div class='metric-value'>{meta.get('benchmark_size', 30)}</div>"
            f"<div class='metric-sub'>questions</div></div>",
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>LLMs Evaluated</div>"
            f"<div class='metric-value'>{len(summaries)}</div>"
            f"<div class='metric-sub'>models</div></div>",
            unsafe_allow_html=True
        )
    with m3:
        best = max(summaries, key=lambda s: s["avg_overall_score"])
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Best Overall</div>"
            f"<div class='metric-value' style='font-size:1.2rem;color:#00ff88;'>{best['llm_name']}</div>"
            f"<div class='metric-sub'>score: {best['avg_overall_score']:.4f}</div></div>",
            unsafe_allow_html=True
        )
    with m4:
        fastest = min(summaries, key=lambda s: s["avg_latency_ms"])
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Fastest Response</div>"
            f"<div class='metric-value' style='font-size:1.2rem;color:#00b4ff;'>{fastest['llm_name']}</div>"
            f"<div class='metric-sub'>{fastest['avg_latency_ms']:.0f} ms avg</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_overview, tab_categories, tab_detail, tab_run = st.tabs([
        "📈 Overview", "📂 By Category", "📋 Detailed Results", "▶️ Run Evaluation"
    ])

    with tab_overview:
        _render_overview_charts(summaries)

    with tab_categories:
        _render_category_breakdown(summaries)

    with tab_detail:
        _render_detailed_results(eval_data.get("detailed_results", {}))

    with tab_run:
        _render_run_eval_tab()


def _render_overview_charts(summaries: List[Dict]) -> None:
    """Render bar chart dan radar chart perbandingan LLM."""

    col_bar, col_radar = st.columns(2)

    # ── Bar Chart ─────────────────────────────────────────────
    with col_bar:
        metrics = ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_overall_score"]
        metric_labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Overall Score"]

        bar_data = []
        for s in summaries:
            for m, label in zip(metrics, metric_labels):
                bar_data.append({
                    "LLM": s["llm_name"],
                    "Metric": label,
                    "Score": s[m],
                })

        df_bar = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            df_bar,
            x="Metric",
            y="Score",
            color="LLM",
            barmode="group",
            title="Metric Comparison",
            color_discrete_map=LLM_COLORS,
            range_y=[0, 1.05],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899b8"),
            title=dict(font=dict(color="#e8eaf0", size=14)),
            xaxis=dict(gridcolor="#1e2d4a", color="#6b7a99", tickangle=-15),
            yaxis=dict(gridcolor="#1e2d4a", color="#6b7a99", tickformat=".2f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899b8")),
            margin=dict(t=40, b=40, l=20, r=10),
            height=320,
        )
        # Tambahkan value labels
        fig_bar.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            textfont=dict(size=10, color="#e8eaf0")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar Chart ───────────────────────────────────────────
    with col_radar:
        categories = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Overall\nScore"]
        radar_keys = ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_overall_score"]

        fig_radar = go.Figure()
        for s in summaries:
            values = [s[k] for k in radar_keys]
            values.append(values[0])  # close polygon
            cats = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=cats,
                fill="toself",
                fillcolor=LLM_COLORS.get(s["llm_name"], "#4a6080") + "30",
                line=dict(color=LLM_COLORS.get(s["llm_name"], "#4a6080"), width=2),
                name=s["llm_name"],
            ))

        fig_radar.update_layout(
            title=dict(text="Performance Radar", font=dict(color="#e8eaf0", size=14)),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    color="#3a4a6a", gridcolor="#1e2d4a",
                    tickfont=dict(color="#6b7a99", size=9),
                ),
                angularaxis=dict(color="#6b7a99", gridcolor="#1e2d4a"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899b8")),
            margin=dict(t=50, b=10, l=50, r=50),
            height=320,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Latency Comparison ────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.9rem; font-weight:600; color:#e8eaf0; margin:0.5rem 0 0.8rem;'>"
        "⏱️ Average Response Latency"
        "</div>",
        unsafe_allow_html=True
    )
    lat_cols = st.columns(len(summaries))
    for col, s in zip(lat_cols, summaries):
        with col:
            color = LLM_COLORS.get(s["llm_name"], "#4a6080")
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{s['llm_name']}</div>"
                f"<div class='metric-value' style='color:{color};'>{s['avg_latency_ms']:.0f}</div>"
                f"<div class='metric-sub'>ms average</div>"
                f"</div>",
                unsafe_allow_html=True
            )


def _render_category_breakdown(summaries: List[Dict]) -> None:
    """Render breakdown per kategori (Security Analysis, Log Analysis, KG QA)."""

    categories = ["security_analysis", "log_analysis", "kg_qa"]
    cat_labels = {
        "security_analysis": "🔐 Security Analysis",
        "log_analysis":      "📋 Log Analysis",
        "kg_qa":             "🔍 KG Question Answering",
    }

    # Build data
    rows = []
    for s in summaries:
        for cat in categories:
            cat_data = s.get("results_by_category", {}).get(cat, {})
            rows.append({
                "LLM": s["llm_name"],
                "Category": cat_labels.get(cat, cat),
                "Faithfulness": cat_data.get("avg_faithfulness", 0),
                "Answer Relevancy": cat_data.get("avg_answer_relevancy", 0),
                "Context Precision": cat_data.get("avg_context_precision", 0),
                "Overall": cat_data.get("avg_overall", 0),
                "N": cat_data.get("count", 0),
            })

    df = pd.DataFrame(rows)

    # Per-category bar charts
    for cat in categories:
        cat_label = cat_labels.get(cat, cat)
        st.markdown(
            f"<div style='font-size:0.95rem; font-weight:600; color:#e8eaf0; "
            f"margin-top:1rem; margin-bottom:0.5rem;'>{cat_label}</div>",
            unsafe_allow_html=True
        )

        cat_df = df[df["Category"] == cat_label].melt(
            id_vars=["LLM"],
            value_vars=["Faithfulness", "Answer Relevancy", "Context Precision", "Overall"],
            var_name="Metric", value_name="Score"
        )

        fig = px.bar(
            cat_df, x="Metric", y="Score", color="LLM",
            barmode="group", range_y=[0, 1.05],
            color_discrete_map=LLM_COLORS,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899b8"),
            xaxis=dict(gridcolor="#1e2d4a", color="#6b7a99"),
            yaxis=dict(gridcolor="#1e2d4a", color="#6b7a99", tickformat=".2f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899b8")),
            margin=dict(t=10, b=10, l=20, r=10),
            height=230,
            showlegend=(cat == "security_analysis"),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_detailed_results(detailed_results: Dict) -> None:
    """Render tabel detail per pertanyaan per LLM."""

    if not detailed_results:
        st.info("Belum ada detailed results.")
        return

    all_rows = []
    for llm_name, results in detailed_results.items():
        for r in results:
            overall = round(
                r.get("faithfulness", 0) * 0.35 +
                r.get("answer_relevancy", 0) * 0.35 +
                r.get("context_precision", 0) * 0.30,
                4
            )
            all_rows.append({
                "LLM": llm_name,
                "ID": r.get("question_id", ""),
                "Category": r.get("category", "").replace("_", " ").title(),
                "Question (shortened)": r.get("question", "")[:60] + "...",
                "Faithfulness": r.get("faithfulness", 0),
                "Relevancy": r.get("answer_relevancy", 0),
                "Precision": r.get("context_precision", 0),
                "Overall": overall,
                "Latency (ms)": int(r.get("latency_ms", 0)),
                "Error": "❌" if r.get("error") else "✅",
            })

    if not all_rows:
        st.warning("Tidak ada detail results tersedia.")
        return

    df_detail = pd.DataFrame(all_rows)

    # LLM filter
    llm_filter = st.selectbox(
        "Filter by LLM",
        options=["All"] + list(detailed_results.keys()),
        key="eval_llm_filter",
    )
    if llm_filter != "All":
        df_detail = df_detail[df_detail["LLM"] == llm_filter]

    st.dataframe(
        df_detail,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Faithfulness": st.column_config.ProgressColumn(
                "Faithfulness", min_value=0, max_value=1, format="%.3f"
            ),
            "Relevancy": st.column_config.ProgressColumn(
                "Relevancy", min_value=0, max_value=1, format="%.3f"
            ),
            "Precision": st.column_config.ProgressColumn(
                "Precision", min_value=0, max_value=1, format="%.3f"
            ),
            "Overall": st.column_config.ProgressColumn(
                "Overall", min_value=0, max_value=1, format="%.3f"
            ),
        }
    )


def _render_run_eval_tab() -> None:
    """Render tab untuk menjalankan evaluasi secara langsung dari UI."""

    st.markdown(
        "<div style='font-size:0.88rem; color:#8899b8; margin-bottom:1rem;'>"
        "Run the evaluation pipeline directly from this interface:"
        "</div>",
        unsafe_allow_html=True
    )

    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        selected_llms = st.multiselect(
            "LLMs to Evaluate",
            options=["gpt-4o-mini", "mistral"],
            default=["gpt-4o-mini", "mistral"],
        )
        cat_filter = st.selectbox(
            "Category Filter",
            options=["all", "security_analysis", "log_analysis", "kg_qa"],
        )
        mock_mode = st.checkbox(
            "Mock Mode (no API calls)",
            value=True,
            help="Use mock answer generator for testing without API keys"
        )

    with col_run:
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        run_clicked = st.button(
            "▶️ Run Evaluation",
            use_container_width=True,
            disabled=not selected_llms,
        )

    if run_clicked:
        if not selected_llms:
            st.error("Pilih minimal satu LLM.")
            return

        cmd_args = (
            f"python evaluation/run_eval.py "
            f"--llm {' '.join(selected_llms)} "
            f"--category {cat_filter}"
            + (" --mock" if mock_mode else "")
        )

        st.info(
            f"**Command to run:**\n```\n{cmd_args}\n```\n\n"
            f"Jalankan command di atas di terminal dari root project directory.\n"
            f"Hasil akan otomatis muncul di tab Overview setelah selesai.",
            icon="▶️"
        )

        st.code(cmd_args, language="bash")

    st.markdown("<hr style='border-color:#1e2d4a; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.85rem; color:#6b7a99;'>"
        "📌 <strong>Evaluation Metrics:</strong><br>"
        "• <strong>Faithfulness</strong> (35%): Is the answer grounded in retrieved KG/log context?<br>"
        "• <strong>Answer Relevancy</strong> (35%): Does the answer address the question?<br>"
        "• <strong>Context Precision</strong> (30%): Is the retrieved context accurate?<br>"
        "• <strong>Latency</strong>: Average response time in milliseconds<br><br>"
        "Judge model: <code>gpt-4o-mini</code> (LLM-as-a-Judge approach)"
        "</div>",
        unsafe_allow_html=True
    )
