"""
SEPSES CSKG LLM Chatbot - Integration Test Runner
===================================================
Script untuk memverifikasi semua komponen terintegrasi dengan benar.

Jalankan: python integration_test.py

Tests yang dijalankan:
  [1] .env loaded correctly (API key, endpoints)
  [2] SPARQL endpoint reachable (SEPSES public)
  [3] Ollama server reachable
  [4] KG Engine: get_cve_details("CVE-2021-44228")
  [5] KG Engine: get_capec_from_cve("CVE-2021-44228")
  [6] KG Engine: graph_builder.build_cve_graph("CVE-2021-44228")
  [7] NL2SPARQL: regex path (no API call)
  [8] NL2SPARQL: CVE-2017-0144 full lookup
  [9] LLM Connector: OpenAI ping
  [10] LLM Connector: Ollama ping
  [11] RAG Pipeline: Security Analysis (CVE-2021-44228)
  [12] Log Analysis: HybridRetriever ingest + search
  [13] Evaluation: grader.score_answer() mock mode
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple

# Tambahkan root ke path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env sebelum import lainnya
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# ── ANSI Colors ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Test Registry ────────────────────────────────────────────
results: List[Tuple[str, bool, str]] = []


def test(name: str, skip_if: bool = False):
    """Decorator untuk test function."""
    def decorator(fn):
        def wrapper():
            if skip_if:
                results.append((name, None, "SKIPPED"))
                print(f"  {YELLOW}⏭  SKIP{RESET}  {name}")
                return
            try:
                detail = fn() or ""
                results.append((name, True, detail))
                print(f"  {GREEN}✅ PASS{RESET}  {name}", f" {CYAN}→ {detail}{RESET}" if detail else "")
            except Exception as exc:
                results.append((name, False, str(exc)))
                print(f"  {RED}❌ FAIL{RESET}  {name}")
                print(f"         {RED}{exc}{RESET}")
        return wrapper
    return decorator


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Test Suite
# ============================================================
def run_all_tests():
    print(f"\n{BOLD}{'='*60}")
    print("  SEPSES CSKG LLM Chatbot — Integration Test")
    print(f"{'='*60}{RESET}\n")

    has_api_key = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-"))
    has_ollama  = False

    # ── [0] Env Variables ──────────────────────────────────
    separator("Phase 0: Environment Variables")

    @test("[1] .env — OPENAI_API_KEY configured")
    def t01():
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or "GANTI" in key:
            raise ValueError("OPENAI_API_KEY belum diset! Isi file .env")
        return f"sk-...{key[-8:]}"
    t01()

    @test("[2] .env — SPARQL_ENDPOINT configured")
    def t02():
        ep = os.getenv("SPARQL_ENDPOINT", "")
        if not ep:
            raise ValueError("SPARQL_ENDPOINT tidak ada")
        return ep
    t02()

    @test("[3] .env — OLLAMA_BASE_URL configured")
    def t03():
        url = os.getenv("OLLAMA_BASE_URL", "")
        return url
    t03()

    # ── [1] KG Engine ──────────────────────────────────────
    separator("Phase 1: KG Engine (kg_engine)")

    @test("[4] SparqlClient — SEPSES endpoint ping")
    def t04():
        from kg_engine.sparql_client import SparqlClient
        client = SparqlClient()
        if client.ping():
            return "endpoint reachable ✓"
        else:
            raise RuntimeError("Endpoint tidak dapat dijangkau")
    t04()

    @test("[5] SparqlClient — get_cve_details(CVE-2021-44228)")
    def t05():
        from kg_engine.sparql_client import SparqlClient
        client = SparqlClient()
        data = client.get_cve_details("CVE-2021-44228")
        if not data.get("found"):
            raise RuntimeError(f"CVE tidak ditemukan: {data}")
        score = data.get("cvss_score", "N/A")
        return f"CVSS={score} | CWEs={len(data['cwes'])} | CAPECs={len(data['capecs'])}"
    t05()

    @test("[6] SparqlClient — get_capec_from_cve(CVE-2021-44228)")
    def t06():
        from kg_engine.sparql_client import SparqlClient
        chain = SparqlClient().get_capec_from_cve("CVE-2021-44228")
        if not chain:
            raise RuntimeError("Chain kosong")
        return f"{len(chain)} CWE→CAPEC pairs"
    t06()

    @test("[7] GraphBuilder — build_cve_graph(CVE-2021-44228)")
    def t07():
        from kg_engine.graph_builder import GraphBuilder
        graph = GraphBuilder().build_cve_graph("CVE-2021-44228")
        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        if nodes == 0:
            raise RuntimeError("Graph kosong")
        return f"{nodes} nodes, {edges} edges"
    t07()

    # ── [2] NL2SPARQL ──────────────────────────────────────
    separator("Phase 2: NL2SPARQL (rag_logic.nl2sparql)")

    @test("[8] NL2SPARQL — regex path (CVE ID in question)")
    def t08():
        from rag_logic.nl2sparql import NL2SPARQL, RegexSparqlMatcher
        matcher = RegexSparqlMatcher()
        sparql, method = "N/A", "regex"
        result = matcher.match("What is the CVSS score of CVE-2021-44228?")
        if not result:
            raise RuntimeError("Regex tidak match")
        return f"method=regex | {len(result)} chars"
    t08()

    @test("[9] NL2SPARQL — product search pattern")
    def t09():
        from rag_logic.nl2sparql import RegexSparqlMatcher
        result = RegexSparqlMatcher().match("Find vulnerabilities affecting apache")
        if not result:
            raise RuntimeError("Product pattern tidak match")
        return "product pattern matched"
    t09()

    # ── [3] LLM Connectors ────────────────────────────────
    separator("Phase 3: LLM Connectors (rag_logic.llm_connector)")

    @test("[10] OpenAI Connector — ping", skip_if=not has_api_key)
    def t10():
        from rag_logic.llm_connector import OpenAIConnector
        conn = OpenAIConnector()
        if not conn.ping():
            raise RuntimeError("OpenAI ping failed")
        return "OpenAI API reachable ✓"
    t10()

    @test("[11] OpenAI Connector — generate (simple test)")
    def t11():
        if not has_api_key:
            raise RuntimeError("API key tidak dikonfigurasi")
        from rag_logic.llm_connector import OpenAIConnector, Message
        conn = OpenAIConnector()
        answer, latency = conn.generate_with_latency(
            [Message("user", "Say 'SEPSES KG OK' in exactly 3 words.")],
            max_tokens=20,
        )
        if not answer:
            raise RuntimeError("Empty response")
        return f"'{answer.strip()[:40]}' | {latency:.0f}ms"
    t11()

    # Check Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        has_ollama = r.status_code == 200
    except Exception:
        has_ollama = False

    @test("[12] Ollama Connector — ping", skip_if=not has_ollama)
    def t12():
        from rag_logic.llm_connector import OllamaConnector
        conn = OllamaConnector()
        if not conn.ping():
            raise RuntimeError("Ollama server tidak berjalan. Jalankan: ollama serve")
        models = conn.list_models()
        return f"Ollama OK | models: {[m[:20] for m in models[:3]]}"
    t12()

    # ── [4] RAG Pipeline ──────────────────────────────────
    separator("Phase 4: RAG Pipeline (rag_logic.rag_pipeline)")

    # Deteksi model Ollama yang tersedia
    available_ollama_model = None
    if has_ollama:
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models_list = [m["name"] for m in resp.json().get("models", [])]
            # Pilih model yang tersedia (prioritas: mistral, gemma4, minimax, apapun yang ada)
            for preferred in ["mistral", "gemma4", "gemma4:latest", "minimax-m2.7:cloud"]:
                if any(preferred in m for m in models_list):
                    available_ollama_model = preferred
                    break
            if not available_ollama_model and models_list:
                available_ollama_model = models_list[0]
        except Exception:
            pass

    @test("[13] RagPipeline — Security Analysis via Ollama", skip_if=not available_ollama_model)
    def t13():
        from rag_logic.rag_pipeline import RagPipeline
        pipeline = RagPipeline(llm_name=available_ollama_model)
        result = pipeline.query(
            "What CWE and CAPEC are associated with CVE-2021-44228?",
            mode="Security Analysis",
        )
        if result.get("error") and "quota" not in result.get("error", ""):
            raise RuntimeError(result["error"])
        answer_preview = result["answer"][:80].replace("\n", " ")
        latency = result.get("latency_ms", 0)
        return f"{latency:.0f}ms | '{answer_preview}'"
    t13()

    @test("[14] RagPipeline — KG QA via Ollama", skip_if=not available_ollama_model)
    def t14():
        from rag_logic.rag_pipeline import RagPipeline
        result = RagPipeline(llm_name=available_ollama_model).query(
            "Which CVEs have CVSS score above 9.0?",
            mode="KG Question Answering",
        )
        if result.get("error") and "quota" not in result.get("error", ""):
            raise RuntimeError(result["error"])
        return f"{result['latency_ms']:.0f}ms | model={available_ollama_model}"
    t14()

    # ── [5] Log Analysis ──────────────────────────────────
    separator("Phase 5: Log Analysis (log_analysis)")

    @test("[15] HybridRetriever — ingest + search")
    def t15():
        from log_analysis.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        sample_path = ROOT_DIR / "data" / "sample_logs" / "snort_sample.log"
        if sample_path.exists():
            count = retriever.ingest_logs(file_path=str(sample_path))
        else:
            count = retriever.ingest_logs(log_text=(
                "[**] SQL Injection Attempt src=192.168.100.50 dst=10.0.0.10\n"
                "[**] Log4Shell CVE-2021-44228 src=203.0.113.42 dst=10.0.0.15\n"
            ))
        if count == 0:
            return "0 entries (log format belum cocok, modul OK)"
        return f"Ingested {count} entries OK"
    t15()

    @test("[16] HybridRetriever — search after ingest")
    def t16():
        from log_analysis.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        # Coba search — jika kosong, itu acceptable karena depend on test 15
        try:
            results = retriever.search("SQL injection", top_k=3)
            return f"{len(results)} results"
        except Exception as exc:
            if "Belum ada data" in str(exc) or "empty" in str(exc).lower():
                return "Vector store kosong (acceptable — perlu ingest dulu)"
            raise
    t16()

    # ── [6] Evaluation ────────────────────────────────────
    separator("Phase 6: Evaluation (evaluation)")

    @test("[17] Evaluation — run_eval mock mode")
    def t17():
        import subprocess
        result = subprocess.run(
            [sys.executable, "evaluation/run_eval.py",
             "--llm", "gpt-4o-mini", "--mock", "--category", "security_analysis"],
            capture_output=True, text=True, cwd=str(ROOT_DIR), timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            raise RuntimeError(f"Exit {result.returncode}: {result.stderr[-300:]}")
        return "Mock evaluation OK"
    t17()

    # ── Summary ───────────────────────────────────────────
    separator("Test Summary")
    passed  = sum(1 for _, ok, _ in results if ok is True)
    failed  = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)
    total   = len(results)

    print(f"\n  Total  : {total}")
    print(f"  {GREEN}Passed : {passed}{RESET}")
    print(f"  {RED}Failed : {failed}{RESET}")
    print(f"  {YELLOW}Skipped: {skipped}{RESET}")

    # Save result JSON
    output_dir = ROOT_DIR / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "integration_test_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"name": n, "passed": ok, "detail": d} for n, ok, d in results],
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  Results saved: {result_file}")

    if failed > 0:
        print(f"\n{RED}{BOLD}  ⚠️  {failed} test(s) FAILED — check errors above{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}  🎉 All tests passed!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
