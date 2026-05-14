# SEPSES CSKG LLM Chatbot

> LLM-based Chatbot Interface for Cybersecurity Analysis using SEPSES Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/RAG-LangChain-green.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://trychroma.com)

---

## 📋 Deskripsi Proyek

Sistem chatbot berbasis LLM yang terintegrasi dengan **SEPSES Cybersecurity Knowledge Graph (CSKG)** untuk analisis keamanan siber. Mengimplementasikan arsitektur **Hybrid RAG + GraphRAG** yang menggabungkan:
- SPARQL query ke SEPSES KG (CVE, CWE, CAPEC, CPE, ATT&CK)
- Semantic search atas log keamanan lokal via ChromaDB
- Multi-LLM evaluation (GPT-4o-mini vs Mistral-7B)

**Referensi Paper:**
- Kiesling et al. (2019). *The SEPSES Knowledge Graph: An Integrated Resource for Cybersecurity*. ISWC 2019.
- Ekelhart et al. (2024). *ICS-SEC KG*. ISWC 2024.
- CEUR Vol-4079 Paper 11.

---

## 👥 Tim Pengembang

| Peran | Nama | Branch |
|-------|------|--------|
| Knowledge Architect | Ajie Armansyah Sunaryo | `feature/kg-engine` |
| RAG Logic Dev | Fahmi Abdillah Zain | `feature/rag-logic` |
| Full-Stack UI Dev | Muhammad Dhafin Alfeizar Gandhan | `feature/frontend-ui` |
| Evaluator & Log Dev | Satya Wira Pramudita | `feature/eval-log-dev` |

---

## 🏗️ Arsitektur

```
User Query
    │
    ▼
Streamlit Frontend (Dhafin)
    │
    ▼
RAG Pipeline Orchestrator (Fahmi)
    ├── NL2SPARQL → SEPSES SPARQL Endpoint (Ajie)
    │       CVE / CWE / CAPEC / CPE / ATT&CK
    └── Vector Search → ChromaDB (Satya)
            Local Security Logs (Snort / Syslog / Windows Event)
    │
    ▼
LLM Generator: GPT-4o-mini | Mistral-7B (Fahmi)
    │
    ▼
Response + KG Graph Visualization + Source Citations
    │
    ▼
LLM-as-a-Judge Evaluator (Satya)
```

---

## 🚀 Cara Menjalankan

### Prerequisites
- Python 3.10+
- (Opsional) Ollama untuk Mistral lokal: [https://ollama.ai](https://ollama.ai)
- (Opsional) Docker untuk Jena Fuseki lokal

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/satyawirapramudita/SEPSES-CSKG-LLM-Chatbot.git
cd SEPSES-CSKG-LLM-Chatbot

# Buat virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi

```bash
# Salin template dan isi nilai yang diperlukan
copy .env.example .env
# Edit .env dengan API key dan konfigurasi yang sesuai
```

### 3. Jalankan Aplikasi

```bash
streamlit run frontend/app.py
```

### 4. Jalankan Evaluasi

```bash
# Mock mode (tanpa API key)
python evaluation/run_eval.py --llm gpt4o-mini mistral --mock

# Real mode
python evaluation/run_eval.py --llm gpt4o-mini mistral --category all
```

---

## 📁 Struktur Proyek

```
SEPSES-CSKG-LLM-Chatbot/
├── .env.example              # Template environment variables
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Fuseki + Ollama + App services
│
├── kg_engine/                # [Ajie] Knowledge Graph Engine
│   ├── sparql_client.py      # SEPSES SPARQL endpoint client
│   ├── graph_builder.py      # NetworkX graph builder
│   ├── ontology_schema.ttl   # SEPSES ontology schema
│   └── queries/              # SPARQL query templates
│       ├── vulnerability_lookup.rq
│       └── get_capec_from_cve.rq
│
├── rag_logic/                # [Fahmi] RAG Pipeline
│   ├── rag_pipeline.py       # Main orchestrator
│   ├── nl2sparql.py          # NL → SPARQL (LangChain)
│   ├── multi_hop.py          # Multi-hop KG reasoning
│   ├── llm_connector.py      # GPT/Mistral abstraction
│   └── prompt_templates.py   # System/user prompts
│
├── log_analysis/             # [Satya] Log Analysis + Vector DB
│   ├── log_parser.py         # Snort/Syslog/WinEvent/Apache parser
│   ├── vector_store.py       # ChromaDB wrapper
│   └── hybrid_retriever.py   # BM25 + Semantic + RRF fusion
│
├── frontend/                 # [Dhafin] Streamlit Frontend
│   ├── app.py                # Multi-page Streamlit app
│   └── components/
│       ├── chat_window.py    # Chat UI + citations
│       ├── graph_visualizer.py # pyvis KG graph
│       ├── log_uploader.py   # Log file upload
│       └── eval_dashboard.py # Evaluation charts
│
├── evaluation/               # [Satya] Evaluation Framework
│   ├── benchmark_dataset.json # 30 pertanyaan benchmark
│   ├── grader.py             # LLM-as-a-Judge pipeline
│   ├── run_eval.py           # CLI runner
│   └── results/              # Output evaluasi
│
└── data/
    ├── cskg_dumps/           # SEPSES RDF dump files
    ├── chroma_db/            # ChromaDB persistent storage
    └── sample_logs/          # Sample security logs
```

---

## 📊 Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| Security Analysis | Analisis CVE, CWE, CAPEC via SPARQL ke SEPSES KG |
| Threat Actor Analysis | Multi-hop traversal CVE→CWE→CAPEC→ATT&CK |
| Malware Investigation | Investigasi teknik malware via KG |
| Log Analysis | Upload + analisis Snort/Syslog/Windows Event Log |
| KG QA | Question-answering langsung atas SEPSES CSKG |
| Graph Visualization | Visualisasi interaktif relasi entitas KG |
| Multi-LLM Evaluation | Perbandingan GPT-4o-mini vs Mistral-7B |

---

## 📚 Sumber Data SEPSES CSKG

| Dataset | URL |
|---------|-----|
| SPARQL Endpoint | https://w3id.org/sepses/sparql |
| RDF Dumps | https://w3id.org/sepses/dumps/ |
| CVE Vocabulary | http://w3id.org/sepses/vocab/ref/cve |
| CWE Vocabulary | http://w3id.org/sepses/vocab/ref/cwe |
| CAPEC Vocabulary | http://w3id.org/sepses/vocab/ref/capec |
| CPE Vocabulary | http://w3id.org/sepses/vocab/ref/cpe |
| CVSS Vocabulary | http://w3id.org/sepses/vocab/ref/cvss |

---

## 🔒 Security Notes

- Semua API key disimpan di `.env` (tidak di-commit ke Git)
- Lihat `.env.example` untuk template konfigurasi
- Input sanitization diimplementasikan di setiap endpoint
- Prepared statements digunakan untuk semua SPARQL queries

---

## 📄 Lisensi

MIT License — lihat [LICENSE](LICENSE) untuk detail.
