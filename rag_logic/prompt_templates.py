"""
SEPSES CSKG LLM Chatbot - Prompt Templates
============================================
Tanggung Jawab  : Fahmi Abdillah Zain (RAG Logic Dev)
Branch          : feature/rag-logic

Deskripsi:
    Semua system/user prompt templates untuk mode analisis yang berbeda.
    Templates menggunakan format f-string dengan placeholder {context} dan {question}.
    Dirancang untuk menghasilkan respons yang grounded, explainable, dan terstruktur.
"""

# ============================================================
# System Prompts — mendefinisikan perilaku LLM
# ============================================================

SYSTEM_SECURITY_ANALYSIS = """You are a senior cybersecurity analyst with expertise in threat intelligence \
and vulnerability research. You have access to structured data from the SEPSES \
Cybersecurity Knowledge Graph (CSKG), which integrates CVE, CWE, CAPEC, CPE, CVSS, \
and MITRE ATT&CK data.

When answering:
1. Always ground your response in the provided KG context.
2. Cite specific CVE IDs, CWE categories, CAPEC patterns, and CVSS scores.
3. Structure your response with clear sections.
4. If the context doesn't contain enough information, clearly state the limitation.
5. Suggest concrete mitigation steps when relevant.
6. Use professional, precise language appropriate for security professionals."""

SYSTEM_LOG_ANALYSIS = """You are a Security Operations Center (SOC) analyst expert in log analysis \
and incident response. You analyze security logs retrieved from a vector database \
and correlate findings with the SEPSES Cybersecurity Knowledge Graph.

When analyzing logs:
1. Identify attack patterns and classify by severity (Critical/High/Medium/Low).
2. Map findings to MITRE ATT&CK tactics and techniques when possible.
3. Reference specific CVEs if mentioned in the logs.
4. Provide actionable recommendations for incident response.
5. Structure output: Findings → Threat Classification → Recommendations."""

SYSTEM_KG_QA = """You are a knowledge graph expert specializing in cybersecurity ontologies. \
You answer questions about the SEPSES Cybersecurity Knowledge Graph structure, \
its data, and relationships between security entities (CVE, CWE, CAPEC, CPE, CVSS).

When answering:
1. Reference specific SPARQL query patterns when explaining how to retrieve data.
2. Explain relationships between entities clearly.
3. Provide accurate counts, statistics, or examples from the retrieved context.
4. If asked to generate a SPARQL query, provide a syntactically correct query."""

SYSTEM_NL2SPARQL = """You are a SPARQL query generator specialized in the SEPSES Cybersecurity \
Knowledge Graph. Convert natural language questions into valid SPARQL SELECT queries.

SEPSES KG Prefixes:
  PREFIX cve:   <http://w3id.org/sepses/vocab/ref/cve#>
  PREFIX cwe:   <http://w3id.org/sepses/vocab/ref/cwe#>
  PREFIX capec: <http://w3id.org/sepses/vocab/ref/capec#>
  PREFIX cpe:   <http://w3id.org/sepses/vocab/ref/cpe#>
  PREFIX cvss:  <http://w3id.org/sepses/vocab/ref/cvss#>

Key properties:
  cve:cveId, cve:description, cve:issued
  cve:hasCWE → cwe:CWE
  cve:hasCPE → cpe:CPE (cpe:productId)
  cve:hasCVSS → cvss:BaseMetric (cvss:baseScore, cvss:attackVector)
  cwe:hasCAPEC → capec:CAPEC (capec:capecId, capec:name)

Resource URIs:
  CVE:   http://w3id.org/sepses/resource/cve/CVE-YYYY-NNNN
  CWE:   http://w3id.org/sepses/resource/cwe/CWE-N
  CAPEC: http://w3id.org/sepses/resource/capec/CAPEC-N

Rules:
- Return ONLY the SPARQL query, no explanation.
- Always include PREFIX declarations.
- Use OPTIONAL for properties that might not exist.
- Add LIMIT 20 unless count query."""

# ============================================================
# Few-Shot Examples untuk NL2SPARQL
# ============================================================

NL2SPARQL_FEW_SHOTS = [
    {
        "question": "What is the CVSS score of CVE-2021-44228?",
        "sparql": """PREFIX cve:  <http://w3id.org/sepses/vocab/ref/cve#>
PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
SELECT ?score ?attackVector WHERE {
  <http://w3id.org/sepses/resource/cve/CVE-2021-44228>
    cve:hasCVSS ?cvssNode .
  ?cvssNode cvss:baseScore    ?score .
  OPTIONAL { ?cvssNode cvss:attackVector ?attackVector . }
}""",
    },
    {
        "question": "Which CVEs are linked to CWE-89?",
        "sparql": """PREFIX cve: <http://w3id.org/sepses/vocab/ref/cve#>
SELECT ?cveId WHERE {
  ?cve cve:cveId ?cveId ;
       cve:hasCWE <http://w3id.org/sepses/resource/cwe/CWE-89> .
} LIMIT 20""",
    },
    {
        "question": "What attack patterns exploit Log4Shell CVE-2021-44228?",
        "sparql": """PREFIX cve:   <http://w3id.org/sepses/vocab/ref/cve#>
PREFIX cwe:   <http://w3id.org/sepses/vocab/ref/cwe#>
PREFIX capec: <http://w3id.org/sepses/vocab/ref/capec#>
SELECT ?cweId ?cweName ?capecId ?capecName WHERE {
  <http://w3id.org/sepses/resource/cve/CVE-2021-44228> cve:hasCWE ?cwe .
  OPTIONAL { ?cwe cwe:cweId ?cweId ; cwe:name ?cweName . }
  OPTIONAL {
    ?cwe cwe:hasCAPEC ?capec .
    ?capec capec:capecId ?capecId ; capec:name ?capecName .
  }
}""",
    },
    {
        "question": "Find all critical CVEs affecting Apache products",
        "sparql": """PREFIX cve:  <http://w3id.org/sepses/vocab/ref/cve#>
PREFIX cpe:  <http://w3id.org/sepses/vocab/ref/cpe#>
PREFIX cvss: <http://w3id.org/sepses/vocab/ref/cvss#>
SELECT ?cveId ?score ?product WHERE {
  ?cve cve:cveId  ?cveId ;
       cve:hasCPE ?cpe ;
       cve:hasCVSS ?cvssNode .
  ?cpe cpe:productId ?product .
  ?cvssNode cvss:baseScore ?score .
  FILTER(CONTAINS(LCASE(STR(?product)), "apache"))
  FILTER(?score >= 9.0)
} ORDER BY DESC(?score) LIMIT 20""",
    },
]


# ============================================================
# User Prompt Templates (f-string)
# ============================================================

def security_analysis_prompt(question: str, kg_context: str, log_context: str = "") -> str:
    """
    Prompt untuk mode Security Analysis.

    Args:
        question   : Pertanyaan user.
        kg_context : Context dari SPARQL query (formatted string).
        log_context: Context dari ChromaDB log search (opsional).

    Returns:
        str: Formatted user prompt.
    """
    log_section = ""
    if log_context:
        log_section = f"""
## Related Security Log Entries
{log_context}
"""

    return f"""## Knowledge Graph Context (from SEPSES CSKG)
{kg_context}
{log_section}
---
## Question
{question}

Please provide a comprehensive security analysis based on the context above. \
Structure your response with:
1. **Summary** — one paragraph overview
2. **Vulnerability Details** — CVE, CWE, CAPEC breakdown
3. **Risk Assessment** — CVSS score interpretation and impact
4. **Attack Chain** — how this vulnerability can be exploited
5. **Recommendations** — concrete mitigation steps
6. **KG Sources** — cite the specific entities retrieved from SEPSES KG"""


def log_analysis_prompt(question: str, log_context: str, kg_context: str = "") -> str:
    """
    Prompt untuk mode Log Analysis.

    Args:
        question   : Pertanyaan user tentang log.
        log_context: Retrieved log entries dari ChromaDB.
        kg_context : KG enrichment untuk CVE mentions (opsional).

    Returns:
        str: Formatted user prompt.
    """
    kg_section = ""
    if kg_context:
        kg_section = f"""
## KG Enrichment (from SEPSES CSKG)
{kg_context}
"""

    return f"""## Retrieved Security Log Entries
{log_context}
{kg_section}
---
## Question
{question}

Analyze the log entries above and provide:
1. **Detected Threats** — list with severity classification
2. **Attack Stage** — MITRE ATT&CK kill chain phase if applicable
3. **Affected Systems** — source/destination IPs and services
4. **CVE Correlation** — if any CVE IDs are mentioned, provide context
5. **Immediate Actions** — prioritized incident response steps"""


def kg_qa_prompt(question: str, kg_context: str) -> str:
    """
    Prompt untuk mode KG Question Answering.

    Args:
        question  : Pertanyaan user tentang KG.
        kg_context: Retrieved data dari SPARQL.

    Returns:
        str: Formatted user prompt.
    """
    return f"""## SEPSES Knowledge Graph Data
{kg_context}

---
## Question
{question}

Answer the question based on the KG data above. Be precise and reference \
specific entities (CVE IDs, CWE numbers, CAPEC IDs, CVSS scores) from the context."""


def nl2sparql_prompt(question: str) -> str:
    """
    Prompt untuk NL2SPARQL conversion.

    Args:
        question: Natural language question.

    Returns:
        str: Formatted prompt dengan few-shot examples.
    """
    examples_str = "\n\n".join(
        f"Q: {ex['question']}\nSPARQL:\n```sparql\n{ex['sparql']}\n```"
        for ex in NL2SPARQL_FEW_SHOTS
    )

    return f"""Convert the following question to a SPARQL query for SEPSES CSKG.

## Examples
{examples_str}

## New Question
Q: {question}
SPARQL:"""


def format_kg_context(data: dict) -> str:
    """
    Format dict hasil SPARQL/get_cve_details menjadi readable string untuk LLM context.

    Args:
        data: Dict dari SparqlClient method atau list of dicts.

    Returns:
        str: Human-readable context string.
    """
    if not data:
        return "No relevant data found in SEPSES Knowledge Graph."

    if isinstance(data, list):
        lines = []
        for i, item in enumerate(data[:10], 1):
            line_parts = [f"{i}."]
            for k, v in item.items():
                if v:
                    line_parts.append(f"{k}: {v}")
            lines.append(" | ".join(line_parts))
        return "\n".join(lines)

    # Dict (hasil get_cve_details)
    lines = []
    if data.get("cve_id"):
        lines.append(f"CVE ID: {data['cve_id']}")
    if data.get("cvss_score"):
        lines.append(f"CVSS Score: {data['cvss_score']}")
    if data.get("attack_vector"):
        lines.append(f"Attack Vector: {data['attack_vector']}")
    if data.get("description"):
        lines.append(f"Description: {data['description'][:300]}")
    if data.get("cwes"):
        lines.append(f"CWEs: {'; '.join(data['cwes'])}")
    if data.get("capecs"):
        lines.append(f"CAPECs: {'; '.join(data['capecs'])}")
    if data.get("cpe_products"):
        lines.append(f"Affected Products: {', '.join(data['cpe_products'][:5])}")
    return "\n".join(lines) if lines else "No structured data available."
