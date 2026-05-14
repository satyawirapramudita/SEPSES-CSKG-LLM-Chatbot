"""
SEPSES CSKG LLM Chatbot - LLM-as-a-Judge Evaluation Grader
===========================================================
Tanggung Jawab  : Satya Wira Pramudita (Evaluator & Log Dev)
Branch          : feature/eval-log-dev
Standar         : IEEE 829, ISO/IEC 12207

Deskripsi:
    Automated evaluation pipeline menggunakan pendekatan "LLM-as-a-Judge".
    Membandingkan performa dua LLM (GPT-4o-mini vs Mistral-7B) pada
    benchmark_dataset.json menggunakan metrik:

    1. Faithfulness    : Apakah jawaban didukung oleh retrieved context?
    2. Answer Relevancy: Seberapa relevan jawaban terhadap pertanyaan?
    3. Context Precision: Ketepatan context yang di-retrieve
    4. SPARQL Accuracy : Validitas SPARQL query yang dihasilkan (untuk KG queries)
    5. Latency         : Waktu respons (ms)

    Output: JSON results + CSV summary di evaluation/results/
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = structlog.get_logger(__name__)

# ============================================================
# Data Classes
# ============================================================

@dataclass
class EvalResult:
    """
    Hasil evaluasi untuk satu pertanyaan dari satu LLM.

    Attributes:
        question_id     : ID dari benchmark_dataset.json
        category        : Kategori pertanyaan
        question        : Teks pertanyaan
        llm_name        : Nama LLM yang dievaluasi
        generated_answer: Jawaban yang dihasilkan LLM
        retrieved_context: Context yang di-retrieve (KG/log)
        faithfulness    : Skor faithfulness (0.0 - 1.0)
        answer_relevancy: Skor answer relevancy (0.0 - 1.0)
        context_precision: Skor context precision (0.0 - 1.0)
        sparql_valid    : Apakah SPARQL yang dihasilkan valid (True/False/None)
        latency_ms      : Waktu respons dalam milidetik
        judge_reasoning : Penjelasan dari LLM judge
        error           : Pesan error jika ada
    """
    question_id: str
    category: str
    question: str
    llm_name: str
    generated_answer: str = ""
    retrieved_context: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    sparql_valid: Optional[bool] = None
    latency_ms: float = 0.0
    judge_reasoning: str = ""
    error: Optional[str] = None

    @property
    def overall_score(self) -> float:
        """Rata-rata weighted score dari semua metrik utama."""
        return round(
            (self.faithfulness * 0.35 +
             self.answer_relevancy * 0.35 +
             self.context_precision * 0.30),
            4
        )


@dataclass
class EvalSummary:
    """Ringkasan hasil evaluasi per LLM."""
    llm_name: str
    total_questions: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_overall_score: float
    avg_latency_ms: float
    error_count: int
    results_by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ============================================================
# LLM Judge Prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert cybersecurity QA evaluator. 
Your task is to evaluate the quality of an AI assistant's answer to a cybersecurity question.
Evaluate strictly and objectively. Return ONLY a valid JSON object.
"""

JUDGE_USER_PROMPT_TEMPLATE = """
Evaluate the following cybersecurity QA result:

**Question**: {question}

**Expected Answer (Ground Truth)**: {expected_answer}

**Generated Answer**: {generated_answer}

**Retrieved Context (from Knowledge Graph/Logs)**: {retrieved_context}

Score the following metrics on a scale of 0.0 to 1.0:

1. **faithfulness** (0-1): Is the generated answer factually supported by the retrieved context? 
   - 1.0: Fully supported by context
   - 0.5: Partially supported  
   - 0.0: Not supported or contradicts context

2. **answer_relevancy** (0-1): How relevant is the generated answer to the question?
   - 1.0: Directly addresses the question with accurate cybersecurity information
   - 0.5: Partially relevant
   - 0.0: Off-topic or wrong

3. **context_precision** (0-1): How precise/accurate is the retrieved context for answering this question?
   - 1.0: Context perfectly matches what's needed
   - 0.5: Context is partially relevant
   - 0.0: Context is irrelevant

Return ONLY this JSON (no explanation outside JSON):
{{
  "faithfulness": <float 0.0-1.0>,
  "answer_relevancy": <float 0.0-1.0>,
  "context_precision": <float 0.0-1.0>,
  "reasoning": "<brief explanation of scores in 2-3 sentences>"
}}
"""

# ============================================================
# Grader Implementation
# ============================================================

class Grader:
    """
    LLM-as-a-Judge automated evaluation grader.

    Workflow:
    1. Load benchmark_dataset.json
    2. Untuk setiap LLM yang dievaluasi:
       a. Kirim setiap pertanyaan ke RAG pipeline (melalui API call atau langsung)
       b. Ukur latency
       c. Kirim (question, expected, generated, context) ke LLM judge
       d. Simpan EvalResult
    3. Hitung EvalSummary per LLM
    4. Simpan ke JSON + CSV
    """

    def __init__(
        self,
        benchmark_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        judge_model: Optional[str] = None,
    ) -> None:
        """
        Inisialisasi Grader.

        Args:
            benchmark_path: Path ke benchmark_dataset.json.
            results_dir   : Direktori untuk menyimpan hasil evaluasi.
            judge_model   : Model LLM yang digunakan sebagai judge.

        Raises:
            FileNotFoundError: Jika benchmark file tidak ditemukan.
            ValueError: Jika OPENAI_API_KEY tidak tersedia.
        """
        self._benchmark_path = benchmark_path or os.path.join(
            os.path.dirname(__file__), "benchmark_dataset.json"
        )
        self._results_dir = results_dir or os.getenv(
            "EVAL_RESULTS_DIR", "./evaluation/results"
        )
        self._judge_model = judge_model or os.getenv("JUDGE_MODEL", "gpt-4o-mini")

        # Validasi API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY tidak ditemukan di environment variables. "
                "Salin .env.example ke .env dan isi API key."
            )

        self._judge_client = OpenAI(api_key=api_key)
        self._benchmark: List[Dict[str, Any]] = []

        # Buat direktori hasil jika belum ada
        os.makedirs(self._results_dir, exist_ok=True)

        self._load_benchmark()

    def _load_benchmark(self) -> None:
        """Load benchmark dataset dari file JSON."""
        path = Path(self._benchmark_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark file tidak ditemukan: {self._benchmark_path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            self._benchmark = json.load(f)
        logger.info("benchmark_loaded", total_questions=len(self._benchmark))

    def evaluate_single(
        self,
        question_item: Dict[str, Any],
        llm_name: str,
        generated_answer: str,
        retrieved_context: str,
        latency_ms: float,
    ) -> EvalResult:
        """
        Evaluasi satu jawaban menggunakan LLM-as-a-Judge.

        Args:
            question_item   : Item dari benchmark_dataset.json.
            llm_name        : Nama LLM yang menghasilkan jawaban.
            generated_answer: Jawaban yang dihasilkan LLM.
            retrieved_context: Context yang digunakan (KG triples / log entries).
            latency_ms      : Waktu respons LLM dalam milidetik.

        Returns:
            EvalResult: Hasil evaluasi lengkap.
        """
        result = EvalResult(
            question_id=question_item["id"],
            category=question_item["category"],
            question=question_item["question"],
            llm_name=llm_name,
            generated_answer=generated_answer,
            retrieved_context=retrieved_context[:2000],  # Truncate untuk judge
            latency_ms=latency_ms,
        )

        try:
            judge_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                question=question_item["question"],
                expected_answer=question_item["expected_answer"],
                generated_answer=generated_answer,
                retrieved_context=retrieved_context[:1500],
            )

            response = self._judge_client.chat.completions.create(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=512,
            )

            judge_output = json.loads(response.choices[0].message.content)

            result.faithfulness = float(judge_output.get("faithfulness", 0.0))
            result.answer_relevancy = float(judge_output.get("answer_relevancy", 0.0))
            result.context_precision = float(judge_output.get("context_precision", 0.0))
            result.judge_reasoning = judge_output.get("reasoning", "")

            logger.info(
                "question_evaluated",
                q_id=result.question_id,
                llm=llm_name,
                faithfulness=result.faithfulness,
                relevancy=result.answer_relevancy,
                precision=result.context_precision,
                overall=result.overall_score,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.error(
                "evaluation_failed",
                q_id=result.question_id,
                llm=llm_name,
                error=str(exc)
            )

        return result

    def evaluate_batch(
        self,
        llm_name: str,
        answer_generator_fn,
        category_filter: Optional[str] = None,
    ) -> List[EvalResult]:
        """
        Evaluasi batch seluruh benchmark untuk satu LLM.

        Args:
            llm_name            : Nama LLM (untuk labeling hasil).
            answer_generator_fn : Callable(question_str) -> (answer_str, context_str, latency_ms).
                                  Harus memanggil RAG pipeline yang sesuai.
            category_filter     : Filter kategori ("security_analysis", "log_analysis", "kg_qa").
                                  None = semua kategori.

        Returns:
            List[EvalResult]: Semua hasil evaluasi.
        """
        questions = self._benchmark
        if category_filter:
            questions = [q for q in questions if q["category"] == category_filter]

        logger.info(
            "batch_evaluation_start",
            llm=llm_name,
            total=len(questions),
            category_filter=category_filter or "all"
        )

        results = []
        for i, item in enumerate(questions):
            logger.info(
                "evaluating_question",
                progress=f"{i + 1}/{len(questions)}",
                q_id=item["id"],
                llm=llm_name
            )
            try:
                start_time = time.time()
                answer, context, _ = answer_generator_fn(item["question"])
                latency_ms = (time.time() - start_time) * 1000

            except Exception as exc:
                logger.error("answer_generation_failed", q_id=item["id"], error=str(exc))
                answer = ""
                context = ""
                latency_ms = 0.0

            result = self.evaluate_single(
                question_item=item,
                llm_name=llm_name,
                generated_answer=answer,
                retrieved_context=context,
                latency_ms=latency_ms,
            )
            results.append(result)

        logger.info(
            "batch_evaluation_complete",
            llm=llm_name,
            total_evaluated=len(results)
        )
        return results

    def compute_summary(self, results: List[EvalResult]) -> EvalSummary:
        """
        Hitung EvalSummary dari daftar EvalResult.

        Args:
            results: Daftar EvalResult dari evaluate_batch.

        Returns:
            EvalSummary: Statistik agregat per LLM.
        """
        if not results:
            raise ValueError("Results tidak boleh kosong.")

        llm_name = results[0].llm_name
        valid_results = [r for r in results if r.error is None]
        error_count = len(results) - len(valid_results)

        def safe_avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        # Agregat per kategori
        categories = set(r.category for r in valid_results)
        results_by_category = {}
        for cat in categories:
            cat_results = [r for r in valid_results if r.category == cat]
            results_by_category[cat] = {
                "avg_faithfulness": safe_avg([r.faithfulness for r in cat_results]),
                "avg_answer_relevancy": safe_avg([r.answer_relevancy for r in cat_results]),
                "avg_context_precision": safe_avg([r.context_precision for r in cat_results]),
                "avg_overall": safe_avg([r.overall_score for r in cat_results]),
                "count": len(cat_results),
            }

        return EvalSummary(
            llm_name=llm_name,
            total_questions=len(results),
            avg_faithfulness=safe_avg([r.faithfulness for r in valid_results]),
            avg_answer_relevancy=safe_avg([r.answer_relevancy for r in valid_results]),
            avg_context_precision=safe_avg([r.context_precision for r in valid_results]),
            avg_overall_score=safe_avg([r.overall_score for r in valid_results]),
            avg_latency_ms=safe_avg([r.latency_ms for r in valid_results]),
            error_count=error_count,
            results_by_category=results_by_category,
        )

    def save_results(
        self,
        all_results: Dict[str, List[EvalResult]],
        summaries: List[EvalSummary],
    ) -> str:
        """
        Simpan hasil evaluasi ke JSON dan CSV.

        Args:
            all_results: Dict {llm_name: [EvalResult]} untuk semua LLM.
            summaries  : List EvalSummary per LLM.

        Returns:
            str: Path file JSON yang disimpan.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self._results_dir, f"eval_results_{timestamp}.json")

        # Serialize ke JSON
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "judge_model": self._judge_model,
                "benchmark_size": len(self._benchmark),
                "llms_evaluated": list(all_results.keys()),
            },
            "summaries": [asdict(s) for s in summaries],
            "detailed_results": {
                llm: [asdict(r) for r in results]
                for llm, results in all_results.items()
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Simpan CSV summary untuk charting
        csv_path = output_path.replace(".json", "_summary.csv")
        summary_rows = []
        for llm, results in all_results.items():
            for r in results:
                summary_rows.append({
                    "llm": llm,
                    "question_id": r.question_id,
                    "category": r.category,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "overall_score": r.overall_score,
                    "latency_ms": r.latency_ms,
                    "error": r.error or "",
                })

        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

        logger.info(
            "results_saved",
            json_path=output_path,
            csv_path=csv_path,
            total_results=sum(len(v) for v in all_results.values())
        )

        return output_path
