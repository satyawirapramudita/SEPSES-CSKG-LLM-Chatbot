"""
SEPSES CSKG LLM Chatbot - LLM Connector
==========================================
Tanggung Jawab  : Fahmi Abdillah Zain (RAG Logic Dev)
Branch          : feature/rag-logic

Deskripsi:
    Abstraksi koneksi ke dua LLM backend:
    1. GPT-4o-mini via OpenAI API
    2. Mistral-7B via Ollama (local)

    Menggunakan interface yang seragam: generate(messages) → str
    Mendukung streaming dan non-streaming mode.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)


# ============================================================
# Message Type
# ============================================================
class Message:
    """Representasi satu pesan dalam conversation."""

    ROLES = {"system", "user", "assistant"}

    def __init__(self, role: str, content: str) -> None:
        """
        Args:
            role   : "system" | "user" | "assistant"
            content: Teks pesan.

        Raises:
            ValueError: Jika role tidak valid.
        """
        if role not in self.ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {self.ROLES}")
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        """Convert ke dict format yang kompatibel dengan OpenAI API."""
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        return f"Message(role={self.role}, content={self.content[:50]!r})"


# ============================================================
# Abstract Base Class
# ============================================================
class BaseLLMConnector(ABC):
    """Abstract interface untuk semua LLM connectors."""

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate respons dari LLM.

        Args:
            messages   : List of Message objects (system + user + history).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens : Maksimum panjang output.

        Returns:
            str: Respons teks dari LLM.

        Raises:
            RuntimeError: Jika LLM tidak tersedia atau terjadi error.
        """
        ...

    @abstractmethod
    def ping(self) -> bool:
        """
        Cek apakah LLM backend tersedia.

        Returns:
            bool: True jika tersedia.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nama model yang digunakan."""
        ...


# ============================================================
# GPT-4o-mini Connector (OpenAI)
# ============================================================
class OpenAIConnector(BaseLLMConnector):
    """
    Connector ke OpenAI GPT-4o-mini via official Python SDK.

    Environment variables yang diperlukan:
        OPENAI_API_KEY : API key (wajib)
        OPENAI_MODEL   : Model name (default: gpt-4o-mini)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Args:
            api_key: OpenAI API key. Default dari env var OPENAI_API_KEY.
            model  : Model identifier. Default dari env var OPENAI_MODEL.

        Raises:
            RuntimeError: Jika openai package tidak terinstall atau API key kosong.
        """
        try:
            from openai import OpenAI  # lazy import
            self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not self._api_key or self._api_key.startswith("sk-GANTI"):
                raise RuntimeError(
                    "OPENAI_API_KEY belum diset! Isi file .env dengan API key yang valid."
                )
            self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._client = OpenAI(api_key=self._api_key)
            logger.info("openai_connector_init", model=self._model)
        except ImportError as exc:
            raise RuntimeError(
                "Package 'openai' belum terinstall. Jalankan: pip install openai"
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response dari GPT-4o-mini."""
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            latency_ms = round((time.time() - start) * 1000)
            logger.info(
                "openai_generate_success",
                model=self._model,
                latency_ms=latency_ms,
                output_tokens=response.usage.completion_tokens if response.usage else None,
            )
            return content

        except Exception as exc:
            logger.error("openai_generate_error", error=str(exc))
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    def generate_with_latency(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> tuple[str, float]:
        """
        Generate dengan return latency dalam ms.

        Returns:
            Tuple[str, float]: (answer, latency_ms)
        """
        start = time.time()
        answer = self.generate(messages, temperature, max_tokens)
        return answer, round((time.time() - start) * 1000, 2)

    def ping(self) -> bool:
        """Test koneksi dengan request minimal."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


# ============================================================
# Mistral-7B Connector (Ollama)
# ============================================================
class OllamaConnector(BaseLLMConnector):
    """
    Connector ke Mistral-7B (atau model Ollama lain) via Ollama REST API.

    Environment variables yang diperlukan:
        OLLAMA_BASE_URL : URL Ollama server (default: http://localhost:11434)
        OLLAMA_MODEL    : Model name (default: mistral)
    """

    OLLAMA_GENERATE_PATH = "/api/chat"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Args:
            base_url: URL Ollama server.
            model   : Model name (harus sudah di-pull via `ollama pull <model>`).
        """
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Package 'requests' belum terinstall.") from exc

        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._model    = model or os.getenv("OLLAMA_MODEL", "mistral")
        self._api_url  = f"{self._base_url}{self.OLLAMA_GENERATE_PATH}"
        logger.info("ollama_connector_init", model=self._model, url=self._api_url)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response dari Mistral via Ollama."""
        import requests

        payload = {
            "model":    self._model,
            "messages": [m.to_dict() for m in messages],
            "stream":   False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.time()
        try:
            resp = requests.post(
                self._api_url,
                json=payload,
                timeout=120,  # Mistral bisa lebih lambat
            )
            resp.raise_for_status()
            data = resp.json()

            # Ollama mengembalikan {"message": {"role": "assistant", "content": "..."}}
            content = data.get("message", {}).get("content", "")
            latency_ms = round((time.time() - start) * 1000)
            logger.info(
                "ollama_generate_success",
                model=self._model,
                latency_ms=latency_ms,
            )
            return content

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama server tidak berjalan di {self._base_url}. "
                f"Jalankan: ollama serve"
            ) from exc
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama timeout setelah 120 detik. Model mungkin belum di-load.")
        except Exception as exc:
            logger.error("ollama_generate_error", error=str(exc))
            raise RuntimeError(f"Ollama error: {exc}") from exc

    def generate_with_latency(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> tuple[str, float]:
        """
        Generate dengan return latency dalam ms.

        Returns:
            Tuple[str, float]: (answer, latency_ms)
        """
        start = time.time()
        answer = self.generate(messages, temperature, max_tokens)
        return answer, round((time.time() - start) * 1000, 2)

    def ping(self) -> bool:
        """Cek apakah Ollama server aktif."""
        import requests
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List semua model yang tersedia di Ollama.

        Returns:
            List[str]: Nama-nama model yang sudah di-pull.
        """
        import requests
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ============================================================
# Google Gemini Connector
# ============================================================
class GeminiConnector(BaseLLMConnector):
    """
    LLM connector untuk Google Gemini via google-generativeai SDK.

    Model yang didukung:
    - gemini-2.0-flash (default, cepat, gratis tier)
    - gemini-1.5-flash
    - gemini-1.5-pro

    Catatan:
    - Gemini tidak punya peran "system" terpisah; system prompt
      digabung ke pesan user pertama sebagai [System Instructions].
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            model  : Nama model Gemini. Default dari GEMINI_MODEL env.
            api_key: API key Google AI Studio. Default dari GEMINI_API_KEY env.

        Raises:
            ImportError: Jika google-generativeai belum terinstall.
            ValueError : Jika API key belum diset.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Package 'google-generativeai' belum terinstall. "
                "Jalankan: pip install google-generativeai"
            )

        self._model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        key = api_key or os.getenv("GEMINI_API_KEY", "")

        if not key or "GANTI" in key:
            raise ValueError(
                "GEMINI_API_KEY belum diset. "
                "Dapatkan gratis di: https://aistudio.google.com/app/apikey\n"
                "Kemudian isi di .env: GEMINI_API_KEY=AIza..."
            )

        genai.configure(api_key=key)
        self._genai = genai
        self._client = genai.GenerativeModel(self._model_name)
        logger.info("gemini_connector_init", model=self._model_name)

    @property
    def model_name(self) -> str:
        """Nama model Gemini yang digunakan."""
        return self._model_name

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate response dari Gemini.

        Args:
            messages   : List Message (system/user/assistant).
            temperature: Kreativitas (0.0 = deterministik).
            max_tokens : Batas token output.

        Returns:
            str: Teks respons Gemini.

        Raises:
            RuntimeError: Jika API call gagal.
        """
        try:
            # Pisahkan system prompt dan history
            system_content = ""
            history = []
            user_msgs = []

            for msg in messages:
                if msg.role == "system":
                    system_content = msg.content
                elif msg.role == "user":
                    user_msgs.append(msg.content)
                elif msg.role == "assistant":
                    if user_msgs:
                        history.append({"role": "user", "parts": ["\n".join(user_msgs)]})
                        user_msgs = []
                    history.append({"role": "model", "parts": [msg.content]})

            # Bangun pesan user akhir (system + user)
            final_parts = []
            if system_content:
                final_parts.append(f"[System Instructions]\n{system_content}\n\n")
            if user_msgs:
                final_parts.append("\n".join(user_msgs))
            elif history and history[-1]["role"] == "user":
                final_parts = history.pop()["parts"]

            chat = self._client.start_chat(history=history)
            response = chat.send_message(
                "".join(final_parts),
                generation_config=self._genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            answer = response.text.strip()
            logger.info("gemini_generate_success", model=self._model_name, chars=len(answer))
            return answer

        except Exception as exc:
            logger.error("gemini_generate_error", error=str(exc))
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    def generate_with_latency(
        self,
        messages: List[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """Generate dengan pengukuran latency. Returns (answer, latency_ms)."""
        start = time.time()
        answer = self.generate(messages, temperature, max_tokens)
        return answer, round((time.time() - start) * 1000, 2)

    def ping(self) -> bool:
        """Verifikasi API key valid dengan generate singkat."""
        try:
            resp = self._client.generate_content("Hi")
            return bool(resp.text)
        except Exception:
            return False


# ============================================================
# Factory Function
# ============================================================
def get_llm_connector(llm_name: str) -> BaseLLMConnector:
    """
    Factory: return LLM connector berdasarkan nama model.

    Args:
        llm_name: Nama model.
                  - Gemini  : "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"
                  - OpenAI  : "gpt-4o-mini", "gpt-4o"
                  - Ollama  : "mistral", "llama3", "gemma4", dsb.

    Returns:
        BaseLLMConnector: Gemini / OpenAI / Ollama connector.

    Raises:
        RuntimeError: Jika backend tidak tersedia.
    """
    llm_lower = llm_name.lower()

    if "gemini" in llm_lower or "google" in llm_lower:
        logger.info("using_gemini_connector", llm=llm_name)
        return GeminiConnector(model=llm_name)
    elif "gpt" in llm_lower or "openai" in llm_lower:
        logger.info("using_openai_connector", llm=llm_name)
        return OpenAIConnector(model=llm_name)
    else:
        logger.info("using_ollama_connector", llm=llm_name)
        return OllamaConnector(model=llm_name)
