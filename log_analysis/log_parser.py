"""
SEPSES CSKG LLM Chatbot - Log Parser Module
============================================
Tanggung Jawab  : Satya Wira Pramudita (Evaluator & Log Dev)
Branch          : feature/eval-log-dev
Standar         : IEEE 830, ISO/IEC 12207

Deskripsi:
    Parser untuk berbagai format security log:
    - Snort IDS Alert Log
    - Syslog (RFC 5424)
    - Windows Event Log (XML/EVTX text export)
    - Apache/Nginx Access Log

    Output: List[LogEntry] yang siap dimasukkan ke ChromaDB.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import structlog

# ============================================================
# Structured Logging Setup
# ============================================================
logger = structlog.get_logger(__name__)


# ============================================================
# Data Classes
# ============================================================
class LogType(str, Enum):
    """Tipe log yang didukung oleh parser."""
    SNORT = "snort_alert"
    SYSLOG = "syslog"
    WINDOWS_EVENT = "windows_event"
    APACHE = "apache_access"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """
    Representasi terstruktur dari satu baris log yang telah diparsing.

    Attributes:
        raw_line    : Baris log asli sebelum diparsing.
        log_type    : Tipe log (enum LogType).
        timestamp   : Timestamp event log (ISO 8601).
        severity    : Level severity (critical/high/medium/low/info).
        source_ip   : IP address sumber (jika ada).
        dest_ip     : IP address tujuan (jika ada).
        message     : Pesan inti dari log entry.
        cve_refs    : List CVE ID yang direferensikan dalam log.
        extra       : Data tambahan spesifik per tipe log.
        doc_id      : ID unik untuk ChromaDB document.
    """
    raw_line: str
    log_type: LogType
    timestamp: str
    severity: str = "info"
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    message: str = ""
    cve_refs: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        """Generate doc_id dari hash raw_line jika belum diset."""
        if not self.doc_id:
            import hashlib
            self.doc_id = hashlib.md5(
                self.raw_line.encode("utf-8"), usedforsecurity=False
            ).hexdigest()

    def to_document_text(self) -> str:
        """
        Konversi LogEntry ke teks yang akan di-embed ke ChromaDB.

        Returns:
            str: Representasi teks terstruktur dari log entry.
        """
        parts = [
            f"[{self.log_type.value.upper()}]",
            f"Timestamp: {self.timestamp}",
            f"Severity: {self.severity}",
        ]
        if self.source_ip:
            parts.append(f"Source IP: {self.source_ip}")
        if self.dest_ip:
            parts.append(f"Destination IP: {self.dest_ip}")
        if self.cve_refs:
            parts.append(f"CVE References: {', '.join(self.cve_refs)}")
        parts.append(f"Message: {self.message}")
        return " | ".join(parts)

    def to_metadata(self) -> dict:
        """
        Return metadata dict untuk ChromaDB document.

        Returns:
            dict: Metadata yang akan disimpan bersama embedding.
        """
        return {
            "log_type": self.log_type.value,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "source_ip": self.source_ip or "",
            "dest_ip": self.dest_ip or "",
            "cve_refs": ",".join(self.cve_refs),
        }


# ============================================================
# Parser Implementation
# ============================================================
class LogParser:
    """
    Parser utama untuk berbagai format security log.

    Metode parse_file() mendeteksi format otomatis dan mendelegasikan
    ke parser yang sesuai.
    """

    # --- Regex Patterns ---
    SNORT_PATTERN = re.compile(
        r"(?P<timestamp>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+"
        r"\[\*\*\]\s+\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+"
        r"(?P<msg>.+?)\s+\[\*\*\]"
        r"(?:.*?\{(?P<proto>\w+)\}\s*(?P<src>[\d.]+)(?::\d+)?\s*->\s*(?P<dst>[\d.]+))?",
        re.IGNORECASE
    )

    SYSLOG_PATTERN = re.compile(
        r"(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+(?P<process>\S+):\s+(?P<message>.+)"
    )

    WIN_EVENT_PATTERN = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<level>\w+)\s+(?P<source>.+?)\s+(?P<event_id>\d+)\s+"
        r"(?P<message>.+)",
        re.IGNORECASE
    )

    APACHE_PATTERN = re.compile(
        r"(?P<src>[\d.]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+"
        r'"(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+'
        r"(?P<status>\d+)\s+(?P<size>\d+)"
    )

    CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

    # Severity mapping dari Snort priority / HTTP status
    SEVERITY_MAP = {
        "1": "critical", "2": "high", "3": "medium", "4": "low",
        "4xx": "medium", "5xx": "high",
        "CRITICAL": "critical", "ERROR": "high",
        "WARNING": "medium", "INFO": "info", "DEBUG": "info"
    }

    def parse_file(self, file_path: str) -> List[LogEntry]:
        """
        Baca file log, deteksi formatnya, dan parse semua baris.

        Args:
            file_path: Path absolut atau relatif ke file log.

        Returns:
            List[LogEntry]: Daftar log entry yang berhasil diparsing.

        Raises:
            FileNotFoundError: Jika file tidak ditemukan.
            ValueError: Jika format log tidak dikenali dan tidak ada baris valid.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("log_file_not_found", path=str(file_path))
            raise FileNotFoundError(f"Log file tidak ditemukan: {file_path}")

        logger.info("parsing_log_file", path=str(file_path), size_bytes=path.stat().st_size)

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            logger.error("log_file_read_error", path=str(file_path), error=str(exc))
            raise

        if not lines:
            logger.warning("log_file_empty", path=str(file_path))
            return []

        # Deteksi tipe log dari beberapa baris pertama
        log_type = self._detect_log_type(lines[:20])
        logger.info("log_type_detected", log_type=log_type.value, total_lines=len(lines))

        entries: List[LogEntry] = []
        failed_count = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = self._parse_line(line, log_type)
                if entry:
                    entries.append(entry)
            except Exception as exc:  # pylint: disable=broad-except
                failed_count += 1
                logger.debug(
                    "line_parse_failed",
                    line_number=i + 1,
                    error=str(exc),
                    raw_line=line[:100]
                )

        logger.info(
            "parsing_complete",
            total_lines=len(lines),
            parsed=len(entries),
            failed=failed_count
        )
        return entries

    def parse_text(self, text: str, log_type: LogType = LogType.UNKNOWN) -> List[LogEntry]:
        """
        Parse log dari string teks langsung (berguna untuk input dari UI).

        Args:
            text    : Konten log sebagai string.
            log_type: Tipe log (opsional, akan dideteksi otomatis jika UNKNOWN).

        Returns:
            List[LogEntry]: Daftar log entry.
        """
        lines = text.strip().splitlines()
        if log_type == LogType.UNKNOWN:
            log_type = self._detect_log_type(lines[:20])

        entries = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = self._parse_line(line, log_type)
                if entry:
                    entries.append(entry)
            except Exception:  # pylint: disable=broad-except
                continue
        return entries

    # ---- Private Methods ----

    def _detect_log_type(self, sample_lines: List[str]) -> LogType:
        """
        Deteksi tipe log berdasarkan pola di beberapa baris awal.

        Args:
            sample_lines: Beberapa baris pertama file log.

        Returns:
            LogType: Tipe log yang terdeteksi.
        """
        sample = "\n".join(sample_lines)
        if "[**]" in sample and "->" in sample:
            return LogType.SNORT
        if re.search(r"\d{4}-\d{2}-\d{2}.*EventID", sample, re.IGNORECASE):
            return LogType.WINDOWS_EVENT
        if re.search(r'"\w+ /\S+ HTTP/', sample):
            return LogType.APACHE
        if re.search(r"\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+:", sample):
            return LogType.SYSLOG
        return LogType.UNKNOWN

    def _parse_line(self, line: str, log_type: LogType) -> Optional[LogEntry]:
        """
        Dispatch parsing ke handler yang sesuai berdasarkan log_type.

        Args:
            line    : Satu baris log.
            log_type: Tipe log yang telah dideteksi.

        Returns:
            Optional[LogEntry]: LogEntry yang diparsing, atau None jika baris invalid.
        """
        parsers = {
            LogType.SNORT: self._parse_snort_line,
            LogType.SYSLOG: self._parse_syslog_line,
            LogType.WINDOWS_EVENT: self._parse_windows_event_line,
            LogType.APACHE: self._parse_apache_line,
            LogType.UNKNOWN: self._parse_generic_line,
        }
        parser_fn = parsers.get(log_type, self._parse_generic_line)
        return parser_fn(line)

    def _extract_cve_refs(self, text: str) -> List[str]:
        """Ekstrak semua CVE ID yang ditemukan dalam teks."""
        return list(set(self.CVE_PATTERN.findall(text)))

    def _parse_snort_line(self, line: str) -> Optional[LogEntry]:
        """Parse satu baris Snort IDS alert."""
        match = self.SNORT_PATTERN.search(line)
        if not match:
            return None

        groups = match.groupdict()
        # Konversi timestamp Snort (MM/DD-HH:MM:SS) ke ISO approximate
        raw_ts = groups.get("timestamp", "")
        timestamp = f"2024-{raw_ts}" if raw_ts else datetime.utcnow().isoformat()

        cve_refs = self._extract_cve_refs(line)

        return LogEntry(
            raw_line=line,
            log_type=LogType.SNORT,
            timestamp=timestamp,
            severity="high",  # Snort alerts selalu perlu perhatian
            source_ip=groups.get("src"),
            dest_ip=groups.get("dst"),
            message=groups.get("msg", line),
            cve_refs=cve_refs,
            extra={
                "gid": groups.get("gid"),
                "sid": groups.get("sid"),
                "rev": groups.get("rev"),
                "proto": groups.get("proto"),
            }
        )

    def _parse_syslog_line(self, line: str) -> Optional[LogEntry]:
        """Parse satu baris Syslog (RFC 5424)."""
        match = self.SYSLOG_PATTERN.search(line)
        if not match:
            return None

        groups = match.groupdict()
        year = datetime.utcnow().year
        timestamp = f"{year} {groups['month']} {groups['day']} {groups['time']}"
        message = groups.get("message", line)
        cve_refs = self._extract_cve_refs(line)

        # Heuristik severity dari konten pesan
        severity = "info"
        msg_upper = message.upper()
        for keyword, sev in [("CRITICAL", "critical"), ("ERROR", "high"),
                              ("FAIL", "high"), ("WARN", "medium"),
                              ("DENIED", "medium"), ("ATTACK", "high")]:
            if keyword in msg_upper:
                severity = sev
                break

        return LogEntry(
            raw_line=line,
            log_type=LogType.SYSLOG,
            timestamp=timestamp,
            severity=severity,
            source_ip=None,
            dest_ip=None,
            message=message,
            cve_refs=cve_refs,
            extra={
                "host": groups.get("host"),
                "process": groups.get("process"),
            }
        )

    def _parse_windows_event_line(self, line: str) -> Optional[LogEntry]:
        """Parse satu baris Windows Event Log (text export)."""
        match = self.WIN_EVENT_PATTERN.search(line)
        if not match:
            return None

        groups = match.groupdict()
        level = groups.get("level", "info").upper()
        severity = self.SEVERITY_MAP.get(level, "info")
        cve_refs = self._extract_cve_refs(line)

        return LogEntry(
            raw_line=line,
            log_type=LogType.WINDOWS_EVENT,
            timestamp=groups.get("timestamp", ""),
            severity=severity,
            message=groups.get("message", line),
            cve_refs=cve_refs,
            extra={
                "event_id": groups.get("event_id"),
                "source": groups.get("source"),
                "level": level,
            }
        )

    def _parse_apache_line(self, line: str) -> Optional[LogEntry]:
        """Parse satu baris Apache/Nginx access log."""
        match = self.APACHE_PATTERN.search(line)
        if not match:
            return None

        groups = match.groupdict()
        status = groups.get("status", "200")
        severity = "high" if status.startswith("5") else \
                   "medium" if status.startswith("4") else "info"

        return LogEntry(
            raw_line=line,
            log_type=LogType.APACHE,
            timestamp=groups.get("timestamp", ""),
            severity=severity,
            source_ip=groups.get("src"),
            dest_ip=None,
            message=f"HTTP {groups.get('method')} {groups.get('path')} -> {status}",
            cve_refs=[],
            extra={
                "method": groups.get("method"),
                "path": groups.get("path"),
                "status": status,
                "size": groups.get("size"),
            }
        )

    def _parse_generic_line(self, line: str) -> Optional[LogEntry]:
        """Fallback parser untuk log dengan format tidak dikenal."""
        if not line:
            return None
        cve_refs = self._extract_cve_refs(line)
        return LogEntry(
            raw_line=line,
            log_type=LogType.UNKNOWN,
            timestamp=datetime.utcnow().isoformat(),
            severity="info",
            message=line,
            cve_refs=cve_refs,
        )
