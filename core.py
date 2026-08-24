"""
Core logic for the Document Summary Assistant: text extraction and
summarization. Kept separate from app.py (the Streamlit UI) so it can be
unit tested without spinning up a Streamlit session.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pypdf
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

logger = logging.getLogger("doc_summary_assistant")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 20
SUPPORTED_IMAGE_TYPES = ("png", "jpg", "jpeg", "webp", "bmp", "tiff")
SUPPORTED_TYPES = ("pdf",) + SUPPORTED_IMAGE_TYPES

# Chunking for map-reduce summarization. Roughly 4 chars/token, so this
# keeps each chunk comfortably within model context alongside the prompt.
CHUNK_CHAR_SIZE = 12000
CHUNK_OVERLAP = 200

LENGTH_INSTRUCTIONS = {
    "Short": "in 2-3 concise sentences",
    "Medium": "in a compact paragraph of about 120-180 words",
    "Long": "in 3-4 detailed paragraphs covering all major sections",
}


class ExtractionError(Exception):
    """Raised when text cannot be extracted from an uploaded file."""


class UnsupportedFileType(ExtractionError):
    pass


class FileTooLarge(ExtractionError):
    pass


class EmptyExtraction(ExtractionError):
    pass


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_file(filename: str, size_bytes: int) -> str:
    """Validate a file's extension and size. Returns the lowercase extension."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileType(
            f"Unsupported file type '.{suffix}'. Supported: {', '.join(SUPPORTED_TYPES)}"
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileTooLarge(
            f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    return suffix


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    text: str
    method: str  # "native_pdf", "ocr_pdf", or "ocr_image"
    pages: int = 1


def extract_text_from_pdf(
    file_bytes: bytes,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> ExtractionResult:
    """Try native text extraction first; fall back to OCR for scanned PDFs."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return ExtractionResult(text=text, method="native_pdf", pages=len(reader.pages))
    except Exception as e:
        logger.warning("Native PDF extraction failed, falling back to OCR: %s", e)

    # Fallback: rasterize pages and OCR them (scanned PDF)
    images = convert_from_bytes(file_bytes)
    text_chunks = []
    for i, img in enumerate(images, start=1):
        text_chunks.append(pytesseract.image_to_string(img))
        if progress_cb:
            progress_cb(i, len(images))
    return ExtractionResult(text="\n".join(text_chunks), method="ocr_pdf", pages=len(images))


def extract_text_from_image(file_bytes: bytes) -> ExtractionResult:
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return ExtractionResult(text=text, method="ocr_image", pages=1)


def extract_text(
    filename: str,
    file_bytes: bytes,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> ExtractionResult:
    suffix = validate_file(filename, len(file_bytes))

    if suffix == "pdf":
        result = extract_text_from_pdf(file_bytes, progress_cb=progress_cb)
    else:
        result = extract_text_from_image(file_bytes)

    if not result.text.strip():
        raise EmptyExtraction(
            "No text could be extracted. The file may be a low-quality scan, "
            "an empty document, or an unsupported layout."
        )
    return result


# --------------------------------------------------------------------------
# Chunking (for map-reduce summarization of long documents)
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_CHAR_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph/sentence
    boundaries where possible so chunks don't cut words or ideas in half."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer to break at a paragraph, then a sentence, then a space.
            for sep in ("\n\n", ". ", " "):
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = max(end - overlap, end) if end == n else end - overlap
    return chunks


# --------------------------------------------------------------------------
# Summarization (map-reduce for long docs, single-pass for short ones)
# --------------------------------------------------------------------------
def _call_with_retry(fn: Callable[[], str], attempts: int = 3, base_delay: float = 1.5) -> str:
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            logger.warning("API call failed (attempt %d/%d): %s", i + 1, attempts, e)
            if i < attempts - 1:
                time.sleep(base_delay * (2**i))
    raise RuntimeError(f"Summarization failed after {attempts} attempts: {last_err}") from last_err


def _generate(client, prompt: str, model: str = "gemini-2.5-flash") -> str:
    def call():
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    return _call_with_retry(call)


def summarize(client, text: str, length: str) -> str:
    """Summarize text. For long documents, summarize each chunk (map) then
    combine those partial summaries into one final summary (reduce), instead
    of naively truncating — this keeps the summary grounded in the whole
    document rather than just its first ~30k characters."""
    chunks = chunk_text(text)

    if len(chunks) == 1:
        prompt = f"""You are an assistant that writes clear, accurate summaries of documents.

Summarize the following document {LENGTH_INSTRUCTIONS[length]}.
After the summary, add a short "Key Points" bulleted list (3-6 bullets) of the
most important facts, decisions, or takeaways in the document.

Document:
\"\"\"
{chunks[0]}
\"\"\"
"""
        return _generate(client, prompt)

    # Map: summarize each chunk independently.
    partial_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = f"""Summarize part {i} of {len(chunks)} of a longer document in 4-6 sentences,
capturing all specific facts, names, numbers, and decisions. This is an
intermediate summary that will be combined with summaries of the other parts.

Part {i}/{len(chunks)}:
\"\"\"
{chunk}
\"\"\"
"""
        partial_summaries.append(_generate(client, prompt))

    # Reduce: combine partial summaries into one coherent final summary.
    combined = "\n\n".join(f"[Part {i+1}] {s}" for i, s in enumerate(partial_summaries))
    reduce_prompt = f"""Below are sequential partial summaries of one long document (in order).
Combine them into a single coherent summary {LENGTH_INSTRUCTIONS[length]}.
Then add a "Key Points" bulleted list (3-6 bullets) of the most important
facts, decisions, or takeaways across the whole document. Avoid repetition
and don't refer to "parts" in your output — write as one unified summary.

Partial summaries:
\"\"\"
{combined}
\"\"\"
"""
    return _generate(client, reduce_prompt)


def compression_stats(original_text: str, summary_text: str) -> dict:
    orig_words = len(original_text.split())
    summ_words = len(summary_text.split())
    reduction = 100 * (1 - summ_words / orig_words) if orig_words else 0
    return {"original_words": orig_words, "summary_words": summ_words, "reduction_pct": round(reduction, 1)}