"""
Unit tests for core.py.

Run with:  pytest tests/
"""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image, ImageDraw
from unittest.mock import MagicMock

from core import (
    validate_file,
    extract_text_from_image,
    extract_text,
    chunk_text,
    summarize,
    compression_stats,
    UnsupportedFileType,
    FileTooLarge,
    EmptyExtraction,
    MAX_FILE_SIZE_MB,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
def make_text_image(text: str) -> bytes:
    img = Image.new("RGB", (800, 120), color="white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_blank_image() -> bytes:
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeGeminiClient:
    """Stub that mimics the google-genai client's response shape."""

    def __init__(self, response_text="This is a fake summary.\n\nKey Points:\n- point one"):
        self._response_text = response_text
        self.call_count = 0
        self.models = self  # so client.models.generate_content(...) works

    def generate_content(self, model, contents):
        self.call_count += 1
        resp = MagicMock()
        resp.text = self._response_text
        return resp


# --------------------------------------------------------------------------
# validate_file
# --------------------------------------------------------------------------
def test_validate_file_accepts_supported_types():
    assert validate_file("report.pdf", 1000) == "pdf"
    assert validate_file("scan.PNG", 1000) == "png"


def test_validate_file_rejects_unsupported_type():
    with pytest.raises(UnsupportedFileType):
        validate_file("archive.zip", 1000)


def test_validate_file_rejects_oversized_file():
    too_big = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024
    with pytest.raises(FileTooLarge):
        validate_file("big.pdf", too_big)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def test_extract_text_from_image_returns_result():
    img_bytes = make_text_image("Hello world")
    result = extract_text_from_image(img_bytes)
    assert result.method == "ocr_image"
    assert result.pages == 1
    assert len(result.text.strip()) > 0


def test_extract_text_raises_on_empty_extraction():
    blank_bytes = make_blank_image()
    with pytest.raises(EmptyExtraction):
        extract_text("blank.png", blank_bytes)


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileType):
        extract_text("notes.docx", b"irrelevant")


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------
def test_chunk_text_single_chunk_when_short():
    text = "short document"
    chunks = chunk_text(text, chunk_size=1000)
    assert chunks == [text]


def test_chunk_text_splits_long_text():
    text = ("Sentence one. " * 2000)  # well over default chunk size
    chunks = chunk_text(text, chunk_size=5000, overlap=100)
    assert len(chunks) > 1
    # No chunk should exceed the requested size by more than a small margin
    assert all(len(c) <= 5200 for c in chunks)
    # Reassembling (ignoring overlap) should still contain the original content
    assert "Sentence one." in chunks[0]


def test_chunk_text_preserves_all_content_roughly():
    text = "A" * 50000
    chunks = chunk_text(text, chunk_size=10000, overlap=0)
    assert sum(len(c) for c in chunks) >= len(text) * 0.95


# --------------------------------------------------------------------------
# summarization
# --------------------------------------------------------------------------
def test_summarize_single_chunk_calls_model_once():
    client = FakeGeminiClient()
    summary = summarize(client, "short document text", "Short")
    assert summary == client._response_text
    assert client.call_count == 1


def test_summarize_long_document_uses_map_reduce():
    client = FakeGeminiClient()
    long_text = "Sentence about topic X. " * 3000  # forces multiple chunks
    summary = summarize(client, long_text, "Medium")
    # map (N chunks) + 1 reduce call
    assert client.call_count > 1
    assert summary == client._response_text


def test_summarize_retries_on_failure_then_succeeds():
    client = FakeGeminiClient()
    calls = {"n": 0}
    original = client.generate_content

    def flaky(model, contents):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient API error")
        return original(model, contents)

    client.generate_content = flaky
    summary = summarize(client, "short text", "Short")
    assert calls["n"] == 2  # failed once, succeeded on retry
    assert summary == client._response_text


# --------------------------------------------------------------------------
# compression stats
# --------------------------------------------------------------------------
def test_compression_stats_basic():
    original = "word " * 100
    summary = "word " * 20
    stats = compression_stats(original, summary)
    assert stats["original_words"] == 100
    assert stats["summary_words"] == 20
    assert stats["reduction_pct"] == 80.0


def test_compression_stats_handles_empty_original():
    stats = compression_stats("", "")
    assert stats["reduction_pct"] == 0
