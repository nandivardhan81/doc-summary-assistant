"""
Document Summary Assistant
---------------------------
Upload a PDF or an image (scanned document) and get a smart, AI-generated
summary — short, medium, or long — with key points highlighted.

Run locally:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key-here"
    streamlit run app.py
"""

import os
import time

import streamlit as st
from google import genai

from core import (
    ExtractionError,
    extract_text,
    summarize,
    compression_stats,
    MAX_FILE_SIZE_MB,
    SUPPORTED_TYPES,
)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Summary Assistant",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { max-width: 900px; margin: 0 auto; }
    .summary-box {
        background-color: #f5f7fa;
        border-left: 4px solid #4a6fa5;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    .stat-pill {
        display: inline-block;
        background-color: #eef2f7;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        margin-right: 0.5rem;
        font-size: 0.85rem;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Document Summary Assistant")
st.caption("Upload a PDF or an image of a document and get an instant, smart summary.")


# --------------------------------------------------------------------------
# Gemini client (cached across reruns within a session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error(
            "No Gemini API key found. Set the `GEMINI_API_KEY` environment variable "
            "(or add it to `.streamlit/secrets.toml` when deploying)."
        )
        st.stop()
    return genai.Client(api_key=api_key)


# --------------------------------------------------------------------------
# Cached extraction — re-running with the same file + no source change
# won't re-OCR or re-parse.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_extract(filename: str, file_bytes: bytes):
    return extract_text(filename, file_bytes)


@st.cache_data(show_spinner=False)
def cached_summarize(_client, text: str, length: str) -> str:
    # _client prefixed with underscore so Streamlit doesn't try to hash it.
    return summarize(_client, text, length)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop a file here, or click to browse",
    type=list(SUPPORTED_TYPES),
    help=f"PDF or image files, up to {MAX_FILE_SIZE_MB} MB.",
)

length = st.radio("Summary length", ["Short", "Medium", "Long"], horizontal=True, index=1)

if uploaded_file is not None:
    if st.button("Generate Summary", type="primary"):
        file_bytes = uploaded_file.getvalue()

        try:
            with st.spinner("Extracting text from your document (this can take a moment for scanned pages)..."):
                result = cached_extract(uploaded_file.name, file_bytes)

            method_labels = {
                "native_pdf": "Extracted directly from PDF text layer",
                "ocr_pdf": "Extracted via OCR (scanned PDF)",
                "ocr_image": "Extracted via OCR (image)",
            }
            st.caption(f"✓ {method_labels.get(result.method, result.method)} · {result.pages} page(s)")

            with st.expander("View extracted text"):
                preview = result.text[:5000] + ("..." if len(result.text) > 5000 else "")
                st.text(preview)

            with st.spinner("Generating summary..."):
                client = get_gemini_client()
                start = time.time()
                summary = cached_summarize(client, result.text, length)
                elapsed = time.time() - start

            st.markdown("### Summary")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

            stats = compression_stats(result.text, summary)
            st.markdown(
                f'<span class="stat-pill">⏱ {elapsed:.1f}s</span>'
                f'<span class="stat-pill">📝 {stats["original_words"]} → {stats["summary_words"]} words</span>'
                f'<span class="stat-pill">↓ {stats["reduction_pct"]}% shorter</span>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "Download summary (.md)",
                data=summary,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_summary.md",
                mime="text/markdown",
            )

        except ExtractionError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Something went wrong while processing this file: {e}")

else:
    st.info(f"Upload a PDF or image (up to {MAX_FILE_SIZE_MB} MB) to get started.")

st.divider()
st.caption(
    "Built with Streamlit, PyPDF/pdf2image + Tesseract OCR for extraction, "
    "and the Gemini API for summarization (with map-reduce chunking for long documents)."
)
