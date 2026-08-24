# Approach

I split the app into `core.py` (extraction + summarization, framework-free)
and `app.py` (Streamlit UI), so the logic can be unit tested without a live
API key or a running app — 14 tests cover file validation, OCR extraction,
chunking edge cases, map-reduce summarization, and retry behavior, all
mocked and offline.

**Extraction:** PDFs go through `pypdf` first for native text; if that
returns nothing (a scanned PDF), pages are rasterized via `pdf2image` and
OCR'd with Tesseract. Images go straight through Tesseract. The method used
is surfaced to the user.

**Summarization:** rather than truncating long documents, text is chunked on
paragraph/sentence boundaries, each chunk is summarized independently (map),
then the partial summaries are merged into one coherent summary with a
key-points list (reduce) — so long documents don't lose content past an
arbitrary cutoff. API calls retry with exponential backoff on transient
failures.

**UX:** drag-and-drop upload, extraction/generation progress, a compression
stat, and a Markdown download of the result. Errors (bad file type, oversize
file, empty extraction, API failure) surface as specific inline messages.

**Hosting:** Streamlit Community Cloud — free, with `packages.txt` handling
the Tesseract/Poppler system dependencies.