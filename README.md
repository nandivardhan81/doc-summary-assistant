# Document Summary Assistant

An app that takes any document (PDF or scanned image) and generates a smart,
AI-written summary — with adjustable length, key-point highlights, and a
map-reduce pipeline so summaries stay accurate on long documents instead of
just being truncated.

## Features

- **Upload** PDF or image files (drag-and-drop or file picker), up to 20 MB
- **Text extraction**
  - PDFs: native text extraction (`pypdf`), with automatic OCR fallback for
    scanned/image-only PDFs
  - Images: OCR via Tesseract (`pytesseract`)
  - Extraction method used (native vs. OCR) is shown to the user
- **Summarization**: short / medium / long summaries plus a bulleted key-points
  list, via the Gemini API
  - **Map-reduce for long documents**: text is chunked on paragraph/sentence
    boundaries, each chunk is summarized independently, then the partial
    summaries are combined into one coherent final summary — so nothing past
    the first N characters gets silently dropped
  - **Automatic retries with backoff** on transient API failures
- **Result tools**: word-count reduction stats, download the summary as
  Markdown
- Input validation (file type, size), extraction caching within a session,
  and clear inline errors for bad files or API failures
- Mobile-responsive layout (Streamlit's default responsive container)

## Project structure