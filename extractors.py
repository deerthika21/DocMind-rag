"""
Turn uploaded files into text segments tagged with page + section,
so every chunk we index can be cited back to where it came from.

Supported: PDF (with OCR fallback for scanned pages), DOCX, TXT/MD/CSV,
and images (PNG/JPG/...) via OCR.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx"} | TEXT_EXTENSIONS | IMAGE_EXTENSIONS

# A PDF page with less text than this is treated as scanned and OCR'd
MIN_PAGE_TEXT = 20

# =========================================================
# OCR
# =========================================================

_ocr_engine = None


def _get_ocr():

    global _ocr_engine

    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        logger.info("[OCR] Engine loaded")

    return _ocr_engine


def ocr_image_bytes(data: bytes) -> str:
    """OCR an image and return its text in reading order (top-down, left-right)."""

    import numpy as np
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return ""

    result, _ = _get_ocr()(img)

    if not result:
        return ""

    # result rows: [box(4 points), text, score]
    boxes = []
    for box, text, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append({
            "text": text,
            "x": min(xs),
            "cy": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
        })

    boxes.sort(key=lambda b: b["cy"])

    # Group boxes whose vertical centres are close into the same line
    lines = []
    for b in boxes:
        if lines and abs(b["cy"] - lines[-1][0]["cy"]) < 0.5 * max(b["h"], 1):
            lines[-1].append(b)
        else:
            lines.append([b])

    text = "\n".join(
        " ".join(b["text"] for b in sorted(line, key=lambda b: b["x"]))
        for line in lines
    )

    # OCR often drops spaces: "DateOfBirth:21.06.2007" -> "Date Of Birth: 21.06.2007"
    text = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r":(?=\S)", ": ", text)

    return text

# =========================================================
# SECTION DETECTION
# =========================================================

_NUMBERED_HEADING = re.compile(
    r"^((\d+(\.\d+)*\.?)|([IVXLC]+\.)|((chapter|section|part|article|clause|annex|appendix)\s+[\w.]+))\s+\S",
    re.IGNORECASE,
)
_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def looks_like_heading(line: str) -> bool:
    """Heuristic heading detector for plain text extracted from PDFs/OCR."""

    s = line.strip()

    if not (3 <= len(s) <= 80):
        return False

    words = s.split()

    if len(words) > 10:
        return False

    if _NUMBERED_HEADING.match(s):
        return not s.endswith((",", ";"))

    if s.endswith((".", ",", ";", "?")):
        return False

    # Table rows / receipts ("TAMIL 010 038 048") are not headings
    if sum(c.isdigit() for c in s) > 0.2 * len(s):
        return False

    if s.isupper() and sum(c.isalpha() for c in s) >= 3:
        return True

    if s.endswith(":") and len(words) <= 6:
        return True

    alpha_words = [w for w in words if w[0].isalpha()]
    if len(alpha_words) >= 2:
        capitalised = sum(w[0].isupper() for w in alpha_words)
        return capitalised / len(alpha_words) >= 0.8

    return False


def _segment(lines, page, section, markdown=False):
    """
    Join lines into one segment, recording where each heading starts.
    Headings label chunks rather than split them, so short headed blocks
    don't turn into tiny, hard-to-retrieve chunks.
    Returns (segment, last_section) so sections carry across pages.
    """

    headings = []
    parts = []
    offset = 0

    for raw in lines:

        line = raw.rstrip()
        md = _MD_HEADING.match(line.strip()) if markdown else None

        if md or (not markdown and looks_like_heading(line)):
            title = (md.group(1) if md else line).strip().rstrip(":")[:80]
            headings.append((offset, title))

        parts.append(line)
        offset += len(line) + 1

    segment = {
        "text": "\n".join(parts),
        "page": page,
        "section": section,
        "headings": headings,
    }

    return segment, (headings[-1][1] if headings else section)


def chunk_segments(segments, chunk_size, chunk_overlap):
    """Split segments into chunks; each chunk gets the section it starts in."""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for seg in segments:

        if not seg["text"].strip():
            continue

        for doc in splitter.create_documents([seg["text"]]):

            start = doc.metadata.get("start_index", 0)
            section = seg["section"]

            # Last heading at or just after the chunk start (a chunk that
            # opens with a heading belongs to that heading)
            for h_offset, title in seg.get("headings", []):
                if h_offset <= start + 40:
                    section = title
                else:
                    break

            chunks.append({
                "text": doc.page_content,
                "page": seg["page"],
                "section": section,
            })

    return chunks

# =========================================================
# FORMAT LOADERS
# =========================================================

def _extract_pdf(path):

    from pypdf import PdfReader

    reader = PdfReader(path)
    segments = []
    section = ""
    ocr_pages = 0

    for page_no, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"[Extract] {path} p{page_no}: text extraction failed: {e}")
            text = ""

        # Scanned page -> OCR the embedded images
        if len(text.strip()) < MIN_PAGE_TEXT:
            try:
                ocr_text = "\n".join(
                    ocr_image_bytes(img.data) for img in page.images
                ).strip()
            except Exception as e:
                logger.warning(f"[Extract] {path} p{page_no}: OCR failed: {e}")
                ocr_text = ""

            if ocr_text:
                text = ocr_text
                ocr_pages += 1

        segment, section = _segment(text.splitlines(), page_no, section)
        segments.append(segment)

    return segments, {"pages": len(reader.pages), "ocr_pages": ocr_pages}


def _extract_docx(path):

    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    buf = []
    headings = []
    offset = 0

    def add(line):
        nonlocal offset
        buf.append(line)
        offset += len(line) + 1

    # Walk paragraphs and tables in document order
    for child in document.element.body.iterchildren():

        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            style = (para.style.name if para.style is not None else "") or ""

            if not text:
                continue

            # Use Word heading styles; fall back to the text heuristic
            if style.startswith(("Heading", "Title")) or looks_like_heading(text):
                headings.append((offset, text[:80]))

            add(text)

        elif tag == "tbl":
            table = Table(child, document)
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    t = cell.text.strip()
                    # merged cells repeat; skip consecutive duplicates
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    add(" | ".join(cells))

    segment = {"text": "\n".join(buf), "page": 0, "section": "", "headings": headings}

    return [segment], {"pages": 0, "ocr_pages": 0}


def _extract_text(path, ext):

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    if ext == ".csv":
        # No headings in CSV; keep the header row with every block of rows
        header, rows = (lines[0], lines[1:]) if lines else ("", [])
        segments = [
            {"text": "\n".join([header] + rows[i:i + 40]), "page": 0, "section": f"rows {i + 1}-{min(i + 40, len(rows))}"}
            for i in range(0, max(len(rows), 1), 40)
        ]
        return segments, {"pages": 0, "ocr_pages": 0}

    segment, _ = _segment(lines, 0, "", markdown=(ext == ".md"))

    return [segment], {"pages": 0, "ocr_pages": 0}


def _extract_image(path):

    with open(path, "rb") as f:
        text = ocr_image_bytes(f.read())

    segment, _ = _segment(text.splitlines(), 1, "")

    return [segment], {"pages": 1, "ocr_pages": 1 if text.strip() else 0}

# =========================================================
# ENTRY POINT
# =========================================================

def extract(path: str):
    """
    Returns (segments, info); pass segments to chunk_segments().
    segments: [{"text", "page" (1-based, 0 = n/a), "section", "headings"}]
    info: {"pages", "ocr_pages", "type"}
    """

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        segments, info = _extract_pdf(path)
    elif ext == ".docx":
        segments, info = _extract_docx(path)
    elif ext in TEXT_EXTENSIONS:
        segments, info = _extract_text(path, ext)
    elif ext in IMAGE_EXTENSIONS:
        segments, info = _extract_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    info["type"] = ext.lstrip(".")

    return segments, info
