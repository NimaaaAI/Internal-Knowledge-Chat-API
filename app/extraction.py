import fitz  # PyMuPDF


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from a PDF byte payload, page by page."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages).strip()
