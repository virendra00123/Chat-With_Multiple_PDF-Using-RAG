import pdfplumber
import re

def extract_text(pdf_path):
    """
    Extract all text from the PDF at pdf_path.
    Returns a single string with all page texts concatenated.
    """
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

def chunk_text(text, chunk_size=120, overlap=30, filename=None):
    """
    Split text into chunks of chunk_size words, with overlap between chunks.
    
    Args:
        text (str): Input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words overlapping between chunks.
        filename (str): Name of the source file, for metadata.
        
    Returns:
        chunks (list of str): List of text chunks.
        metadata (list of dict): Metadata for each chunk (e.g., source, chunk index).
    """
    # Clean multiple newlines and extra spaces
    cleaned_text = re.sub(r'\n+', '\n', text.strip())
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    words = cleaned_text.split(' ')
    
    chunks = []
    metadata = []

    start = 0
    chunk_index = 0
    text_len = len(words)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_words = words[start:end]
        chunk_text_str = ' '.join(chunk_words)
        chunks.append(chunk_text_str)
        metadata.append({
            "source": filename if filename else "unknown",
            "chunk_index": chunk_index,
            "start_word": start,
            "end_word": end
        })
        chunk_index += 1
        start += chunk_size - overlap  # move start forward with overlap

    return chunks, metadata
