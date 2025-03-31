"""
PDF processing utilities for the Simple Local RAG project
"""
import os
import requests
import fitz  # PyMuPDF
from tqdm.auto import tqdm
from typing import List, Dict, Any

def download_pdf(url: str, filename: str) -> bool:
    """
    Download a PDF file from a URL if it doesn't already exist.

    Args:
        url: The URL of the PDF to download
        filename: The local filename to save the downloaded file

    Returns:
        bool: True if download was successful or file exists, False otherwise
    """
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return True

    print(f"File doesn't exist, downloading from {url}...")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {filename}")
        return True
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return False

def text_formatter(text: str) -> str:
    """
    Performs minor formatting on text from PDF.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        str: Formatted text
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str, page_offset: int = 0) -> List[Dict[str, Any]]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Args:
        pdf_path: The file path to the PDF document to be opened and read
        page_offset: Number to offset page numbers by (for correct references)

    Returns:
        List[Dict]: A list of dictionaries, each containing the page number
        (adjusted by offset), character count, word count, sentence count, 
        token count, and the extracted text for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        
        pages_and_texts.append({
            "page_number": page_number - page_offset,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,  # 1 token ~= 4 chars
            "text": text
        })
    
    return pages_and_texts 