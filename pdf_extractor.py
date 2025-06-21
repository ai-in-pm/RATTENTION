#!/usr/bin/env python3
"""
PDF Content Extractor for RATTENTION Paper Analysis
Extracts text content from the RATTENTION Paper.pdf for analysis
"""

import sys
import os

# Try different PDF libraries
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        print("Error: No PDF library available. Please install PyPDF2 or pdfplumber.")
        sys.exit(1)

def extract_text_pypdf2(pdf_path):
    """Extract text using PyPDF2"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- PAGE {page_num + 1} ---\n"
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting with PyPDF2: {e}")
        return None
    return text

def extract_text_pdfplumber(pdf_path):
    """Extract text using pdfplumber"""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- PAGE {page_num + 1} ---\n"
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error extracting with pdfplumber: {e}")
        return None
    return text

def extract_pdf_content(pdf_path):
    """Extract content from PDF using available library"""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    
    print(f"Extracting PDF content using {PDF_LIBRARY}...")
    
    if PDF_LIBRARY == "PyPDF2":
        return extract_text_pypdf2(pdf_path)
    elif PDF_LIBRARY == "pdfplumber":
        return extract_text_pdfplumber(pdf_path)
    
    return None

def save_extracted_text(text, output_path):
    """Save extracted text to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving text: {e}")
        return False

def main():
    # Define paths
    pdf_path = "RATTENTION Paper.pdf"
    output_path = "rattention_paper_content.txt"
    
    # Extract content
    extracted_text = extract_pdf_content(pdf_path)
    
    if extracted_text:
        # Save to file
        if save_extracted_text(extracted_text, output_path):
            print(f"\nExtraction completed successfully!")
            print(f"Content length: {len(extracted_text)} characters")
            print(f"Output file: {output_path}")
        else:
            print("Failed to save extracted content")
    else:
        print("Failed to extract PDF content")

if __name__ == "__main__":
    main()
