from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path, convert_from_bytes
from typing import List, Tuple
import streamlit as st
import os

from helpers.embeddings_util import load_or_process_embeddings, prepare_documents_from_embeddings, chunk_prompt, process_chunks_to_embeddings


def extract_text_from_saved_pdf(pdf_path) -> List[tuple[str, int, str]]:
    document_name = pdf_path.split('/')[-1]
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            pages_text = [(document_name, i+1, page.extract_text()) for i, page in enumerate(reader.pages)]
        # returns list of tuples where each tuple contains the document name, page number and extracted text for that page
        print(pages_text)
        return pages_text
    except FileNotFoundError:
        print("PDF file not found")
    except Exception as e:
        print(f"Error reading PDF file: {e}")

def extract_text_from_uploaded_pdf(uploaded_pdf) -> List[tuple[str, int, str]]:
    """
    Extracts text from each page of an uploaded PDF.

    Parameters:
    - uploaded_pdf: A file-like object representing the uploaded PDF.

    Returns:
    - A list of tuples where each tuple contains:
        - Document name
        - Page number (1-indexed)
        - Extracted text for that page
    """
    try:
        document_name = uploaded_pdf.name
        reader = PdfReader(uploaded_pdf)
        pages_text = [(document_name, i+1, page.extract_text()) for i, page in enumerate(reader.pages)]

        # returns list of tuples where each tuple contains the document name, page number and extracted text for that page
        # print(pages_text)
        return pages_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")

def process_pdf_upload(uploaded_file):
    """

    :param uploaded_file:
    :return:
    """
    if not uploaded_file:
        return None, None

    # Save uploaded file to folder
    file_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text and prepare file paths
    pdf_content = extract_text_from_uploaded_pdf(uploaded_file)
    base_name = os.path.splitext(uploaded_file.name)[0]
    csv_file_path = os.path.join("data", f'{base_name}.embeddings.csv')

    #Load existing embeddings or create new ones
    if os.path.exists(csv_file_path):
        with st.spinner("Processing file"):
            pdf_embeddings = load_or_process_embeddings(csv_file_path)
            pdf_chunks = prepare_documents_from_embeddings(pdf_embeddings)
            st.toast("Loaded pdf embeddings successfully!")
    else:
        with st.spinner("Processing file"):
            pdf_chunks = chunk_prompt(pdf_content)
            pdf_embeddings, pdf_file_path = process_chunks_to_embeddings(pdf_chunks, uploaded_file.name)

    return pdf_embeddings, pdf_chunks

def validate_pdf(uploaded_pdf):
    try:
        uploaded_pdf.seek(0)
        reader = PdfReader(uploaded_pdf)
        _ = len(reader.pages)
        return True
    except Exception as e:
        print(f"File validation error: {e}")
        return False

def convert_pdf_page_to_img(uploaded_pdf, page_number):
    """
    Convert specific pages of a PDF to images.

    Parameters:
    - uploaded_pdf: A file-like object representing the uploaded PDF.
    - page_numbers: A list of page numbers to convert.

    Returns:
    - A list of images, one for each page number in page_numbers.
    """
    # Convert_from_path function returns a list of images for the specified pages.
    # `first_page` and `last_page` parameters are used to define the range.
    # Note: Pages are 1-indexed, hence first_page=page_number, last_page=page_number+1
    images = []
    if validate_pdf(uploaded_pdf):
        try:
            page_img = convert_from_bytes(uploaded_pdf.read(), first_page=page_number)
            images.extend(page_img)
        except Exception as e:
            print(f"Failed to convert PDF page to image: {e}")
            raise RuntimeError(f"Failed to covert PDF page to image: {e}")
    else:
        raise RuntimeError("The uploaded file is not a valid PDF.")

    return images

def convert_saved_pdf_page_to_img(pdf_path, page_numbers):
    """
    Convert specific pages of a PDF to images.

    Parameters:
    - pdf_path: Path to the PDF file.
    - page_numbers: A list of page numbers to convert.

    Returns:
    - A list of images, one for each page number in page_numbers.
    """
    # Convert_from_path function returns a list of images for the specified pages.
    # `first_page` and `last_page` parameters are used to define the range.
    # Note: Pages are 1-indexed, hence first_page=page_number, last_page=page_number+1
    images = []
    try:
        for page_number in page_numbers:
            page_imgs = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
            images.extend(page_imgs)
    except Exception as e:
        print(f"Failed to convert PDF page to image: {e}")

    return images

def display_pdf_page(first_relevant_page_num, most_relevant_context_index, pdf_chunks, uploaded_pdf_path):
    with st.expander("Relevant Page", expanded=False):
        ## Per office hours with Dr. A, he wants us to show the most relevant page rather than the first relevant page. He'll change the requirements
        if most_relevant_context_index is not None:
            images = convert_saved_pdf_page_to_img(uploaded_pdf_path, [most_relevant_context_index])
            if images:
                st.image(images, caption=f"Page {most_relevant_context_index}")

        if most_relevant_context_index != -1:
            most_relevant_info = pdf_chunks[most_relevant_context_index]
            if most_relevant_info['page_number'] != first_relevant_page_num:
                st.write(f"The most relevant content is found on page {most_relevant_info['page_number']}, but the first relevant content shown is from page {first_relevant_page_num}.")

