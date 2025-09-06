from openai import OpenAI
import pdfplumber
import streamlit as st
from PyPDF2 import PdfReader
import re
from typing import List
import os

# Load secrets from .toml
openai_model = st.secrets["openai_api_model"]
openai_api_key = st.secrets["openai_api_key"]
base_url = st.secrets["openai_api_base_url"]

client = OpenAI(
    base_url= base_url,
    api_key=openai_api_key,
)

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's best embeddings as of Feb 2024
BATCH_SIZE = 20  # you can submit up to 2048 embedding inputs per request


### -------------------- CONSTRUCTION ADMINISTRATION REVIEW --------------------
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

# def handle_pdf_query(prompt_text, pdf_embeddings, pdf_chunks):
#     if pdf_embeddings is None:
#         st.session_state.pdf_messages.append({"role": "user", "content": prompt_text})
#         asyncio.run(chat(st.session_state.pdf_messages, prompt_text))
#     return


### -------------------- CONSTRUCTION ADMINISTRATION DATA COLLECTION --------------------
supported_extensions = ['.pdf', '.docx', '.xlsx', '.xls']

def get_all_files_from_folder(root_folder: str) -> list:
    """
    Recursively walks through the root folder and returns a list of all file paths.

    :param root_folder: The path to the top-level folder.
    :return: A list of full file paths.
    """
    all_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    return all_files

def extract_rfi_list_from_dir(root_folder):
    """
    Recursively walks through the root folder and returns a list of all file paths.

    :param root_folder: The path to the top-level folder.
    :return: A list of full file paths.
    """
    all_rfi_file_data = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)

            if filename.lower().endswith(".pdf"):
                try:
                    rfi_data = extract_rfi_data(full_path)
                    if rfi_data:
                        all_rfi_file_data.append(rfi_data)
                except Exception as e:
                    print(f"Error extracting rfi data: {e}")
    return all_rfi_file_data

def extract_rfi_data(rfi_file) -> dict:
    extracted_data = {
        "Project": None,
        "Project Address": None,
        "General Contractor": None,
        "Architect Team": None,
        "Project Management Team": None,
        "GC Team": None,
        "MEP Consultants": None,
        "Consultants": None,
        "RFI Number": None,
        "RFI Name": None,
        "Issue Date": None,
        "Due Date": None,
        "Question": None
    }

    try:
        gc_info_text = ""
        proj_info_text = ""
        rfi_title_text =""

        with pdfplumber.open(rfi_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
            #print(full_text)

            first_page = pdf.pages[0]

            #-------- GC Info extraction ---------
            gc_info_bounding_box = (180, 0, 350, 90)
            gc_region = first_page.within_bbox(gc_info_bounding_box)
            if gc_region:
                gc_info_text += gc_region.extract_text()
                #print(gc_info_text)
            gc_lines = gc_info_text.strip().splitlines()
            gc_company_name = gc_lines[0].strip()
            gc_address_1 = gc_lines[1].strip()
            gc_address_2 = gc_lines[2].strip()
            gc_full_address = f'{gc_address_1}, {gc_address_2}'

            extracted_data["General Contractor"] = gc_company_name

            #-------- Project Info extraction ---------
            proj_info_bounding_box = (400, 0, 600, 90)
            proj_info_region = first_page.within_bbox(proj_info_bounding_box)
            if proj_info_region:
                proj_info_text += proj_info_region.extract_text()
            project_lines = proj_info_text.strip().splitlines()
            project_first_line = project_lines[0].strip()
            project_address_1 = project_lines[1].strip()
            project_address_2 = project_lines[2].strip()
            project_full_address = f'{project_address_1}, {project_address_2}'

            project_name_match = re.search(r"Project:\s*\d{2}-\d{2}-\d{3}\s+(.+)", project_first_line)
            if project_name_match:
                project_name = project_name_match.group(1).strip()
                extracted_data["Project"] = project_name
            extracted_data["Project Address"] = project_full_address

            #-------- RFI Title extraction ---------
            rfi_title_bounding_box = (0, 90, 600, 130)
            rfi_title_region = first_page.within_bbox(rfi_title_bounding_box)
            if rfi_title_region:
                rfi_title_text += rfi_title_region.extract_text()

            rfi_pattern = r"RFI\s+#\s*(\d+):\s*(.+)"
            rfi_match = re.search(rfi_pattern, rfi_title_text, re.IGNORECASE)
            if rfi_match:
                rfi_num = rfi_match.group(1).strip()
                rfi_name = rfi_match.group(2).strip()
                #print(f"Found RFI number: {rfi_num}")
                extracted_data["RFI Number"] = rfi_num
                extracted_data["RFI Name"] = rfi_name

        text = full_text.replace("\xa0", " ").replace("\u2013", "-").strip()

        # --- Extract Issue Date ---

        issue_date_pattern = r"Date\s+Initiated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})"
        issue_date_match = re.search(issue_date_pattern, full_text)
        if issue_date_match:
            issue_date = issue_date_match.group(1)
            issue_date = issue_date.replace("\xa0", "").strip()
            #print(f"Found issue date: {issue_date}")
            extracted_data["Issue Date"] = issue_date

        # ---- Extract Question -------
        # if it ends with Awaiting an Official Response
        question_pattern = (
            r"at\s+\d{1,2}:\d{2}\s+[AP]M\s+\w+\s*\n"        # Match timestamp line
            r"(.*?)"                                        # Non-greedy match of question text
            r"(?=Awaiting an Official Response)"           # Stop before this phrase
        )
        #
        alt_question_pattern = (
            r"at\s+\d{1,2}:\d{2}\s+[AP]M\s+[A-Z]+\s*\n+"  # after time stamp (e.g., at 12:58 PM EST)
            r"(.*?)"                                      # capture question
            r"(?=Page\s+1\s+of)"                          # stop at Page 1 of
        )
        question_match = re.search(alt_question_pattern, text, re.IGNORECASE | re.DOTALL)

        if question_match:
            question_text = question_match.group(1).strip()
            extracted_data["Question"] = question_text

    except Exception as e:
        print(f"Error extracting rfi information: {e}")

    return extracted_data

def extract_people_by_team(text:str) -> dict:
    people_by_team = {}

    entries = re.findall(r'([A-Za-z\s]+)\s+\(([^)]+)\)', text)

    for name, affiliation in entries:
        name = name.strip()
        affiliation = affiliation.strip()
        people_by_team[affiliation].append(name)

    return people_by_team