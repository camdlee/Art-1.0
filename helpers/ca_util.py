from openai import OpenAI
import numpy as np
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader
import re
from typing import Dict, List
import os
from pdf2image.pdf2image import convert_from_bytes
from sklearn.neighbors import NearestNeighbors

client = OpenAI(
    # This is the default and can be omitted
    # base_url = 'http://aitools.cs.vt.edu:7860/openai/v1',
    # api_key="aitools"
    base_url='https://api.aimlapi.com/v1',
    api_key='ed2ac7d3785a4ffbabd819e1044ce8e6'
)

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's best embeddings as of Feb 2024
BATCH_SIZE = 20  # you can submit up to 2048 embedding inputs per request


### -------------------- CONSTRUCTION ADMINISTRATION LOG FUNCTIONS --------------------
def extract_data_from_submittal(submittal_file) -> dict:
    """
    Extracts key metadata from a PDF submittal file using text patterns.

    This function attempts to parse specific fields from the contents of a construction submittal PDF,
    including the submittal number, submittal name, dates (received, due), and response codes. It uses
    `pdfplumber` and `PyPDF2` to read the PDF and regular expressions to extract relevant data.

    Parameters
    ----------
    submittal_file : file-like object
    A file-like object representing the PDF submittal document. The object should be compatible
    with `pdfplumber.open()` and `PyPDF2.PdfReader`.

    Returns
    -------
    dict
    A dictionary containing extracted submittal metadata. The dictionary may include the following keys:
    - "Submittal Number": str
    - "Submittal Name": str
    - "Date Received From GC": str (MM/DD/YYYY)
    - "Due Date": str (MM/DD/YYYY)
    - "Response": str

    Notes
    -----
    - Some data fields such as "Date Sent to Engineers", "Date Returned", and "Comments" are stubbed out and not yet implemented.
    - The function uses filename information to infer the "Response" value.
    - Regular expressions are used for pattern matching and may need to be updated for different PDF templates.
    - Incomplete or malformed PDFs may result in missing or incorrect fields.
    """

    extracted_data = {}
    with pdfplumber.open(submittal_file) as pdf:
        reader = PdfReader(submittal_file)

        for page in reader.pages:
            text = page.extract_text()

            # SUBMITTAL NUMBER
            try:
                sub_num_pattern = r"Submittal #\s*(\d[\d\s\-\.]*)(?=\n|$)"
                sub_num_match = re.search(sub_num_pattern, text)
                if sub_num_match:
                    submittal_num = sub_num_match.group(1)
                    extracted_data["Submittal Number"] = submittal_num
            except Exception as e:
                print(f"Error extracting submittal number: {e}")

            # SUBMITTAL NAME
            try:
                spec_name_pattern = r"(.*?)\s*SPEC SECTION"
                spec_name_match = re.search(spec_name_pattern, text)
                if spec_name_match:
                    spec_name = spec_name_match.group(1)
                    spec_name = spec_name.replace("\xa0", " ")
                    # print(f"Found spec name: {spec_name.strip()}")
                    extracted_data["Submittal Name"] = spec_name.strip()
            except Exception as e:
                print(f"Error extracting submittal name: {e}")

            # DATE RECEIVED FROM GC
            try:
                issue_date_pattern = r"ISSUE\s*DATE:\s*(\d{1,2}\s*\/\s*\d{1,2}\s*\/\s*\d{4})"
                issue_date_match = re.search(issue_date_pattern, text)
                if issue_date_match:
                    issue_date = issue_date_match.group(1)
                    # print(f"Found issue date: {issue_date}")
                    extracted_data["Date Received From GC"] = issue_date
            except Exception as e:
                print(f"Error extracting issue date: {e}")

            # DATE SENT TO ENGINEERS
            # try:
            #     sub_num_pattern = r"Submittal #\s*(\d[\d\s\-\.]*)(?=\n|$)"
            #     sub_num_match = re.search(sub_num_pattern, text)
            #     if sub_num_match:
            #         submittal_num = sub_num_match.group(1)
            #         extracted_data["Submittal Number"] = submittal_num
            # except Exception as e:
            #     print(f"Error extracting submittal number: {e}")

            # DATE RECEIVED FROM ENGINEERS
            # try:
            #     sub_num_pattern = r"Submittal #\s*(\d[\d\s\-\.]*)(?=\n|$)"
            #     sub_num_match = re.search(sub_num_pattern, text)
            #     if sub_num_match:
            #         submittal_num = sub_num_match.group(1)
            #         extracted_data["Submittal Number"] = submittal_num
            # except Exception as e:
            #     print(f"Error extracting submittal number: {e}")

            # DUE DATE
            try:
                due_date_pattern = r"DUE\s*DATE:\s*(\d{1,2}\s*\/\s*\d{1,2}\s*\/\s*\d{4})"
                due_date_match = re.search(due_date_pattern, text)
                if due_date_match:
                    due_date = due_date_match.group(1)
                    # print(f"Found issue date: {issue_date}")
                    extracted_data["Due Date"] = due_date
            except Exception as e:
                print(f"Error extracting due date: {e}")

            # DATE RETURNED
            # try:
            #     sub_num_pattern = r"Submittal #\s*(\d[\d\s\-\.]*)(?=\n|$)"
            #     sub_num_match = re.search(sub_num_pattern, text)
            #     if sub_num_match:
            #         submittal_num = sub_num_match.group(1)
            #         extracted_data["Submittal Number"] = submittal_num
            # except Exception as e:
            #     print(f"Error extracting submittal number: {e}")

            # RESPONSE
            try:
                file_name = submittal_file.name
                last_3_chars = file_name[-7:-4]
                response = re.sub(r'[^a-zA-Z]', '', last_3_chars)
                # print(f"File Name: {file_name} \nResponse: {response}")
                extracted_data["Response"] = response
            except Exception as e:
                print(f"Error extracting submittal response")

            # COMMENTS
            # try:
            #     due_date_pattern = r"DUE\s*DATE:\s*(\d{1,2}\s*\/\s*\d{1,2}\s*\/\s*\d{4})"
            #     due_date_match = re.search(due_date_pattern, text)
            #     if due_date_match:
            #         due_date = due_date_match.group(1)
            #         # print(f"Found issue date: {issue_date}")
            #         extracted_data["Due Date"] = due_date
            # except Exception as e:
            #     print(f"Error extracting due date: {e}")

        return extracted_data

def extract_data_from_rfi(rfi_file) -> dict:
    """
    Extract fields values from a rfi
    :param rfi_file:
    :return:
    """
    extracted_data = {
        "RFI Number": None,
        "RFI Name": None,
        "Issue Date": None,
        "Due Date": None,
    }

    with pdfplumber.open(rfi_file) as pdf:
        reader = PdfReader(rfi_file)
        # loop through each page to find the patterns for each field
        for page in reader.pages:
            text = page.extract_text()
            print(f"PDF Text: {text}")
            try:
                # RFI Number and Name field
                rfi_pattern = r"RFI\s+#\s*(\d+):\s*(.+?)(?=\n|Status)"
                rfi_match = re.search(rfi_pattern, text)
                if rfi_match:
                    rfi_num = rfi_match.group(1)
                    rfi_name = rfi_match.group(2)
                    print(f"Found RFI number: {rfi_num}")
                    extracted_data["RFI Number"] = rfi_num
                    extracted_data["RFI Name"] = rfi_name
            except Exception as e:
                print(f"Error extracting rfi number: {e}")

            try:
                # Date initiated field
                issue_date_pattern = r"Date\s+Initiated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})"
                issue_date_match = re.search(issue_date_pattern, text)
                if issue_date_match:
                    issue_date = issue_date_match.group(1)
                    issue_date = issue_date.replace("\xa0", "").strip()
                    print(f"Found issue date: {issue_date}")
                    extracted_data["Issue Date"] = issue_date
            except Exception as e:
                print(f"Error extracting issue date: {e}")

            try:
                # Due date field
                due_date_pattern = r"Due\s*Date\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})"
                due_date_match = re.search(due_date_pattern, text)
                if due_date_match:
                    due_date = due_date_match.group(1)
                    extracted_data["Due Date"] = due_date
            except Exception as e:
                print(f"Error extracting due date: {e}")

    return extracted_data


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
        question_pattern = r"at\s+\d{1,2}:\d{2}\s+[AP]M\s+\w+\s*\n(.+)"
        question_match = re.search(question_pattern, text, re.IGNORECASE | re.DOTALL)

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