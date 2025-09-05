import pdfplumber
import streamlit as st
from PyPDF2 import PdfReader
import re

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
                    print(f"Found spec name: {spec_name.strip()}")
                    extracted_data["Submittal Name"] = spec_name.strip()
            except Exception as e:
                print(f"Error extracting submittal name: {e}")

            # DATE RECEIVED FROM GC
            try:
                issue_date_pattern = r"ISSUE\s*DATE:\s*(\d{1,2}\s*\/\s*\d{1,2}\s*\/\s*\d{4})"
                issue_date_match = re.search(issue_date_pattern, text)
                if issue_date_match:
                    issue_date = issue_date_match.group(1)
                    print(f"Found issue date: {issue_date}")
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
                    print(f"Found due date: {issue_date}")
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
                print(f"File Name: {file_name} \nResponse: {response}")
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