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
        print(pages_text)
        return pages_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")

def process_chunks_to_embeddings(chunks: list[tuple[str, int, str]]):
    embeddings = []
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = [chunk[2] for chunk in chunks[batch_start:batch_end]] # extract text for embeddings
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch, encoding_format="float"
            )
            batch_embeddings = [e.embedding for e in response.data]

            for i, embedding in enumerate(batch_embeddings):
                doc_name, page_num, text = chunks[batch_start + i]
                embeddings.append([doc_name, page_num, text])

        except Exception as e:
            print(f'Failed to generate embeddings: {e}')
            continue

    if not embeddings:
        print("No embeddings were generated successfully.")
        return None

    try:
        df = pd.DataFrame(embeddings, columns=["document_name", "page_number", "embedding", "context"])

    # Further processing
    except pd.errors.EmptyDataError:
        print("No data to create DataFrame.")
    except Exception as e:
        print(f"Unexpected error while processing DataFrame: {e}")
    csv_file_path = "data/Mineral_Fiber_SpecSheet_v3.embeddings.csv"
    df.to_csv(csv_file_path, index=False)
    return df

def load_embeddings(embeddings_file_path):
    if os.path.exists(embeddings_file_path):
        embeddings_df = pd.read_csv(embeddings_file_path)
        expected_columns = ["document_name", "page_number", "embedding", "context"]
        if all(column in embeddings_df.columns for column in expected_columns):
            print("Loaded embeddings from existing CSV file.")
            return embeddings_df
        else:
            print("Error loading csv file")
            return None
    return None

def display_page(st, first_relevant_page_num, most_relevant_context_index, pdf_chunks, uploaded_pdf):
    with st.expander("Relevant Page", expanded=False):
        ## Per office hours with Dr. A, he wants us to show the most relevant page rather than the first relevant page. He'll change the requirements
        if most_relevant_context_index is not None:
            images = convert_pdf_page_to_img(uploaded_pdf, [most_relevant_context_index])
            if images:
                st.image(images[0], caption=f"Page {most_relevant_context_index}")

        if most_relevant_context_index != -1:
            most_relevant_info = pdf_chunks[most_relevant_context_index]
            if most_relevant_info['page_number'] != first_relevant_page_num:
                st.write(f"The most relevant content is found on page {most_relevant_info['page_number']}, but the first relevant content shown is from page {first_relevant_page_num}.")

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

async def analyze_with_ai(submittal_embedding, manual_embeddings_df):
    try:
        # Step 1: Prepare the embeddings and create a NearestNeighbors model
        manual_embeddings = np.vstack(manual_embeddings_df['embedding'].values)  # Convert list of embeddings into a 2D numpy array
        knn = NearestNeighbors(n_neighbors=1, metric='cosine')  # Use cosine similarity as the distance metric
        knn.fit(manual_embeddings)  # Fit the model to the manual embeddings

        # Step 2: Find the nearest neighbor (most similar section) from the manual
        distances, indices = knn.kneighbors([submittal_embedding])  # Find the nearest neighbor for the submittal embedding

        # Step 3: Get the text and similarity score of the closest match
        best_match_idx = indices[0][0]
        best_match_text = manual_embeddings_df.iloc[best_match_idx]['text']
        best_match_distance = distances[0][0]  # Lower distance means higher similarity

        # Step 4: Generate AI feedback based on the most similar section from the manual
        threshold = 0.75  # You can adjust this threshold based on your requirements
        similarity_score = 1 - best_match_distance  # Cosine similarity is 1 - distance

        if similarity_score > threshold:
            ai_feedback = f"The submittal aligns well with the project manual's section: {best_match_text[:100]}... with a similarity of {similarity_score:.2f}."
        else:
            ai_feedback = "The submittal does not appear to comply with the project manual."

        # Step 5: Provide AI feedback
        print(ai_feedback)
        return ai_feedback

    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        return "Error analyzing the submittal."

def build_search_index(embeddings):
    """
    Builds a nearest neighbor search index for embeddings.
    """
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(embeddings)
    return nn

def semantic_search(nn_model, query_embeddings):
    """
    Searches for the most similar chunks in the index for each query embedding.
    """
    distances, indices = nn_model.kneighbors(query_embeddings)
    return indices, distances

def create_comparison_report(query_chunks, manual_chunks, indices, distances, threshold=0.8):
    """
    Matches submittal chunks to project manual chunks and creates a comparison report.
    """
    report = []
    for i, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
        best_match_idx = neighbor_indices[0]
        best_similarity = 1 - neighbor_distances[0]  # Convert cosine distance to similarity score

        report.append({
            "query_chunk": query_chunks[i],
            "manual_chunk": manual_chunks[best_match_idx],
            "similarity_score": best_similarity,
            "status": "Match" if best_similarity >= threshold else "No Match"
        })
    return report

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