import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
import tiktoken as tkn
import os
import streamlit as st


# Load secrets from .toml
openai_model = st.secrets["openai_api_model"]
openai_api_key = st.secrets["openai_api_key"]
base_url = st.secrets["openai_api_base_url"]

client = OpenAI(
    base_url= base_url,
    api_key=openai_api_key,
)
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 20

def chunk_prompt(inputs: List[Tuple[str, int, str]], chunk_size: int = 1500, overlap: int = 225) -> List[Tuple[str, int, str]]:
    """
    Splits text from a list of inputs into chunks of approximately `chunk_size` tokens, with a given overlap,
    while preserving the document name and page number for each chunk.

    Parameters:
    - inputs (List[Tuple[str, int, str]]): A list of tuples containing document name, page number, and the text to be chunked.
    - chunk_size (int): The desired number of tokens for each chunk.
    - overlap (int): The number of tokens for overlap between chunks.

    Returns:
    - List[Tuple[str, int, str]]: A list of tuples containing document name, page number, and prompt chunks.
    """

    encoding = tkn.encoding_for_model("gpt-3.5-turbo")

    output_chunks = []

    for doc_name, page_num, prompt in inputs:
        tokens = list(encoding.encode(prompt))

        if len(tokens) <= chunk_size:
            output_chunks.append((doc_name, page_num, prompt))
            continue

        position = 0
        while position < len(tokens):
            start_pos = max(0, position - overlap)
            end_pos = min(position + chunk_size, len(tokens))

            chunk_tokens = tokens[start_pos:end_pos]
            chunk_text = ''.join(encoding.decode_bytes(chunk_tokens).decode('utf-8', errors='ignore'))

            output_chunks.append((doc_name, page_num, chunk_text))

            position += chunk_size

    return output_chunks

def chunks_to_embeddings_csv(chunks: list[tuple[str, int, str]]):
    embeddings = []
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = [chunk[2] for chunk in chunks[batch_start:batch_end]] # extract text for embeddings
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch, encoding_format="float"
            )
        except Exception as e:
            print(f'Failed to generate embeddings: {e}')

        batch_embeddings = [str(e.embedding) for e in response.data]

        for i, embedding in enumerate(batch_embeddings):
            doc_name, page_num, text = chunks[batch_start + i]
            embeddings.append((doc_name, page_num, embedding, text))

    try:
        df = pd.DataFrame(embeddings, columns=["document_name", "page_number", "embedding", "context"])
    # Further processing
    except pd.errors.EmptyDataError:
        print("No data to create DataFrame.")
    except Exception as e:
        print(f"Unexpected error while processing DataFrame: {e}")

    # Save the DataFrame to a CSV file
    csv_file_path = "data/SampleProjectManual.embeddings.csv"
    df.to_csv(csv_file_path, index=False)
    print("Completed csv file")

def process_chunks_to_embeddings(chunks: list[tuple[str, int, str]], source_filename):
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
                embeddings.append([doc_name, page_num, embedding, text])

        except Exception as e:
            print(f'Failed to generate embeddings: {e}')
            continue

    if not embeddings:
        print("No embeddings were generated successfully.")
        return None

    try:
        df = pd.DataFrame(embeddings, columns=["document_name", "page_number", "embedding", "context"])
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        csv_file_path = os.path.join("data", f'{base_name}.embeddings.csv')
        df.to_csv(csv_file_path, index=False)
        return df, csv_file_path

    # Further processing
    except pd.errors.EmptyDataError:
        print("No data to create DataFrame.")
    except Exception as e:
        print(f"Unexpected error while processing DataFrame: {e}")

def prepare_documents_from_embeddings(embeddings_df):
    documents = []
    for _, row in embeddings_df.iterrows():
        document = {
            "document_name": row["document_name"],
            "page_number": row["page_number"],
            "context": row["context"],
            "embedding": row["embedding"]
        }
        documents.append(document)
    return documents

def convert_embedding(row):
    # Assuming the embedding is stored in a column named 'embedding'
    # and is formatted as a string representation of a list: "[0.1, 0.2, ...]"
    try:
        embedding_str = row['embedding'].strip("[]")  # Remove the square brackets
        embedding_list = embedding_str.split(",")  # Split the string into a list of strings
        embedding_array = np.array(embedding_list, dtype=float)  # Convert to numpy array
        return embedding_array
    except ValueError:
        print("Failed to convert embedding into numpy array")
        return np.array([])
    except Exception as e:
        print(f"Unexpected error in convert_embedding: {e}")
        return np.array([])

def convert_embeddings_dataframe(embeddings_df):
    try:
        converted_embeddings = embeddings_df.apply(convert_embedding, axis= 1)
        embeddings_df['converted_embeddings'] = converted_embeddings
        return np.vstack(embeddings_df['converted_embeddings'].values)
    except Exception as e:
        print(f"Failed to convert embeddings dataframe: {e}")
        return None

def generate_prompt_embedding_array(query: str):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=query, encoding_format="float")
        embedding = response.data[0].embedding
        print(embedding)
        prompt_embedding_array = np.array(embedding).reshape(1, -1)
        return prompt_embedding_array
    except Exception as e:
        print(f"An error occurred while generating embeddings")
        return None

def load_or_process_embeddings(embeddings_file_path):
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

def ask_book(embeddings_matrix, prompt_embeddings, embeddings_df):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(embeddings_matrix)

    distances, indices = nbrs.kneighbors(prompt_embeddings)
    print("Nearest Neighbors Indices:", indices)
    print("Distances:", distances)

    count = 0
    for idx in indices[0]:
        # Assuming 'documents' is a list of contexts or summaries for each embedding
        # You might need to adjust how you access document information based on your DataFrame structure
        doc = embeddings_df.iloc[idx]['context']  # Adjust this line as needed
        print("""[{idx}]@{distance} {doc}""".format(idx=idx, distance=distances[0][count], doc=doc.replace("\n", " ")))
        print("-" * 100)
        print("\n")
        count += 1

    return indices, distances

def process_search_results(all_relevant_context, indices, distances, pdf_chunks):
    """
    Process search results from vector similarity search.

    Args:
        all_relevant_context (str): Existing concatenated context.
        indices (List[List[int]]): List of lists of chunk indices from the search.
        distances (List[List[float]]): List of lists of distances for each index.
        pdf_chunks (List[Dict]): Each dict contains 'page_number' and 'context'.

    Returns:
        Tuple[str, List[int], int]:
            - Updated all_relevant_context string,
            - Sorted unique relevant page numbers,
            - Index of the most relevant context.
    """
    relevant_page_numbers = []
    most_relevant_distance = float('inf')
    most_relevant_context_index = -1
    for idx, distance in zip(indices[0], distances[0]):
        relevant_info = pdf_chunks[idx]
        all_relevant_context += f"Page Number: {relevant_info['page_number']}\nContext: {relevant_info['context']}\n------\n"
        relevant_page_numbers.append(relevant_info['page_number'])

        if distance < most_relevant_distance:
            most_relevant_context_index = idx
            most_relevant_distance = distance
    relevant_page_numbers = sorted(set(relevant_page_numbers))
    return all_relevant_context, relevant_page_numbers, most_relevant_context_index

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

def search_pdf_with_prompt(pdf_embeddings, pdf_chunks, prompt):
    """Run semantic search and return relevant context."""
    prompt_embedding_array = generate_prompt_embedding_array(prompt)
    embeddings_matrix = convert_embeddings_dataframe(pdf_embeddings)
    indices, distances = ask_book(embeddings_matrix, prompt_embedding_array, pdf_embeddings)
    all_relevant_context, relevant_pages, most_relevant_index = process_search_results("", indices, distances, pdf_chunks)

    first_page = sorted(set(relevant_pages))[0] if relevant_pages else None
    ai_prompt = f"""
    Answer the following question using the context:
    %Question:
    ```{prompt}```
    %Context:
    ```{all_relevant_context}```
    """

    return {
        "all_context": all_relevant_context,
        "first_page": first_page,
        "most_relevant_index": most_relevant_index,
        "ai_prompt": ai_prompt
    }

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