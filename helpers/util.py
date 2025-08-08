import io
import os
from typing import Dict
import re

from dotenv import load_dotenv
from openai import OpenAI
import pandas
from streamlit.delta_generator import DeltaGenerator
import numpy as np
from PyPDF2 import PdfReader
import pandas as pd
import services.llm
import tiktoken as tkn
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from pdf2image import convert_from_path

# Load .env file
load_dotenv()
openai_model = os.getenv('OPENAI_API_MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE_URL')

client = OpenAI(
    base_url= base_url,
    api_key=openai_api_key,
)

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's best embeddings as of Feb 2024
BATCH_SIZE = 20  # you can submit up to 2048 embedding inputs per request

# ------------------------------------------------- FUNCTIONS -------------------------------------------------
async def run_conversation(messages: List[Dict[str, str]], message_placeholder: DeltaGenerator) \
        -> List[Dict[str, str]]:
    full_response = ""
    message_placeholder.markdown("Thinking...")
    chunks = services.llm.converse(messages)
    chunk = await anext(chunks, "END OF CHAT")
    while chunk != "END OF CHAT":
        print(f"Received chunk from LLM service: {chunk}")
        if chunk.startswith("EXCEPTION"):
            full_response = ":red[We are having trouble generating advice.  Please wait a minute and try again.]"
            break
        full_response = full_response + chunk
        message_placeholder.markdown(full_response + "▌")
        chunk = await anext(chunks, "END OF CHAT")
    message_placeholder.markdown(full_response)
    messages.append({"role": "assistant", "content": full_response})
    return messages