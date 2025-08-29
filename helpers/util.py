import io
import os
from typing import Dict
import re
import streamlit as st
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

async def run_conversation_with_context(messages, message_placeholder, context):
    # Combine context with last user message
    last_user_msg = messages[-1]["content"]
    ai_prompt = f"""
    Answer the following question using the context:
    %Question:
    ```{last_user_msg}```
    %Context:
    ```{context}```
    """

    # Temporarily append as "assistant" to capture output
    full_response = ""
    message_placeholder.markdown("Thinking...")
    chunks = services.llm.converse([{"role": "user", "content": ai_prompt}])
    chunk = await anext(chunks, "END OF CHAT")
    while chunk != "END OF CHAT":
        if chunk.startswith("EXCEPTION"):
            full_response = ":red[We are having trouble generating advice. Please wait a minute and try again.]"
            break
        full_response += chunk
        message_placeholder.markdown(full_response + "▌")
        chunk = await anext(chunks, "END OF CHAT")

    message_placeholder.markdown(full_response)
    messages.append({"role": "assistant", "content": full_response})
    return messages

async def chat(messages, ai_prompt=None, state_key="pdf_messages"):
    # Show user message (already in messages)
    with st.chat_message("user"):
        st.markdown(messages[-1]["content"])

    # Show assistant message placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        if ai_prompt:
            # Run AI call internally without adding the huge prompt to chat
            response_messages = await run_conversation_with_context(messages, placeholder, ai_prompt)
            messages.extend(response_messages)
        st.session_state[state_key] = messages

    return messages

def render_messages(messages):
    """Render all messages in Streamlit chat UI."""
    for msg in messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])