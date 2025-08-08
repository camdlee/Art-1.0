import PyPDF2
import asyncio
import os
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer
from asyncio import sleep
import pandas as pd

import helpers.sidebar
import helpers.util
from aitools_autogen.bluebrint_RFI_review import RFIReviewBlueprint
from aitools_autogen.config import llm_config_openai as llm_config
from aitools_autogen.utils import clear_working_dir
from streamlit_file_browser import st_file_browser

from helpers import util
from helpers.ca_util import extract_data_from_submittal, extract_data_from_rfi, extract_text_from_uploaded_pdf, \
    process_chunks_to_embeddings, display_page, load_embeddings, analyze_with_ai, convert_pdf_page_to_img, \
    build_search_index, semantic_search, create_comparison_report
from helpers.util import chunk_prompt, prepare_documents_from_embeddings, generate_prompt_embedding_array, \
    convert_embeddings_dataframe, process_search_results, ask_book, display_evidence
from services import prompts

st.set_page_config(
    page_title="AI Powered CA Review",
    page_icon="🔍",
    layout="wide"
)
## --------------------------------------------- SIDEBAR ------------------------------------------------
helpers.sidebar.show()

## --------------------------------------------- SESSION STATES ------------------------------------------------
# Ensure the session state is initialized
if "proj_manual_messages" not in st.session_state:
    initial_messages = [{"role": "system",
                         "content": prompts.quick_chat_system_prompt()}]
    st.session_state.proj_manual_messages = initial_messages

## --------------------------------------------- HEADER ------------------------------------------------
st.header("Construction Document Review")
st.write("Uploaded your construction documents and leverage AI to review your documents")

#     try:
#         file_path = os.path.join("data", uploaded_files.name)
#         os.makedirs("data", exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_files.read())
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#
# with st.spinner("Loading file"):
#     if os.path.exists("data/SampleProjectManual.pdf"):
#         st.success("Project manual found")
#         st.session_state.project_manual_path = "data/SampleProjectManual.pdf"
#         proj_manual_embeddings_df = load_or_process_embeddings("data/SampleProjectManual.embeddings.csv")
#         if proj_manual_embeddings_df is not None:
#             proj_manual_chunks = prepare_documents_from_embeddings(proj_manual_embeddings_df)
#         else:
#             proj_manual_content = extract_text_from_pdf(st.session_state.project_manual_path)
#             proj_manual_chunks = chunk_prompt(proj_manual_content)
#             chunks_to_embeddings_csv(proj_manual_chunks)
# st.session_state.project_manual = uploaded_files[0]
# st.write(f"Uploaded file name: {st.session_state.project_manual.name}")
# reader = PyPDF2.PdfReader(st.session_state.project_manual)
# for page in reader.pages:
#     print(page.extract_text())
# st.text(page.extract_text())


# ---------------------------------------- TABS --------------------------------------
tabs = st.tabs(['Project Manual Review', 'RFI Review'])

## --------------------------------------------- PROJECT MANUAL QUESTIONS ------------------------------------------------
with tabs[0]:
    st.write("This tab applies semantic searching to review your project manual and find the most pertinent answer to your question. At the end, it will display the page you're looking for.")
    ## --------------------------------------------- FILE UPLOAD ------------------------------------------------
    uploaded_file = st.file_uploader("Please provide your project manual", type=['pdf'])

    if uploaded_file is not None:
        proj_manual_embeddings = None
        proj_manual_content = extract_text_from_uploaded_pdf(uploaded_file)
        # check if embeddings exist locally
        if os.path.exists("data/SampleProjectManual.embeddings.csv"):
            with st.spinner("Processing file"):
                proj_manual_embeddings = load_embeddings("data/SampleProjectManual.embeddings.csv")
                proj_manual_chunks = prepare_documents_from_embeddings(proj_manual_embeddings)
                st.toast("Loaded project manual embeddings")
                st.write(proj_manual_embeddings)
        else:
            # create embeddings and chunks for uploaded pdf
            with st.spinner("Processing file"):

                proj_manual_chunks = chunk_prompt(proj_manual_content)
                proj_manual_embeddings = process_chunks_to_embeddings(proj_manual_chunks)
    if uploaded_file is not None:
        st.write("Ask any general question from your project manual")

        # Print all messages in the session state
        for message in [m for m in st.session_state.proj_manual_messages if m["role"] != "system"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        ## --------------------------------------------- CHAT FEATURE ------------------------------------------------
        # Chat with the LLM, and update the messages list with the response.
        # Handles the chat UI and partial responses along the way.
        async def chat(messages):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                messages = await util.run_conversation(messages, message_placeholder)
                st.session_state.proj_manual_messages = messages
            return messages

        all_relevant_context = ""
        # React to the user prompt
        if prompt := st.chat_input("Ask a question about the project manual..."):
            if proj_manual_embeddings is not None:
                # Generate prompt embedding

                # prompt_embeddings = generate_prompt_embedding(prompt)
                prompt_embedding_array = generate_prompt_embedding_array(prompt)

                # Convert embeddings in embeddings_df for comparison
                embeddings_matrix = convert_embeddings_dataframe(proj_manual_embeddings)

                # Perform semantic search
                indices, distances = ask_book(embeddings_matrix, prompt_embedding_array, proj_manual_embeddings)

                all_relevant_context, relevant_page_numbers, most_relevant_context_index = process_search_results(all_relevant_context, indices, distances, proj_manual_chunks)

                print(f"Relevant context: {all_relevant_context}")
                print(f"Most relevant context index: {most_relevant_context_index}")
                print(f"Relevant page numbers: {relevant_page_numbers}")

                # remove duplicate page numbers where relevant info is located
                relevant_page_numbers = list(set(relevant_page_numbers))
                relevant_page_numbers.sort()

                first_relevant_page_num = relevant_page_numbers[0]

                if all_relevant_context:
                    prompt_template = """
                        Answer the following question using the context provided:
                        %Question:
                        ```
                        {question}
                        ```
                        %Context:
                        ```
                        {context}
                        ```
                        """
                    # Prepare the prompt for converse2
                    new_prompt = prompt_template.format(question = prompt, context= all_relevant_context)
                    st.session_state.proj_manual_messages.append({"role": "user", "content": new_prompt})

                    asyncio.run(chat(st.session_state.proj_manual_messages))

                display_evidence(st, first_relevant_page_num, most_relevant_context_index, proj_manual_chunks, "data/SampleProjectManual.pdf")

            else:
                st.session_state.proj_manual_messages.append({"role": "user", "content": prompt})
                asyncio.run(chat(st.session_state.proj_manual_messages))

# with tabs[1]:
#     st.write("IN PROGRESS ⚒️")
#     st.write("Once you've provided you're project manual, upload your submittals to see if it complies with the project manual")
#     uploaded_submittal = st.file_uploader("Upload your submittals", type=['pdf'])
#
#     if uploaded_submittal:
#         # check if embeddings exist locally
#         mineral_fiber_insulation_content = extract_text_from_uploaded_pdf(uploaded_submittal)
#         if os.path.exists("data/Mineral_Fiber_SpecSheet_v3.embeddings.csv"):
#             mineral_fiber_insulation_embeddings = load_embeddings("data/Mineral_Fiber_SpecSheet_v3.embeddings.csv")
#             mineral_fiber_insulation_chunks = prepare_documents_from_embeddings(mineral_fiber_insulation_embeddings)
#             st.write(mineral_fiber_insulation_embeddings)
#         else:
#             # create embeddings and chunks for uploaded pdf
#             with st.spinner("Processing file"):
#                 mineral_fiber_insulation_chunks = chunk_prompt(mineral_fiber_insulation_content)
#                 mineral_fiber_insulation_embeddings = process_chunks_to_embeddings(mineral_fiber_insulation_chunks)
#                 st.write(mineral_fiber_insulation_embeddings)
#     # print(type(proj_manual_embeddings))
#     # print(proj_manual_embeddings['embedding'].values.shape)
#
#
#     check_submittal_against_proj_manual = st.button("Check submittal against the project manual")
#     if check_submittal_against_proj_manual:
#         # Step 4: Build the search index
#         nn_proj_manual = build_search_index(proj_manual_embeddings['embedding'].values.tolist())
#
#         # Step 5: Perform semantic search
#         indices, distances = semantic_search(nn_proj_manual, mineral_fiber_insulation_embeddings['embedding'].values)
#
#         # Step 6: Generate the comparison report
#         report = create_comparison_report(mineral_fiber_insulation_chunks, proj_manual_chunks, indices, distances)
#
#         for entry in report:
#             print(f"Query: {entry['query_chunk']}")
#             print(f"Match: {entry['manual_chunk']}")
#             print(f"Similarity Score: {entry['similarity_score']:.2f}")
#             print(f"Status: {entry['status']}")
#             print("-" * 50)
#         # submittal_embedding_array = generate_prompt_embedding_array(mineral_fiber_insulation_content)
#         # embeddings_matrix = convert_embeddings_dataframe(mineral_fiber_insulation_embeddings)
#         # # Perform semantic search for the submittal content against the project manual embeddings
#         # indices, distances = ask_book(embeddings_matrix, submittal_embedding_array, mineral_fiber_insulation_embeddings)
#         #
#         # all_relevant_context, relevant_page_numbers, most_relevant_context_index = process_search_results(all_relevant_context, indices, distances, mineral_fiber_insulation_chunks)
#         #
#         # print(f"Relevant context: {all_relevant_context}")
#         # print(f"Most relevant context index: {most_relevant_context_index}")
#         # print(f"Relevant page numbers: {relevant_page_numbers}")
#         #
#         # # remove duplicate page numbers where relevant info is located
#         # relevant_page_numbers = list(set(relevant_page_numbers))
#         # relevant_page_numbers.sort()
#         #
#         # first_relevant_page_num = relevant_page_numbers[0]
#         #
#         # if all_relevant_context:
#         #     prompt_template = """
#         #         Analyze if the submittal complies with the project manual based on the following context:
#         #         %Submittal:
#         #         ```
#         #         {submittal}
#         #         ```
#         #         %Project Manual Context:
#         #         ```
#         #         {context}
#         #         ```
#         #         """
#         #     # Prepare the prompt for the LLM
#         #     new_prompt = prompt_template.format(submittal=mineral_fiber_insulation_content, context=proj_manual_content)
#         #     st.session_state.messages.append({"role": "user", "content": new_prompt})
#         #
#         #     asyncio.run(chat(st.session_state.messages))
#         #
#         # display_page(st, first_relevant_page_num, most_relevant_context_index, mineral_fiber_insulation_chunks, uploaded_submittal)

with tabs[1]:
    st.write("This feature utilizes AI agents to assume the various roles involved in the review of an RFI (Request For Information) including the architect, general contractor, engineer, etc. A response from each party will be displayed providing actionable items to consider during your RFI review.")
    uploaded_rfi = st.file_uploader("Upload your RFI", type=['pdf'])

    if uploaded_rfi is not None:
        rfi_embeddings = None
        rfi_content = ""
        reader = PdfReader(uploaded_rfi)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            rfi_content += text
            rfi_content = rfi_content.replace("\n", " ").replace("  ", " ").strip()
        st.write(rfi_content)
        st.session_state.rfi_question = rfi_content
        # if os.path.exists("data/RFIExample.embeddings.csv"):
        #     with st.spinner("Processing file"):
        #         rfi_embeddings = load_embeddings("data/RFIExample.embeddings.csv")
        #         rfi_chunks = prepare_documents_from_embeddings(rfi_embeddings)
        #         st.toast("Loaded rfi embeddings")
        #         st.write(rfi_embeddings)
        # else:
        #     # create embeddings and chunks for uploaded pdf
        #     with st.spinner("Processing file"):
        #         rfi_chunks = chunk_prompt(rfi_content)
        #         rfi_embeddings = process_chunks_to_embeddings(rfi_chunks)
        #         rfi_embeddings.to_csv("data/RFIExample.embeddings.csv")
        #         st.write(rfi_embeddings)

    if st.session_state.get("rfi_review_blueprint", None) is None:
        st.session_state.rfi_review_blueprint = RFIReviewBlueprint()

    async def run_rfi_review_blueprint(message: str, seed: int = 43) -> str:
        await sleep(3)
        llm_config["seed"] = seed
        st.session_state.rfi_review_blueprint.clear_message_history()
        await st.session_state.rfi_review_blueprint.initiate_work(message=message)
        # await st.session_state.rfi_review_blueprint.recommend_best_solution()
        return st.session_state.rfi_review_blueprint.conversation_history

    results = st.empty()
    rfi_agents = st.button("Start the agents to review the RFI", key="rfi_agent", type="primary")
    final_agent_responses=[]
    if rfi_agents:
        with st.spinner("Generating Responses"):
            task = f"Provide 3 solutions based on the following questions: {st.session_state.rfi_question}"
            print(f"{task}")
            final_agent_responses = asyncio.run(run_rfi_review_blueprint(message=task))


    if len(final_agent_responses) != 0:
        for entry in final_agent_responses:
            with st.expander(f"{entry["role"]}"):
                st.markdown(f"{entry["message"]}")
        #
        # with st.expander(f'Recommendation:'):
        #     st.markdown(f"{final_recommendation}")

