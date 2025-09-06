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
from aitools_autogen.blueprint_RFI_review import RFIReviewBlueprint
from aitools_autogen.config import llm_config_openai as llm_config
from aitools_autogen.utils import clear_working_dir
from streamlit_file_browser import st_file_browser

from helpers import util
from helpers.pdf_utils import convert_pdf_page_to_img, convert_saved_pdf_page_to_img, display_pdf_page, extract_text_from_saved_pdf, extract_text_from_uploaded_pdf, process_pdf_upload
from helpers.embeddings_util import create_comparison_report, semantic_search, analyze_with_ai, \
    load_or_process_embeddings, build_search_index, generate_prompt_embedding_array, convert_embeddings_dataframe, \
    ask_book, process_search_results, prepare_documents_from_embeddings, chunk_prompt, \
    process_chunks_to_embeddings

from services import prompts

st.set_page_config(
    page_title="AI Powered CA Review",
    page_icon="🔍",
    layout="wide"
)
## --------------------------------------------- SIDEBAR ------------------------------------------------
helpers.sidebar.show()

## --------------------------------------------- SESSION STATES ------------------------------------------------

if "state_messages" not in st.session_state:
    initial_messages = [{"role": "system",
                         "content": prompts.quick_chat_system_prompt()}]
    st.session_state.pdf_messages = initial_messages
if "all_relevant_context" not in st.session_state:
    st.session_state.all_relevant_context = ""

async def chat(messages, state_key = "state_messages"):
    """Handles user/assistant conversation given a state_key"""
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages_placeholder = st.empty()
        messages = await util.run_conversation(messages, messages_placeholder)
        st.session_state[state_key] = messages

    return messages

def render_messages(messages):
    for msg in messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

## --------------------------------------------- HEADER ------------------------------------------------
st.header("Construction Document Review")
st.write("***Uploaded your construction documents and leverage AI to review your documents***")

# ---------------------------------------- TABS --------------------------------------
tabs = st.tabs(['PDF / Project Manual Review', 'RFI Review'])

## --------------------------------------------- PROJECT MANUAL QUESTIONS ------------------------------------------------
with tabs[0]:
    st.write("Ask questions about your large pdf file to save time sifting through to find answers!\n This tab applies semantic searching to review your project manual and find the most pertinent answer to your question.")
    with st.expander("Instructions"):
        st.info(
            """
            1. Upload your project manual by clicking the “Browse files” button.
                - The system will process the PDF and create embeddings for AI search.
            2. Ask a question using the chat box labeled “Ask a question about the pdf...”
            3. View AI response. The AI will find the most relevant pages and display an answer in the chat. A preview of the relevant page from the PDF will also be shown so you can verify context.
            """
        )
    ## --------------------------------------------- FILE UPLOAD ------------------------------------------------
    uploaded_file = st.file_uploader("Please provide your pdf", type=['pdf'])

    pdf_embeddings, pdf_chunks = process_pdf_upload(uploaded_file)

    if uploaded_file is not None:
        st.write("Ask any general question from your pdf")

        # Print all messages in the session state
        render_messages(st.session_state.pdf_messages)

        all_relevant_context = ""
        # React to the user prompt
        if prompt := st.chat_input("Ask a question about the pdf..."):
            if pdf_embeddings is not None:
                # Generate prompt embedding
                prompt_embedding_array = generate_prompt_embedding_array(prompt)

                # Convert embeddings in embeddings_df for comparison
                embeddings_matrix = convert_embeddings_dataframe(pdf_embeddings)

                # Perform semantic search
                indices, distances = ask_book(embeddings_matrix, prompt_embedding_array, pdf_embeddings)

                all_relevant_context, relevant_page_numbers, most_relevant_context_index = process_search_results(all_relevant_context, indices, distances, pdf_chunks)

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
                    st.session_state.pdf_messages.append({"role": "user", "content": new_prompt})

                    asyncio.run(chat(st.session_state.pdf_messages))

                display_pdf_page(first_relevant_page_num, most_relevant_context_index, pdf_chunks, "data/SampleProjectManual.pdf")

            else:
                st.session_state.pdf_messages.append({"role": "user", "content": prompt})
                asyncio.run(chat(st.session_state.pdf_messages))

# with tabs[1]:
#     st.write("IN PROGRESS ⚒️")
#     st.write("Once you've provided you're project manual, upload your submittals to see if it complies with the project manual")
#     uploaded_submittal = st.file_uploader("Upload your submittals", type=['pdf'])
#
#     if uploaded_submittal:
#         # check if embeddings exist locally
#         mineral_fiber_insulation_content = extract_text_from_uploaded_pdf(uploaded_submittal)
#         if os.path.exists("data/Mineral_Fiber_SpecSheet_v3.embeddings.csv"):
#             mineral_fiber_insulation_embeddings = load_or_process_embeddings("data/Mineral_Fiber_SpecSheet_v3.embeddings.csv")
#             mineral_fiber_insulation_chunks = prepare_documents_from_embeddings(mineral_fiber_insulation_embeddings)
#             st.write(mineral_fiber_insulation_embeddings)
#         else:
#             # create embeddings and chunks for uploaded pdf
#             with st.spinner("Processing file"):
#                 mineral_fiber_insulation_chunks = chunk_prompt(mineral_fiber_insulation_content)
#                 mineral_fiber_insulation_embeddings = process_chunks_to_embeddings(mineral_fiber_insulation_chunks)
#                 st.write(mineral_fiber_insulation_embeddings)
#
#
#     check_submittal_against_pdf = st.button("Check submittal against the project manual")
#     if check_submittal_against_pdf:
#         # Step 4: Build the search index
#         nn_pdf = build_search_index(pdf_embeddings['embedding'].values.tolist())
#
#         # Step 5: Perform semantic search
#         indices, distances = semantic_search(nn_pdf, mineral_fiber_insulation_embeddings['embedding'].values)
#
#         # Step 6: Generate the comparison report
#         report = create_comparison_report(mineral_fiber_insulation_chunks, pdf_chunks, indices, distances)
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
#         # display_pdf_page(first_relevant_page_num, most_relevant_context_index, mineral_fiber_insulation_chunks, uploaded_submittal)

with tabs[1]:
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = []

    st.write("Deploy AI agents to assume the various roles involved in the review of an RFI (Request For Information) including the architect, general contractor, engineer, etc!")
    with st.expander("Instructions"):
        st.info(
            """
            1. Type your RFI questions below.
            2. Click the 'Start the agents to review the RFI' button.
            3. Expand each agent's response to see recommendations and actionable items.
            """
        )
    with st.expander("Sample RFI questions to use"):
        st.write(
            """
            1. **Existing drywall chase not shown in drawings**
                - There is an existing drywall chase around column 3 that is not shown on the demo drawing D1105 or shown graphically on drawing IN1105. The chase size that is currently there is 63”wide x 19-1/2” deep. It is covering an existing drywall column that is 33-1/2”wide x 13” deep. GC suggests that the existing larger chase be demo'd and patched to make paint ready. Please advise on how to proceed.
            2. **Ceiling conflict with existing chiller pipe**
                - Please see attached markup and photos for your reference. Drawing D210_E calls for the two bulkheads (running plan west-east) in records storage to be demoed. These bulkheads conceal existing chiller pipe, structural beams, ductwork, etc. that are installed lower than typ. ceiling height. Demoing these bulkheads as shown in D210_E presents several logistical problems with relocating existing ceiling devices and will incur high cost. Part 1: General Contractor recommends leaving the existing bulkheads to remain up to the back of the interior oﬃces (see highlighted blue area in attached markup). This would allow all existing ceiling equipment/ductwork to be concealed by the drywall bulkheads without requiring relocation. The bulkheads from that point forward in the interior oﬃces can be demoed as they do not conceal any ceiling equipment (see highlighted purple area in attached markup). General Contractor recommends installing a drywall bulkhead in the rear of the IO oﬃces plan north-south (see highlighted green area in attached markup) for ceiling consistency in all four oﬃces (1067-1070). Please note that there could be added costs for drywall bulkhead installation. Part 2: Per Architect ﬁeld report 11.26.2024 and site walk with General Contractor and Architect on 11.25.2024, a ceiling height conﬂict was noted with a base building chiller pipe. Architect proposed lowering all ceilings in the records storage room. The door for oﬃce 1079 was noted to be in conﬂict with the lowered ceiling. The options are to create a pocket in the ceiling for the door or cut the door at the top. Please conﬁrm which direction to proceed in to avoid conﬂicts with the ceiling and chiller pipe and note that the proposed solutions above could present added costs. Please provide updated an updated RCP, demo RCP, lighting plan, and mechanical plan for both parts of this RFI noted above.
            """
        )

    rfi_content = st.chat_input("Input your RFI questions here")
    if rfi_content:
        st.session_state.rfi_question = rfi_content

    if st.session_state.get("rfi_review_blueprint", None) is None:
        st.session_state.rfi_review_blueprint = RFIReviewBlueprint()

    async def run_rfi_review_blueprint(message: str, seed: int = 43) -> str:
        await sleep(3)
        llm_config["seed"] = seed
        st.session_state.rfi_review_blueprint.clear_message_history()
        await st.session_state.rfi_review_blueprint.initiate_work(message=message)
        return st.session_state.rfi_review_blueprint.conversation_history

    results = st.empty()
    final_agent_responses=[]
    if st.button("Start the agents to review the RFI", key="rfi_agent", type="primary"):
        if st.session_state.get("rfi_question"):
            with st.spinner("Generating Responses"):
                task = f"Provide 3 solutions based on the following questions: {st.session_state.rfi_question}"
                print(f"{task}")
                final_agent_responses = asyncio.run(run_rfi_review_blueprint(message=task))


    if final_agent_responses:
        for entry in final_agent_responses:
            with st.expander(f'{entry["role"]}'):
                st.markdown(f'{entry["message"]}')