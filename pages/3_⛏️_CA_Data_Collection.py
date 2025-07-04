import os
import streamlit as st
import pandas as pd
import helpers.sidebar
import pdfplumber
import matplotlib.pyplot as plt
#import helpers.util
from helpers.ca_util import extract_rfi_list_from_dir, get_all_files_from_folder, extract_rfi_data

st.set_page_config(
    page_title="Construction Document Finder",
    page_icon="📝",
    layout="wide",
)


helpers.sidebar.show()

## ------- HEADER -------
st.header("Construction Administration - Data Extraction")
st.write("Work in Progress ⚙️")

## ------- Project Directory Selection --------
# select the project folder

# proj_dir = st.text_input(label="Project Directory", key="project_directory", )
#
# # st.write(proj_dir)
#
# file_list = get_all_files_from_folder(proj_dir)
# st.write(file_list)
#
# project_rfi_list = extract_rfi_list_from_dir(proj_dir)
# #st.write(project_rfi_list)
#
# project_rfi_list_df = pd.DataFrame(project_rfi_list, columns=["Project", "Project Address", "General Contractor", "Architect Team", "Project Management Team", "GC Team", "MEP Consultants", "Consultants", "RFI Number", "RFI Name", "Issue Date", "Due Date", "Question"])
# st.dataframe(project_rfi_list_df)
#
# uploaded_file = st.file_uploader("Upload file", type=['pdf'])
# if uploaded_file is not None:
#     extract_rfi_data(uploaded_file)

    # with pdfplumber.open(uploaded_file) as pdf:
    #     page = pdf.pages[0]
    #
    #     bounding_box = (180, 0, 350, 90)
    #     bb_region = page.within_bbox(bounding_box)
    #     text_in_bb = bb_region.extract_text()
    #
    #     st.write(text_in_bb)
    #
    #     im = page.to_image(resolution = 150)
    #
    #     im.draw_rect(bounding_box, stroke="green", stroke_width=2)
    #
    #     st.image(im.annotated, caption="PDF with bounding box")
    #     # fig = im.debug_tablefinder()  # or .debug_word() to see word boxes
    #     # st.pyplot(fig)

# styled_project_rfi_list_df = project_rfi_list_df.set_table_styles(
#     {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
#     {"selector": "td", "props": [("text-align", "center")]},
#     {"selector": "td:nth-child(1)", "props": [("width", "200px")]},  # Project
#     {"selector": "td:nth-child(2)", "props": [("width", "250px")]},  # Project Address
#     {"selector": "td:nth-child(3)", "props": [("width", "150px")]},  # General Contractor
#     {"selector": "td:nth-child(4)", "props": [("width", "150px")]},  # Architect Team
#     {"selector": "td:nth-child(5)", "props": [("width", "150px")]},  # Project Management Team
#     {"selector": "td:nth-child(6)", "props": [("width", "150px")]},  # GC Team
#     {"selector": "td:nth-child(7)", "props": [("width", "150px")]},  # MEP Consultants
#     {"selector": "td:nth-child(8)", "props": [("width", "150px")]},  # Consultants
#     {"selector": "td:nth-child(9)", "props": [("width", "150px")]},  # RFI Number
#     {"selector": "td:nth-child(10)", "props": [("width", "150px")]},  # RFI Name
#     {"selector": "td:nth-child(11)", "props": [("width", "150px")]},  # Issue Date
#     {"selector": "td:nth-child(12)", "props": [("width", "150px")]},  # Due Date
#     {"selector": "td:nth-child(13)", "props": [("width", "150px")]},  # Question
# )


# st.dataframe(styled_project_rfi_list_df, use_container_width=True)
#
# excel_file = 'project_rfi_list.xlsx'
#
# project_rfi_list_df.to_excel(excel_file, index=False)
#
# with open(excel_file, "rb") as file:
#     if project_rfi_list != []:
#         st.download_button(
#             key = "project_rfi_list download",
#             label = "Download Project RFI List",
#             data = file,
#             file_name = "project_rfi_list.xlsx",
#             mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )
