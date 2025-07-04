import streamlit as st
import pandas as pd
import helpers.sidebar
#import helpers.util
from helpers.ca_util import extract_data_from_submittal, extract_data_from_rfi

st.set_page_config(
    page_title="CA Log",
    page_icon="📝",
    layout="wide",
)


helpers.sidebar.show()

## ------- HEADER -------
st.header("Construction Administration - Submittal & RFI Log")
st.write("Automate your submittal and RFI logs by uploading your pdfs. The data will be formatted into a downloadable file.")

## ------- SUBMITTAL FILE UPLOAD --------
uploaded_submittals = st.file_uploader("Upload your completed submittals", type=['pdf'], accept_multiple_files=True)

if uploaded_submittals is not None:
    sub_data_list = []

    for file in uploaded_submittals:
        data = extract_data_from_submittal(file)
        sub_data_list.append(data)

    print(f'{sub_data_list}')

    sub_df = pd.DataFrame(sub_data_list, columns=['Submittal Number', 'Submittal Name', 'Date Received from GC', 'Date Sent to Engineers', 'Date Received from Engineers', 'Due Date',
                                                  'Date Returned', 'Response', 'Comments'])
    styled_sub_df = sub_df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("text-align", "center")]},
            {"selector": "td:nth-child(1)", "props": [("width", "200px")]},  # Submittal Number
            {"selector": "td:nth-child(2)", "props": [("width", "250px")]},  # Submittal Name
            {"selector": "td:nth-child(3)", "props": [("width", "150px")]},  # Date Received from GC / Issue Date
            {"selector": "td:nth-child(4)", "props": [("width", "150px")]},  # Date Sent to Engineers
            {"selector": "td:nth-child(5)", "props": [("width", "150px")]},  # Date Received from Engineers
            {"selector": "td:nth-child(6)", "props": [("width", "150px")]},  # Due Date
            {"selector": "td:nth-child(7)", "props": [("width", "150px")]},  # Date Returned
            {"selector": "td:nth-child(8)", "props": [("width", "150px")]},  # Response
            {"selector": "td:nth-child(9)", "props": [("width", "150px")]},  # Comments
        ]
    )

    st.dataframe(styled_sub_df, use_container_width=True)

    excel_file = "submittal_log.xlsx"
    sub_df.to_excel(excel_file, index=False)

    with open(excel_file, "rb") as file:
        if sub_data_list != []:
            st.download_button(
                key="submittal download",
                label="Download Submittal Excel",
                data=file,
                file_name = "submittal_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

## --------------------------------------------- RFI FILE UPLOAD ------------------------------------------------
uploaded_rfis = st.file_uploader("Upload your completed RFIs", type=['pdf'], accept_multiple_files=True)
if uploaded_rfis is not None:
    # storing data as list in dictionaries
    rfi_data_list = []

    # loop through each pdf
    for file in uploaded_rfis:
        # sort through
        data = extract_data_from_rfi(file)
        rfi_data_list.append(data)
    print(f'{rfi_data_list}')

    # Create dataframe from data_list
    rfi_df = pd.DataFrame(rfi_data_list, columns=["RFI Number", "RFI Name", "Issue Date", "Due Date"])
    styled_rfi_df = rfi_df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("text-align", "center")]},
            {"selector": "td:nth-child(1)", "props": [("width", "100px")]},  # RFI Number
            {"selector": "td:nth-child(2)", "props": [("width", "250px")]},  # RFI Name
            {"selector": "td:nth-child(3)", "props": [("width", "150px")]},  # Issue Date
            {"selector": "td:nth-child(4)", "props": [("width", "150px")]},  # Due Date
        ]
    )
    st.dataframe(styled_rfi_df, use_container_width=True)

    # Convert DataFrame to Excel
    excel_file = 'submittal_log.xlsx'
    rfi_df.to_excel(excel_file, index=False)

    # Display the Excel file for download
    with open(excel_file, 'rb') as file:
        if rfi_data_list != []:
            st.download_button(
                key="rfi download",
                label="Download Excel File",
                data=file,
                file_name="data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )