import streamlit as st
import helpers.sidebar

st.set_page_config(
    page_title="Home Page",
    #page_id="home",
    page_icon="🏠",
    layout="wide",
)

helpers.sidebar.show()

st.header("Home Page")

