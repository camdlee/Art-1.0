import streamlit as st

def show() -> None:
    with st.sidebar:

        reload_button = st.button('Reload Page')
        if reload_button:
            st.session_state.clear()
            st.rerun()