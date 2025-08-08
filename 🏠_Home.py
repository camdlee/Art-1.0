import streamlit as st
import streamlit_authenticator as stauth
import helpers.sidebar

st.set_page_config(
    page_title="Home Page",
    #page_id="home",
    page_icon="🏠",
    layout="wide",
)

helpers.sidebar.show()

st.header("Home Page")
st.write("This is a prototype application to support daily tasks in architecture and interior design.")

st.write("The goal of this application is to leverage the power of large language models to automate or streamline daily tasks such as reviewing RFIs (Requests For Information) or project manuals. On the sidebar to the left, you'll see different areas where Art can support your design work.")
st.write("The current LLM model this application uses OpenAI's API gpt-4-turbo-preview.")

st.divider()

st.subheader("Overview of each feature:")
features_list = [
                 "CA Log - Automate documentation of submittals and RFIs",
                 "CA Review - Use AI to assist reviewing construction documents"
                 ]

st.markdown("- " + "\n- ".join(features_list))
# st.write(hashed)
# print(hashed)

#  $2b$12$dwVsMGU3s7cHLqtVg9G3/enK2G8p/13IFhQMYoian0iqFE.78NzSC