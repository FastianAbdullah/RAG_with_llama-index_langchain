import streamlit as st

groq_api_key = st.secrets["GROQ_API_KEY"]
user_agent = st.secrets["USER_AGENT"]

print(f"GROQ_API_KEY: {groq_api_key}")
print(f"USER_AGENT: {user_agent}")
