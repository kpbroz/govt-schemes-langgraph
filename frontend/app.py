import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/ask/"

st.title("Interactive Query Assistant")
st.write("Ask me anything!")

# Input box
user_query = st.text_input("Your question:")

if st.button("Submit"):
    if user_query.strip():
        # Send request to the backend
        response = requests.post(API_URL, json={"question": user_query})
        if response.status_code == 200:
            res_json = response.json()
            st.success(f"Response: {res_json['response']}")
        else:
            st.error("Error communicating with backend.")
    else:
        st.warning("Please enter a question.")

