import os
import re  # Import re for regular expression matching
import streamlit as st
import requests
import pandas as pd
import openai
from io import BytesIO
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css
from datetime import datetime
import base64
from PyPDF2 import PdfReader
from docx import Document

client = openai

CSV_URL = "https://raw.githubusercontent.com/scooter7/suppliermatch/main/docs/csv_data.csv"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/suppliermatch/contents/History"

def main():
    # Set page config
    st.set_page_config(
        page_title="Strategic Insights Supplier Match",
    )
    
    # Hide the Streamlit toolbar
    hide_toolbar_css = """
    <style>
        .css-14xtw13.e8zbici0 { display: none !important; }
    </style>
    """
    st.markdown(hide_toolbar_css, unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Strategic Insights Supplier Match</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # File uploader for RFP
    uploaded_file = st.file_uploader("Upload your RFP (PDF or Word)", type=["pdf", "docx"])
    
    if uploaded_file:
        summary = summarize_rfp(uploaded_file)
        if summary:
            st.write("Summarized Scope of Work:")
            st.write(summary)
            matching_providers = find_matching_providers(summary)
            st.write("Matching Providers (Filtered Company Details):")
            st.write(matching_providers)

def summarize_rfp(uploaded_file):
    """Summarize the RFP document using OpenAI."""
    # Extract the text from PDF or Word file
    if uploaded_file.type == "application/pdf":
        text = extract_pdf_text(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx_text(uploaded_file)
    else:
        return None

    # Check if the extracted text is available
    if not text:
        st.error("No text found in the uploaded file.")
        return None
    
    openai.api_key = st.secrets["openai_api_key"]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes RFP documents."},
                {"role": "user", "content": f"Please summarize the following text with a focus on the type of work or services being requested:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        
        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return None

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def find_matching_providers(summary):
    """Find providers in the CSV that match the summarized scope of work based on relevant industries."""
    csv_data = get_csv_data()
    if csv_data is None:
        return pd.DataFrame()  # Return an empty DataFrame if fetching the CSV failed

    openai.api_key = st.secrets["openai_api_key"]

    companies_data = []
    for index, row in csv_data.iterrows():
        company_name = row['Company']
        primary_industry = row['Primary Industry']
        companies_data.append(f"{company_name}: {primary_industry}")

    companies_text = "\n".join(companies_data)

    try:
        # Ask OpenAI to evaluate all companies at once
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that helps find companies for specific scopes of services based on their industries."},
                {"role": "user", "content": f"Given the following list of companies and their industries, determine which companies would be a good fit for the following scope of work:\n\n{summary}\n\nCompanies:\n{companies_text}"}
            ],
            max_tokens=1000,
            temperature=0.5
        )

        # Extract the response content
        response_text = response.choices[0].message.content.strip()

        # Use regex to extract company numbers or names (assuming the company numbers/names are in the form "Company X")
        matching_companies = re.findall(r'Company\s*(\d+)', response_text)

        if matching_companies:
            st.write(f"Matching Companies: {matching_companies}")

            # Filter the CSV DataFrame by matching company numbers
            matching_providers_df = csv_data[csv_data.index.isin([int(company_num) - 1 for company_num in matching_companies])]

            if matching_providers_df.empty:
                st.write("No matching companies found.")
            return matching_providers_df
        else:
            st.write("No matching companies extracted from the response.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return pd.DataFrame()

def get_csv_data():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        st.error(f"Failed to fetch CSV data: {response.status_code}, {response.text}")
        return None
    csv_data = pd.read_csv(BytesIO(response.content), encoding='utf-8')
    csv_data.columns = csv_data.columns.str.strip()  # Strip spaces from column headers
    return csv_data

if __name__ == '__main__':
    main()
