import os
import re
import streamlit as st
import requests
import pandas as pd
import openai
from io import BytesIO
from datetime import datetime
import base64
from PyPDF2 import PdfReader
from docx import Document

# Initialize the OpenAI client
client = openai
client.api_key = st.secrets["openai_api_key"]

# Define the css variable directly
css = '''
<style>
/* Add your custom CSS styling here */
body {
    background-color: #f5f5f5;
}
/* Hide the footer */
footer {visibility: hidden;}
/* Hide the header */
header {visibility: hidden;}
</style>
'''

CSV_URL = "https://raw.githubusercontent.com/scooter7/suppliermatch/main/docs/csv_data.csv"

def main():
    # Set page config
    st.set_page_config(
        page_title="Strategic Insights Supplier Match",
        page_icon="https://strategicinsights.com/wp-content/uploads/2024/04/logo2.png"
    )
    
    # Hide the Streamlit toolbar
    hide_toolbar_css = """
    <style>
        .css-14xtw13.e8zbici0 { display: none !important; }
    </style>
    """
    st.markdown(hide_toolbar_css, unsafe_allow_html=True)

    # Use the css variable
    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Strategic Insights Supplier Match</h1>
        <img src="https://oppaccess.com/integrated-research-platform/wp-content/uploads/2024/04/SI-DSjpg-2.jpg" alt="Icon" style="height:300px; width:400px;">
        <p align="left">Upload an RFP!</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # File uploader for RFP
    uploaded_file = st.file_uploader("Upload your RFP (PDF or Word)", type=["pdf", "docx"])
    
    if uploaded_file:
        summary = summarize_rfp(uploaded_file)
        if summary:
            st.write("**Summarized Scope of Work:**")
            st.write(summary)
            matching_providers = find_matching_providers(summary)
            if not matching_providers.empty:
                st.write("**Matching Providers (Filtered Company Details):**")
                st.write(matching_providers)
            else:
                st.write("No matching companies found.")

def get_csv_data():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        st.error(f"Failed to fetch CSV data: {response.status_code}, {response.text}")
        return None
    csv_data = pd.read_csv(BytesIO(response.content), encoding='utf-8')
    # Clean up column headers
    csv_data.columns = csv_data.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    return csv_data

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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
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
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def find_matching_providers(summary):
    """Find providers in the CSV that match the summarized scope of work based on relevant industries."""
    csv_data = get_csv_data()
    if csv_data is None:
        return pd.DataFrame()  # Return an empty DataFrame if fetching the CSV failed

    # Use the exact column name
    primary_industry_column = 'Primary Industry .1'

    if primary_industry_column not in csv_data.columns:
        st.error(f"Column '{primary_industry_column}' not found in the data.")
        return pd.DataFrame()

    # Prepare companies data
    companies_data = []
    for index, row in csv_data.iterrows():
        company_name = row['Company']
        primary_industry = row[primary_industry_column]
        companies_data.append(f"{company_name}: {primary_industry}")

    # Limit the number of companies to avoid exceeding context length
    max_companies = 30  # Adjust this number based on testing
    companies_data = companies_data[:max_companies]

    companies_text = "\n".join(companies_data)

    # Estimate token usage
    total_text = f"Scope of Work:\n{summary}\n\nCompanies:\n{companies_text}"
    estimated_tokens = estimate_tokens(total_text)
    max_allowed_tokens = 7500  # Reserve some tokens for the model's response

    # Adjust the number of companies if estimated tokens exceed the limit
    while estimated_tokens > max_allowed_tokens and max_companies > 5:
        max_companies -= 5
        companies_data = companies_data[:max_companies]
        companies_text = "\n".join(companies_data)
        total_text = f"Scope of Work:\n{summary}\n\nCompanies:\n{companies_text}"
        estimated_tokens = estimate_tokens(total_text)

    if estimated_tokens > max_allowed_tokens:
        st.error("The combined length of the scope of work and companies list exceeds the maximum allowed tokens.")
        return pd.DataFrame()

    try:
        # Ask OpenAI to evaluate all companies at once
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that helps find companies suitable for specific scopes of services based on their industries. "
                        "Provide a list of matching companies from the provided list, based on the scope of work. "
                        "List the company names only, separated by commas."
                    )
                },
                {
                    "role": "user",
                    "content": f"Scope of Work:\n{summary}\n\nCompanies:\n{companies_text}"
                }
            ],
            max_tokens=500,
            temperature=0.5
        )

        response_text = response.choices[0].message.content.strip()

        # Extract the matching company names from the response
        available_company_names = [cd.split(':')[0].strip() for cd in companies_data]
        matching_companies = [name.strip() for name in response_text.split(',') if name.strip() in available_company_names]

        if matching_companies:
            st.write(f"**Matching Companies:** {', '.join(matching_companies)}")

            # Filter the CSV DataFrame by matching company names
            matching_providers_df = csv_data[csv_data['Company'].isin(matching_companies)]

            if matching_providers_df.empty:
                st.write("No matching companies found.")
            return matching_providers_df
        else:
            st.write("No matching companies extracted from the response.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return pd.DataFrame()

def estimate_tokens(text):
    """Estimate the number of tokens in a text using tiktoken."""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except ImportError:
        # Fallback estimation: assume 1 token per 4 characters
        return len(text) / 4

if __name__ == '__main__':
    main()
