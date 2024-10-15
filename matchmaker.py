import os
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
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import base64
import re
from PyPDF2 import PdfReader
from docx import Document

client = openai

CSV_URL = "https://raw.githubusercontent.com/scooter7/suppliermatch/main/docs/csv_data.csv"
GITHUB_HISTORY_URL = "https://api.github.com/repos/scooter7/suppliermatch/contents/History"

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

    st.write(css, unsafe_allow_html=True)
    header_html = """
    <div style="text-align: center;">
        <h1 style="font-weight: bold;">Strategic Insights Supplier Match</h1>
        <img src="https://oppaccess.com/integrated-research-platform/wp-content/uploads/2024/04/SI-DSjpg-2.jpg" alt="Icon" style="height:300px; width:400px;">
        <p align="left">Ask about our suppliers or upload an RFP!</p>
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

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    csv_data = get_csv_data()
    if csv_data is not None:
        text_chunks = get_text_chunks(csv_data)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
    user_question = st.text_input("Ask about our suppliers:")
    if user_question:
        handle_userinput(user_question)

def get_csv_data():
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        st.error(f"Failed to fetch CSV data: {response.status_code}, {response.text}")
        return None
    # Explicitly set encoding and strip any spaces from column names
    csv_data = pd.read_csv(BytesIO(response.content), encoding='utf-8')
    csv_data.columns = csv_data.columns.str.strip()  # Strip spaces from column headers
    st.write("CSV Columns:", csv_data.columns)  # Debugging: Display the column names
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
    
    # Use the new OpenAI API with ChatCompletion (model: gpt-4o-mini)
    openai.api_key = st.secrets["openai_api_key"]

    try:
        # Create a response using the custom API call structure you've shared
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes RFP documents."},
                {"role": "user", "content": f"Please summarize the following text with a focus on the type of work or services being requested:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        
        # Correct way to handle the response (no subscripting)
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

    # Initialize an empty list to hold the matching companies
    matching_providers = []

    try:
        for index, row in csv_data.iterrows():
            primary_industry = row['Primary Industry']
            company_name = row['Company Name']

            # Ask OpenAI to evaluate if this company is a good match based on its industry and the summarized scope
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that helps find companies for specific scopes of services based on their industries."},
                    {"role": "user", "content": f"Does the company '{company_name}' with the primary industry '{primary_industry}' fit the following scope of services?\n\n{summary}"}
                ],
                max_tokens=50,
                temperature=0.5
            )

            # Get the decision from the model's response (yes/no answer)
            decision = response.choices[0].message.content.strip().lower()

            # If OpenAI thinks the company is a good fit, add it to the list of matching providers
            if 'yes' in decision:
                matching_providers.append(row)

    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return pd.DataFrame()

    # Convert the list of matching providers back to a DataFrame
    matching_providers_df = pd.DataFrame(matching_providers)

    if matching_providers_df.empty:
        st.write("No matching companies found.")
    return matching_providers_df

def get_text_chunks(csv_data):
    # Combine all text in the 'Primary Industry' column into a single string
    text = " ".join(csv_data['Primary Industry'].fillna('').tolist())  # Adjust column selection if needed
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        # Save chat history after each interaction
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

if __name__ == '__main__':
    main()
