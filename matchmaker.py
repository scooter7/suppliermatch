import os
import re
import streamlit as st
import requests
import pandas as pd
import openai
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from datetime import datetime
import base64

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

    # Load CSV data at the start to make it queryable later
    csv_data = get_csv_data()

    # File uploader for RFP
    uploaded_file = st.file_uploader("Upload your RFP (PDF or Word)", type=["pdf", "docx"])
    
    if uploaded_file:
        summary = summarize_rfp(uploaded_file)
        if summary:
            st.write("Summarized Scope of Work:")
            st.write(summary)
            matching_providers = find_matching_providers(summary, csv_data)
            st.write("Matching Providers (Filtered Company Details):")
            st.write(matching_providers)

    # Add supplier query functionality
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if csv_data is not None:
        text_chunks = get_text_chunks(csv_data)
        if text_chunks:
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
    
    user_question = st.text_input("Ask about our suppliers:")
    if user_question:
        handle_userinput(user_question, csv_data)

def get_csv_data():
    """Fetch CSV data from the GitHub repository."""
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        st.error(f"Failed to fetch CSV data: {response.status_code}, {response.text}")
        return None
    csv_data = pd.read_csv(BytesIO(response.content), encoding='utf-8')
    csv_data.columns = csv_data.columns.str.strip()  # Clean up column headers
    return csv_data

def summarize_rfp(uploaded_file):
    """Summarize the RFP document using OpenAI."""
    text = extract_file_text(uploaded_file)
    if not text:
        st.error("No text found in the uploaded file.")
        return None
    
    openai.api_key = st.secrets["openai_api_key"]

    try:
        response = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes RFP documents."},
                {"role": "user", "content": f"Please summarize the following text with a focus on the type of work or services being requested:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        
        summary = response.choices[0].message["content"].strip()
        return summary

    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return None

def extract_file_text(file):
    """Extract text from PDF or DOCX file."""
    if file.type == "application/pdf":
        return extract_pdf_text(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_docx_text(file)
    return None

def extract_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_docx_text(docx_file):
    """Extract text from a DOCX file."""
    from docx import Document
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def find_matching_providers(summary, csv_data):
    """Find providers in the CSV that match the summarized scope of work based on relevant industries."""
    openai.api_key = st.secrets["openai_api_key"]

    companies_data = []
    for index, row in csv_data.iterrows():
        company_name = row['Company']
        primary_industry = row['Primary Industry']
        companies_data.append(f"{company_name}: {primary_industry}")

    companies_text = "\n".join(companies_data)

    try:
        # Ask OpenAI to evaluate all companies at once
        response = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that helps find companies for specific scopes of services based on their industries."},
                {"role": "user", "content": f"Given the following list of companies and their industries, determine which companies would be a good fit for the following scope of work:\n\n{summary}\n\nCompanies:\n{companies_text}"}
            ],
            max_tokens=1000,
            temperature=0.5
        )

        # Extract the response content
        response_text = response.choices[0].message["content"].strip()

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

def get_text_chunks(csv_data):
    """Split the CSV data into text chunks for processing."""
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

def handle_userinput(user_question, csv_data):
    """Handle user input and query CSV data for supplier questions."""
    if 'conversation' in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            modified_content = modify_response_language(message.content)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", modified_content), unsafe_allow_html=True)
        save_chat_history(st.session_state.chat_history)
    else:
        st.error("The conversation model is not initialized. Please wait until the model is ready.")

def modify_response_language(original_response):
    """Modify the language of the response to be more conversational."""
    response = original_response.replace(" they ", " we ")
    response = original_response.replace("They ", "We ")
    response = original_response.replace(" their ", " our ")
    response = original_response.replace("Their ", "Our ")
    response = original_response.replace(" them ", " us ")
    response = original_response.replace("Them ", "Us ")
    return response

def save_chat_history(chat_history):
    """Save the chat history to GitHub."""
    github_token = st.secrets["github"]["access_token"]
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"chat_history_{date_str}.txt"
    chat_content = "\n\n".join(f"{'User:' if i % 2 == 0 else 'Bot:'} {message.content}" for i, message in enumerate(chat_history))
    
    encoded_content = base64.b64encode(chat_content.encode('utf-8')).decode('utf-8')
    data = {
        "message": f"Save chat history on {date_str}",
        "content": encoded_content,
        "branch": "main"
    }
    response = requests.put(f"{GITHUB_HISTORY_URL}/{file_name}", headers=headers, json=data)
    if response.status_code == 201:
        st.success("Chat history saved successfully.")
    else:
        st.error(f"Failed to save chat history: {response.status_code}, {response.text}")

if __name__ == '__main__':
    main()
