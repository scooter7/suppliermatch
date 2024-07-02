import google.generativeai as genai
import pandas as pd
import json
import typing_extensions
import streamlit as st

# Use Streamlit secrets for API key
api_key = st.secrets["gemini_api_key"]

# Gemini
genai.configure(api_key=api_key)
model_pandas = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="You are an expert python developer who works with pandas. You make sure to generate simple pandas 'command' for the user queries in JSON format. No need to add 'print' function. Analyse the datatypes of the columns before generating the command. If unfeasible, return 'None'. ")
model_response = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="Your task is to comprehend. You must analyse the user query and response data to generate a response data in natural language.")

# Response Schema
class Command(typing_extensions.TypedDict):
    command: str

# Streamlit
st.title('Gemini for CSV')
st.write('Talk with your CSV data using Gemini Flash!')

# Add File
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.to_list())
    dtype = str(df.dtypes.to_dict())
    
    # User Query
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
   
    if user_query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
    
        final_query = f"The dataframe name is 'df'. df has the columns {cols} and their datatypes are {dtype}. df is in the following format: {desc}. The head of df is: {head}. You cannot use df.info() or any command that cannot be printed. Write a pandas command for this query on the dataframe df: {user_query}"

        with st.spinner('Analyzing the data...'):
            response = model_pandas.generate_content(
                final_query,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=Command,
                    temperature=0.3
                )
            )

            command = json.loads(response.text)['command']
            print(command)
        try:
            exec(f"data = {command}")
            natural_response = f"The user query is {final_query}. The output of the command is {str(data)}. If the data is 'None', you can say 'Please ask a query to get started'. Do not mention the command used. Generate a response in natural language for the output."
            bot_response = model_response.generate_content(
                natural_response,
                generation_config=genai.GenerationConfig(temperature=0.7)
            )
            st.chat_message("assistant").write(bot_response.text)
            st.session_state.messages.append({"role": "assistant", "content": bot_response.text})

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": "Error"})
