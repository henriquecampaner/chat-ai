import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
  page_title="Your virtual assistant",
  page_icon="🤖",
)
st.title("Your virtual assistant 🤖")

model_class = "openai" # "hf_hub", "openai", "ollama"

def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
    llm = HuggingFaceEndpoint(
        repo_id = model,
        temperature = temperature, 
        max_new_tokens = 512,
        return_full_text = False,
        stop= ["<|eot_id|>"],
    )

    return llm

def model_openai(model = "gpt-4o-mini", temperature = 0.1):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
    )
    return llm

def model_ollama(model = "phi3", temperature = 0.1):
    llm = ChatOllama(model = model, temperature = temperature)
    return llm

def model_response(user_query, chat_history, model_class):
    # Load LLM
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    # Create prompt
    system_prompt = """
        You are a helpful assistant that answers general questions. Respond in {language}.
    """

    language = "English"

    # Pipeline
    if model_class.startswith("hf"):
      user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt),
    ])

    # Chain
    chain = prompt_template | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you today?")]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.write(message.content)

user_query = st.chat_input("Type your question")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        response = st.write_stream(model_response(
            user_query,
            st.session_state.chat_history,
            model_class,
        ))
    st.session_state.chat_history.append(AIMessage(content=response))