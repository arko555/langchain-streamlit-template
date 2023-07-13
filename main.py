"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

import os
from dotenv import load_dotenv

load_dotenv('config.env')

def load_chain(query):
    """Logic for loading the chain you want to use should go here."""

    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),  # get yours from pinecone.io. there is a free tier.
        environment=os.environ.get("pinecone_env")
)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Pinecone.from_existing_index('pdf-bot', embedding=embeddings)
    
    repo_ids = ["facebook/mbart-large-50","google/flan-t5-base","google/flan-t5-large"]
    #### Loading LLM ###
    model = HuggingFaceHub(repo_id=repo_ids[1],
                       model_kwargs={"temperature": 0, "max_length":200},
                       huggingfacehub_api_token=os.environ.get('HUGGING_FACE_API_KEY'))
    
    sources_chain = load_qa_with_sources_chain(model, chain_type="stuff")
    documents = vectorstore.similarity_search(query)
    result = sources_chain.run(input_documents=documents, question=query)

    return result

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = load_chain(query=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
