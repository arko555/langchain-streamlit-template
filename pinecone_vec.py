from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
import pinecone 
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv('config.env')

my_loader = DirectoryLoader('./research_papers', glob='**/*.pdf')
documents = my_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print("Loaded embedddings")

index_name = 'pdf-bot'
dimension=384

pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),  # get yours from pinecone.io. there is a free tier.
        environment=os.environ.get("pinecone_env")
)

# delete index if it exists
if index_name not in pinecone.list_indexes():

# create index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=dimension)
    print("index creation completed")

db = Pinecone.from_documents(docs, embeddings, index_name=index_name)
print("completed vector DB creation")