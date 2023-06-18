from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
import pinecone 
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


my_loader = DirectoryLoader('./research_papers', glob='**/*.pdf')
documents = my_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#pinecone.init(
#    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
#    environment=os.environ['PINECONE_ENV']  # next to api key in console
#)

#db = Pinecone.from_documents(docs, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

db = FAISS.from_documents(docs, embeddings)

query = "what is attention mechanism"
docs = db.similarity_search(query)
print(docs[0].page_content)