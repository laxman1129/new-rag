from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import shutil
import time


def load_docs(file_path):
    print("loading docs...")
    shutil.rmtree('db')
    time.sleep(10)
    loader = CSVLoader(file_path='src_docs/data.csv')
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="db")
    print("docs loaded...")



