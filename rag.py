from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import time


def phi_q4():
    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/laxmanchari/workspace/genAI/models/Phi-3-mini-4k-instruct-q4.gguf",
        n_gpu_layers=33,
        n_batch=512,
        temperature=0,
        top_p=10,
        verbose=False,
        n_ctx=4096
    )
    return llm


def phi_f16():
    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/laxmanchari/workspace/genAI/models/Phi-3-mini-4k-instruct-fp16.gguf",
        n_gpu_layers=33,
        n_batch=512,
        temperature=0,
        top_p=10,
        verbose=False,
        n_ctx=4096
    )
    return llm


def llama_q5():
    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/laxmanchari/workspace/genAI/models/Meta-Llama-3-8B-Instruct.Q5_1.gguf",
        n_gpu_layers=33,
        n_batch=512,
        temperature=0,
        top_p=10,
        verbose=False,
        n_ctx=4096
    )
    return llm


def llama_q8():
    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/laxmanchari/workspace/genAI/models/Meta-Llama-3-8B-Instruct.Q5_1.gguf",
        n_gpu_layers=33,
        n_batch=512,
        temperature=0,
        top_p=10,
        verbose=False,
        n_ctx=4096
    )
    return llm


def query_docs():
    print("RAG initiated...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="db")
    retriever = vectorstore.as_retriever()
    query = input('\n Please enter a question : ')

    evaluate(phi_q4(), retriever, "phi_q4", query)
    evaluate(phi_f16(), retriever, "phi_f16", query)
    evaluate(llama_q5(), retriever, "llama_q5", query)
    evaluate(llama_q8(), retriever, "llama_q8", query)


def evaluate(llm, retriever, name, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    if query.strip() == 'exit':
        exit(0)
    prompt = f"""
                <|system|>
                You are an AI assistant that follows instruction extremely well.
                Please be truthful and give direct answers
                </s>
                <|user|>
                {query}
                </s>
                <|assistant|>
    """
    start_time = time.time()
    res = qa(prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{name} ::::  Elapsed time: {elapsed_time} seconds")
    print("-" * 50)
    answer = res['result']
    print(f"{answer}")
    print("-" * 50)

