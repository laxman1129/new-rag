from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import time

def query_docs():
    print("RAG initiated...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="db")
    retriever = vectorstore.as_retriever()

    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/laxmanchari/workspace/genAI/models/Phi-3-mini-4k-instruct-q4.gguf",
        n_gpu_layers=33,
        n_batch=512,
        temperature=0,
        top_p=10,
        verbose=True,
        n_ctx=4096
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    while True:
        query = input('\n Please enter a question : ')
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
        print(f"::::  Elapsed time: {elapsed_time} seconds")
        print("-" * 50)
        answer = res['result']
        print(f"{answer}")
        print("-" * 50)
