import os
from langchain_chroma import Chroma

def save_or_load_vectorstore(documents, embedding, persist_directory="./chroma_db"):
    if os.path.exists(persist_directory):
        print("Loading existing vectorstore...")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print("Creating new vectorstore...")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
        vectorstore.persist()
        return vectorstore