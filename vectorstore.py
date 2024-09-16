import os
from langchain_chroma import Chroma
from langchain_postgres import PGVector 

def save_or_load_vectorstore(documents, embedding):
    if connection_string := os.environ.get('DATABASE_URL'):
        try:
            return PGVector.from_existing_index(
                embedding,
                collection_name="documents",
                connection=connection_string,
            )
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}")
            print("Creating new vectorstore...")
            return PGVector.from_documents(
                documents,
                embedding,
                collection_name="documents",
                connection=connection_string,
            )

    else:  # We're local
        persist_directory = "./chroma_db" # TODO switch out Chroma for PGVector locally
        if os.path.exists(persist_directory):
            print("Loading existing local vectorstore...")
            return Chroma(persist_directory=persist_directory, embedding_function=embedding)
        else:
            print("Creating new local vectorstore...")
            return Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                persist_directory=persist_directory,
            )