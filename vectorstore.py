import os
from langchain_chroma import Chroma
from langchain_postgres import PGVector 

def save_or_load_vectorstore(documents, embedding):
    # if conn := os.environ.get('DATABASE_URL'):
    #     print("inside conn part")
    #     connection_string = conn.replace("postgres://", "postgresql+asyncpg://")
    #     try:
    #         print("Loading existing vectorstore...")
    #         vectorstore = PGVector.from_existing_index(
    #             embedding,
    #             collection_name="documents",
    #             connection=connection_string,
    #             async_mode=True,
    #         )
    #         print("Successfully loaded existing vectorstore.")
    #         return vectorstore

    #     except Exception as e:
    #         print(f"Error loading existing vectorstore: {e}")
    #         print("Creating new vectorstore...")
    #         vectorstore = PGVector.from_documents(
    #             documents,
    #             embedding,
    #             collection_name="documents",
    #             connection=connection_string,
    #             async_mode=True,
    #         )
    #         print("Successfully created new vectorstore.")
    #         return vectorstore

    # else:  # We're local
        print("local vectorstore mode")
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