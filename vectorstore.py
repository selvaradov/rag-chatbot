import pickle
from typing import Union

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from sqlalchemy import text
from load_data import process_csv, process_unstructured, get_metadata_options

import dotenv

dotenv.load_dotenv()

# Connection string for the local PostgreSQL database - in deployment get from DATABASE_URL
CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME = "airtable"


async def check_collection_exists(connection: AsyncConnection, collection_name: str):
    collection_exists = False
    table_exists = (
        await connection.execute(
            text("""
        SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename = 'langchain_pg_collection'
        );
        """),
        )
    ).scalar()
    print(f"Table exists: {table_exists}")
    if not table_exists:
        return False

    collection_exists = (
        await connection.execute(
            text("""
        SELECT EXISTS (
            SELECT 1
            FROM langchain_pg_collection
            WHERE name = :collection_name
        );
        """),
            {"collection_name": collection_name},
        )
    ).scalar()

    print(f"Collection exists: {collection_exists}")

    return bool(collection_exists)


async def setup_vectorstore(
    csv_path: str,
    unstructured_path: Union[None, str],
    load_qa: bool,
    qa_path: str,
    metadata_options_path: str,
):
    engine = create_async_engine(CONNECTION_STRING)
    async with engine.connect() as connection:
        collection_exists = await check_collection_exists(connection, COLLECTION_NAME)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    if not collection_exists:
        # Load and process raw documents
        csv_docs = process_csv(csv_path)

        if unstructured_path:
            unstructured_docs = process_unstructured(unstructured_path)
            all_input_docs = csv_docs + unstructured_docs
        else:
            all_input_docs = csv_docs

        qa_docs = []
        if load_qa:
            with open(qa_path, "rb") as f:
                # Second value are the failed pairs but they're not needed here (and empty anyway)
                qa_docs, _ = pickle.load(f)

        all_docs = qa_docs + all_input_docs
        metadata_options = get_metadata_options(all_docs)

        with open(metadata_options_path, "wb") as f:
            pickle.dump(metadata_options, f)

        # Create and populate the vector store
        vectorstore = await PGVector.afrom_documents(
            documents=all_docs,
            embedding=embeddings,
            connection=engine,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        print(
            f"Successfully created collection '{COLLECTION_NAME}' and populated it with documents."
        )

    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Loading it...")
        vectorstore = await PGVector.afrom_existing_index(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=engine,
            use_jsonb=True,
        )
        metadata_options = pickle.load(open(metadata_options_path, "rb"))

    return vectorstore, metadata_options
