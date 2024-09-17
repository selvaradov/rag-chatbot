from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import json
import uuid
import os


def process_csv_file(file_path):
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for i, row in df.iterrows():
        timeframe = row["Date"]
        for column in df.columns[1:]:  # Skip the Date column
            content = row[column]
            if pd.isna(content) or content == "":
                continue

            assert isinstance(
                content, str
            ), f"Content for column {column} in row {i} is not a string: {content}"

            chunks = text_splitter.split_text(content)

            for j, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path,
                    "table": file_name.replace(".csv", ""),
                    "topic": column,
                    "timeframe": timeframe,
                    "row": i,
                    "row_data": json.dumps(
                        row.to_dict()
                    ),  # Store entire row for context
                    "id": f"{file_name}_{column}_{timeframe}_{j}",  # putting it into metadata too because seems to not be working well as id property on its own
                }
                enriched_chunk = f"<meta>\n<timeframe>\n{timeframe}\n</timeframe>\n<topic>\n{column}\n</topic>\n</meta>\n{chunk}"
                doc = Document(
                    page_content=enriched_chunk,
                    metadata=metadata,
                    id=f"{file_name}_{column}_{timeframe}_{j}",
                )
                documents.append(doc)

    return documents


def process_csv_dir(dir_path):
    all_documents = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                documents = process_csv(file_path)
                all_documents.extend(documents)
    return all_documents


def process_csv(file_or_dir):
    if os.path.isfile(file_or_dir):
        return process_csv_file(file_or_dir)
    elif os.path.isdir(file_or_dir):
        return process_csv_dir(file_or_dir)
    else:
        raise ValueError("The path provided is neither a file nor a directory.")


def process_csv_dir_vanilla(dir_path):
    loader = DirectoryLoader(dir_path, glob="**/*.csv", loader_cls=CSVLoader)
    documents = loader.load()
    documents_with_ids = [
        Document(
            page_content=doc.page_content,
            metadata=doc.metadata,
            id=str(uuid.uuid4()),  # Set the ID parameter
        )
        for doc in documents
    ]

    return documents_with_ids


def process_unstructured(path):
    if os.path.isfile(path):
        loader = TextLoader(path)
    elif os.path.isdir(path):
        loader = DirectoryLoader(
            path, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader
        )
    else:
        raise ValueError("The path provided is neither a file nor a directory.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs = splitter.split_documents(documents)

    for doc in split_docs:
        file_name = os.path.basename(doc.metadata["source"])
        start_index = doc.metadata.get("start_index", 0)
        doc_id = f"{file_name}_{start_index}"
        doc.id = doc_id
        doc.metadata["id"] = doc_id

    return split_docs
