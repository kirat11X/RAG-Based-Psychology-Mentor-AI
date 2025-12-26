import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader, JSONLoader
from langchain_core.documents import Document

from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import json

def load_ndjson(file_path):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                docs.append(Document(page_content=str(data), metadata={"source": file_path}))
    return docs
# ...existing c

from populate_dataset import clear_database

CHROMA_PATH = "chroma"
DATA_PATH="data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--data-paths",
        nargs="+",
        default=["data"],
        help="List of folders or PDF files to load.",
    )
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(args.data_paths)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
def load_documents(data_paths):
    all_docs = []
    for path in data_paths:
        # Load all PDFs in the directory
        loader = PyPDFDirectoryLoader(path)
        all_docs.extend(loader.load())
        # Load all CSV files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if filename.endswith(".csv"):
                csv_loader = CSVLoader(file_path)
                all_docs.extend(csv_loader.load())
    return all_docs


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    new_chunk_ids = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk.metadata["id"])

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        # Add in batches
        batch_size = 5000
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            batch_ids = new_chunk_ids[i:i+batch_size]
            db.add_documents(batch, ids=batch_ids)
    else:
        print("No new documents to add.")
def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


if __name__ == "__main__":
    main()









