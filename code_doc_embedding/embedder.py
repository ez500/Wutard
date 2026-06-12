import os

import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer


def build_robot_code_vector_database():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection(name="rowdy25_codebase")

    documents = []
    ids = []
    file_count = 0

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/Rowdy25"):
        for filename in files:
            if filename.endswith(".md") or filename.endswith(".java"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        documents.append(content)
                        ids.append(f"{file_count}_{file_path}")
                        file_count += 1

    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"Completed embedding for {len(documents)} files to ./db/{collection.id}")
    else:
        print("No files found to embed.")


def build_wpilib_vector_database():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection(name="wpilib_docs")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    documents = []
    ids = []
    metadatas = []
    file_count = 0

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/wpilib-docs/source/docs/software"):
        for filename in files:
            if filename.endswith(".rst") or filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        splits = markdown_splitter.split_text(content)

                        for idx, split in enumerate(splits):
                            documents.append(split.page_content)
                            ids.append(f"{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append(split.metadata)
                            file_count += 1

    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"Completed embedding for {len(documents)} files to ./db/{collection.id}")
    else:
        print("No files found to embed.")


if __name__ == "__main__":
    build_robot_code_vector_database()
    build_wpilib_vector_database()

    # client = chromadb.PersistentClient(path="./db")
    # collection = client.get_collection("rowdy25_codebase")
    # print(f"Total collection count: {collection.count()}")
    # sample_data = collection.peek(limit=2)
    # print("\nSample Documents:")
    # print(sample_data['documents'][0])
    # print(sample_data['documents'][1])

