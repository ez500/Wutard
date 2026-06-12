import os

from bs4 import BeautifulSoup
import chromadb
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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


def build_external_docs_vector_database(documents, ids, metadatas):
    if documents:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./db")
        collection = client.get_or_create_collection(name="external_docs")

        cleaned_documents = []
        cleaned_ids = []
        cleaned_metadatas = []

        for document, id, metadata in zip(documents, ids, metadatas):
            if not document or not document.strip():
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            vendor = id.split("_", 1)[0] if "_" in id else "unknown"

            normalized_metadata = {
                "vendor": vendor,
                "id": id,
                **metadata,
            }

            normalized_metadata = {
                key: str(value) if value is not None else "" for key, value in normalized_metadata.items()
            }

            cleaned_documents.append(document)
            cleaned_ids.append(id)
            cleaned_metadatas.append(normalized_metadata)

        if not cleaned_documents or not cleaned_ids or not cleaned_metadatas:
            print("No files found to embed.")
            return

        embeddings = model.encode(cleaned_documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=cleaned_documents,
            metadatas=cleaned_metadatas,
            ids=cleaned_ids
        )

        vendor_name = cleaned_ids[0].split("_", 1)[0]
        print(f"Completed embedding for {len(cleaned_documents)} files to ./db/{collection.id} for "
              f"{vendor_name} external vendor")
    else:
        print("No files found to embed.")


def build_wpilib_vector_database():
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
                            ids.append(f"wpilib_{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append({
                                "source": f"WPILib/{filename}",
                                "file_path": file_path,
                                "filename": filename,
                                "chunk_index": idx,
                                **split.metadata,
                            })
                            file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_doglog_vector_database():
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

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/doglog/web/src/content/docs"):
        for filename in files:
            if filename.endswith(".mdoc") or filename.endswith(".md") or filename.endswith(".mdx"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        splits = markdown_splitter.split_text(content)

                        for idx, split in enumerate(splits):
                            documents.append(split.page_content)
                            ids.append(f"doglog_{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append({
                                "source": f"DogLog/{filename}",
                                "file_path": file_path,
                                "filename": filename,
                                "chunk_index": idx,
                                **split.metadata,
                            })
                            file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_photonlib_vector_database():
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

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/photonvision/docs/source/docs"):
        for filename in files:
            if filename.endswith(".mdoc") or filename.endswith(".md") or filename.endswith(".mdx"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        splits = markdown_splitter.split_text(content)

                        for idx, split in enumerate(splits):
                            documents.append(split.page_content)
                            ids.append(f"photonlib_{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append({
                                "source": f"PhotonLib/{filename}",
                                "file_path": file_path,
                                "filename": filename,
                                "chunk_index": idx,
                                **split.metadata,
                            })
                            file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_phoenix6_vector_database():
    headers_to_split_on = [
        r"\n\n",
        r"\n=+\n",
        r"\n-+\n",
        r"\n~+\n",
        r"\n",
        r" "
    ]
    rst_splitter = RecursiveCharacterTextSplitter(
        separators=headers_to_split_on,
        is_separator_regex=True,
        chunk_size=1000,
        chunk_overlap=150
    )

    documents = []
    ids = []
    metadatas = []
    file_count = 0

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/Phoenix6-Documentation/source/docs"):
        for filename in files:
            if filename.endswith(".rst"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        splits = rst_splitter.split_text(content)

                        for idx, split in enumerate(splits):
                            documents.append(split)
                            ids.append(f"phoenix6_{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append({
                                "source": f"Phoenix6/{filename}",
                                "file_path": file_path,
                                "filename": filename,
                                "chunk_index": idx,
                            })
                            file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_pathplanner_vector_database():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    xml_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    documents = []
    ids = []
    metadatas = []
    file_count = 0

    for root, _, files in os.walk("/home/bigfatmidget/Projects/Robotics/photonvision/docs/source/docs"):
        for filename in files:
            if filename.endswith(".mdoc") or filename.endswith(".md") or filename.endswith(".mdx"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    if content.strip():
                        splits = markdown_splitter.split_text(content)

                        for idx, split in enumerate(splits):
                            documents.append(split.page_content)
                            ids.append(f"pathplanner_{file_count}_{file_path}_chunk_{idx}")
                            metadatas.append({
                                "source": f"PathPlanner/{filename}",
                                "file_path": file_path,
                                "filename": filename,
                                "chunk_index": idx,
                                **split.metadata,
                            })
                            file_count += 1

            elif filename.endswith(".topic"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read()
                    soup = BeautifulSoup(raw_content, "xml")
                    clean_text = soup.get_text(separator="\n\n", strip=True)
                    splits = xml_splitter.split_text(clean_text)

                    for idx, split in enumerate(splits):
                        documents.append(split)
                        ids.append(f"pathplanner_{file_count}_{file_path}_chunk_{idx}")
                        metadatas.append({
                            "source": f"PathPlanner/{filename}",
                            "file_path": file_path,
                            "filename": filename,
                            "chunk_index": idx,
                        })
                        file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_revlib_vector_database():
    loader = RecursiveUrlLoader(
        url="https://docs.revrobotics.com/revlib",
        max_depth=3,
        extractor=lambda html_content: (
            BeautifulSoup(html_content, "html.parser").get_text(separator="\n", strip=True)
        )
    )

    web_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    documents = []
    ids = []
    metadatas = []
    file_count = 0

    for doc in web_docs:
        splits = text_splitter.split_text(doc.page_content)

        for idx, split in enumerate(splits):
            if len(split.strip()) > 50:
                documents.append(split)
                ids.append(f"revlib_{file_count}_{doc.metadata['source']}_chunk_{idx}")
                metadatas.append({
                    "source": doc.metadata.get('source', "REVLib"),
                    "chunk_index": idx,
                })
                file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def build_reduxlib_vector_database():
    loader = RecursiveUrlLoader(
        url="https://docs.reduxrobotics.com/",
        max_depth=3,
        extractor=lambda html_content: (
            BeautifulSoup(html_content, "html.parser").get_text(separator="\n", strip=True)
        )
    )

    web_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    documents = []
    ids = []
    metadatas = []
    file_count = 0

    for doc in web_docs:
        splits = text_splitter.split_text(doc.page_content)

        for idx, split in enumerate(splits):
            if len(split.strip()) > 50:
                documents.append(split)
                ids.append(f"reduxlib_{file_count}_{doc.metadata['source']}_chunk_{idx}")
                metadatas.append({
                    "source": doc.metadata.get('source', "ReduxLib"),
                    "chunk_index": idx,
                })
                file_count += 1

    build_external_docs_vector_database(documents, ids, metadatas)


def search_vector_databases(query, collection_name="rowdy25_codebase", top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results['documents'][0] if results['documents'] else ["No relevant results found"]


if __name__ == "__main__":
    build_robot_code_vector_database()
    build_wpilib_vector_database()
    build_doglog_vector_database()
    build_photonlib_vector_database()
    build_phoenix6_vector_database()
    build_pathplanner_vector_database()
    build_revlib_vector_database()
    build_reduxlib_vector_database()

    # client = chromadb.PersistentClient(path="./db")
    # collection = client.get_collection("rowdy25_codebase")
    # print(f"Total collection count: {collection.count()}")
    # sample_data = collection.peek(limit=2)
    # print("\nSample Documents:")
    # print(sample_data['documents'][0])
    # print(sample_data['documents'][1])

