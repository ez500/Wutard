import os
import re
from typing import Any

import anthropic
import chromadb
from anthropic.types import ToolParam, MessageParam, ThinkingBlock, TextBlock, ToolUseBlock
from chromadb.errors import InternalError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

with open('claude_key', 'r') as f:
    api_key = f.readline().strip()

STOP_WORDS = {"the", "a", "an", "and", "or", "but", "is", "in", "to", "of", "it", "for"}


class AgenticRAGService:
    def __init__(self):
        print("Starting asynchronous Agentic RAG setup")

        print("Loading Claude Client")
        self.claude_client = anthropic.AsyncAnthropic(api_key=api_key)

        print("Loading JINAAI embedding model")
        self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
        self.embedding_model.max_seq_length = 2048

        print("Connecting to ChromaDB")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_db_path = os.path.join(base_dir, "code_doc_embedding", "db")

        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.code_collection = self.chroma_client.get_collection("rowdy25_codebase")
        self.external_docs_collection = self.chroma_client.get_collection("external_docs")

        print("Building BM25 Indices")
        self.bm25_code, self.code_ids = self._build_bm25_index(self.code_collection)
        self.bm25_external_docs, self.external_docs_ids = self._build_bm25_index(self.external_docs_collection)

        print("Agentic RAG setup complete.")

    # Internal Helpers

    @staticmethod
    def _tokenize_code(text):
        raw_tokens = re.split(r'\W+', text.lower())
        return [token for token in raw_tokens if token and token not in STOP_WORDS]

    def _build_bm25_index(self, collection):
        db_data = collection.get()
        all_ids = db_data["ids"]
        all_docs = db_data["documents"]

        corpus = [self._tokenize_code(doc) if doc is not None else [] for doc in all_docs]
        bm25_index = BM25Okapi(corpus)
        return bm25_index, all_ids

    @staticmethod
    def _query_document_embeddings(collection, query_embedding, top_k=3, vendor_filter=None):
        where_clause = None
        if vendor_filter:
            if len(vendor_filter) == 1:
                where_clause = {"source": {"$contains": vendor_filter[0]}}
            else:
                where_clause = {"$or": [{"source": {"$contains": vendor}} for vendor in vendor_filter]}

        query_kwargs: dict[str, Any] = {"query_embeddings": query_embedding, "n_results": min(top_k, 8), }

        if where_clause is not None:
            query_kwargs["where"] = where_clause

        try:
            results = collection.query(**query_kwargs)
            return AgenticRAGService._retrieve_relevant_context(results, collection)
        except InternalError:
            raise RuntimeError(
                "Local Chroma vector database index could not be loaded. Usually this means './code_doc_embedding/db' "
                "is corrupted or was created with an incompatible Chroma version. Stop the bot, move the database (db) "
                "to a backup location, and rebuild it using code_doc_embedding/embedder.py.")

    @staticmethod
    def _retrieve_relevant_context(initial_results, collection):
        final_results = {}
        retrieved_context = ""

        result_metadatas = initial_results["metadatas"]
        if result_metadatas and result_metadatas[0]:
            results_by_file = {}
            for metadata in result_metadatas[0]:
                if not metadata:
                    continue

                file_path = metadata.get("file_path")
                chunk_idx = metadata.get("chunk_index")
                if not isinstance(file_path, str):
                    continue
                if not isinstance(chunk_idx, (str, int)) or isinstance(chunk_idx, bool):
                    continue
                chunk_idx = int(chunk_idx)

                if file_path not in results_by_file:
                    results_by_file[file_path] = set()

                results_by_file[file_path].update([max(0, chunk_idx - 1), chunk_idx, chunk_idx + 1])

            or_conditions = [
                {
                    "$and": [
                        {"file_path": file_path},
                        {"chunk_index": {"$in": list(chunk_indices)}}
                    ]
                }
                for file_path, chunk_indices in results_by_file.items()
            ]

            where_clause = or_conditions[0] if len(or_conditions) == 1 else {"$or": or_conditions}

            expanded_results = collection.get(where=where_clause)
            expanded_documents = expanded_results["documents"] or []
            expanded_ids = expanded_results["ids"] or []
            expanded_metadatas = expanded_results["metadatas"] or []

            zipped_results = zip(
                expanded_documents,
                expanded_ids,
                expanded_metadatas
            )

            sorted_results = sorted(
                zipped_results,
                key=lambda by_meta: (by_meta[2]["file_path"], int(by_meta[2]["chunk_index"]))
            )

            final_results = {
                "documents": [[result[0] for result in sorted_results]],
                "ids": [[result[1] for result in sorted_results]],
                "metadatas": [[result[2] for result in sorted_results]]
            }

        if final_results['documents'] and final_results['documents'][0]:
            for i, doc in enumerate(final_results['documents'][0]):
                file_id = final_results['ids'][0][i]
                source = final_results['metadatas'][0][i].get('source', 'Unknown')

                retrieved_context += f"\n--- Start of snippet {file_id} (from {source}) ---\n"
                retrieved_context += doc
                retrieved_context += "\n--- End of snippet ---\n"
        return retrieved_context if retrieved_context else "No relevant documentation found"

    # Semantic Search

    def search_rowdy25(self, query):
        return self._query_document_embeddings(self.code_collection, self.embedding_model.encode([query]).tolist())

    def search_external_docs(self, query, top_k=3, vendor_filter=None):
        return self._query_document_embeddings(
            self.external_docs_collection,
            self.embedding_model.encode([query]).tolist(),
            top_k,
            vendor_filter
        )

    # Claude wrapper

    async def system_guardrail(self, user_query):
        # TODO: RUN THE FIRST CHECK OF RELEVANCE + ONE QUESTION ONLY
        pass

    async def run_agentic_query(self, user_query):
        # TODO: CREATE CONVERSATION HISTORY
        system_prompt = (f"You are an expert FRC programmer. Answer the user's question concisely using the "
                         f"following snippets from our team's codebase: "
                         f"{self.search_rowdy25(user_query)}")

        tool_schema: list[ToolParam] = [{
            "name": "search_external_docs",
            "description": "Searches the WPILib and vendor dependency documentation. Use this when you need"
                           "specific API details, hardware characteristics, framework constraints, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for, e.g., 'PathPlanner AutoBuilder constructors'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of document chunks (results) to return, depending on how large or "
                                       "broad in scope the query is. Use 1-3 for specific API checks, 3-6 for "
                                       "standard queries, and 6-8 for broad conceptual questions.",
                        "minimum": 1,
                        "maximum": 8
                    },
                    "vendor_filter": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["WPILib", "DogLog", "PhotonLib", "Phoenix6", "PathPlanner", "REVLib", "ReduxLib"]
                        },
                        "description": "Optional list of vendors to search by. Leave empty to search all vendors."}
                },
                "required": ["query"]
            }
        }]

        conversation_history: list[MessageParam] = [{"role": "user", "content": user_query}]

        while True:
            print("Sending prompt to Claude...")
            response = await self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                tools=tool_schema,
                system=system_prompt,
                messages=conversation_history
            )

            if response.stop_reason == "end_turn":
                thinking_blocks = [block for block in response.content if isinstance(block, ThinkingBlock)]
                if thinking_blocks:
                    print(f"Retrieved {len(thinking_blocks)} thinking blocks:")
                    for i, block in enumerate(thinking_blocks):
                        print(f"Block {i + 1}:\n{block.thinking}\n\n")
                else:
                    print("No extended thinking blocks in this turn.")

                text_block = next((block for block in response.content if isinstance(block, TextBlock)), None)
                if text_block:
                    return text_block.text
                else:
                    return "I'm sorry, you're going to have to ask Geeson over there. That's life!"

            if response.stop_reason == "tool_use":
                print(f"External documentation required, deploying retrieval")
                conversation_history.append({"role": "assistant", "content": response.content})

                tool_blocks = [block for block in response.content if isinstance(block, ToolUseBlock)]
                tool_results = []
                for tool_block in tool_blocks:
                    if tool_block and tool_block.name == "search_external_docs":
                        tool_result = self.search_external_docs(tool_block.input["query"])
                        print(f"Extracted documentation:\n{tool_result}\n\n")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": tool_result
                        })

                conversation_history.append({"role": "user", "content": tool_results})
