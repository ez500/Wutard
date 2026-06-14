import asyncio
import os
import re
from typing import Any

import anthropic
import chromadb
import numpy as np
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

    def _query_document_embeddings(self, collection, query, top_k=3, vendor_filter=None):
        query_embedding = self.embedding_model.encode([query]).tolist()
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
            return results["ids"][0] if results["ids"] else []
        except InternalError:
            raise RuntimeError(
                "Local Chroma vector database index could not be loaded. Usually this means './code_doc_embedding/db' "
                "is corrupted or was created with an incompatible Chroma version. Stop the bot, move the database (db) "
                "to a backup location, and rebuild it using code_doc_embedding/embedder.py.")

    def _query_keyword_index(
            self,
            collection,
            keyword_index,
            query,
            top_k=3,
            vendor_filter=None
    ):
        where_clause = None
        if vendor_filter:
            if len(vendor_filter) == 1:
                where_clause = {"source": {"$contains": vendor_filter[0]}}
            else:
                where_clause = {"$or": [{"source": {"$contains": vendor}} for vendor in vendor_filter]}

        tokenized_query = self._tokenize_code(query)
        bm25_scores = keyword_index.get_scores(tokenized_query)

        keyword_hits = []

        allowed_ids = set()
        if where_clause:
            allowed_ids = set(collection.get(where=where_clause)["ids"])

        top_indices = np.argsort(bm25_scores)[::-1]
        for idx in top_indices:
            doc_id = self.code_ids[idx]
            if where_clause and doc_id not in allowed_ids:
                continue
            keyword_hits.append(doc_id)

            if len(keyword_hits) == top_k:
                break
        return keyword_hits

    @staticmethod
    def _reciprocal_rank_fusion(vector_hits, keyword_hits, top_k=3):
        rrf_scores = {}
        k = 60

        for rank, doc_id in enumerate(vector_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        for rank, doc_id in enumerate(keyword_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        ranked_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_final_ids = [doc_id for doc_id, _ in ranked_doc_ids[:top_k]]
        return top_final_ids if top_final_ids else ["No relevant results found"]

    @staticmethod
    def _retrieve_relevant_context(init_result_ids, collection):
        init_results = collection.get(ids=init_result_ids)
        init_result_docs = init_results.get("documents") or []
        init_result_metadatas = init_results.get("metadatas") or []

        if not init_result_metadatas:
            return "\n\n".join(init_result_docs) if init_result_docs else "No relevant results found"

        results_by_file = {}
        for metadata in init_result_metadatas:
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

        if not results_by_file:
            return "\n\n".join(init_result_docs) if init_result_docs else "No relevant results found"

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

        sorted_results = sorted(
            zip(expanded_documents, expanded_ids, expanded_metadatas),
            key=lambda result: (
                str(result[2].get("file_path", "")),
                int(result[2].get("chunk_index") or 0)
            )
        )

        retrieved_context = ""
        for document, doc_id, metadata in sorted_results:
            source = metadata.get("source", "Unknown")
            retrieved_context += f"\n--- Start of snippet {doc_id} (from {source}) ---\n"
            retrieved_context += document
            retrieved_context += "\n--- End of snippet ---\n"
        return retrieved_context if retrieved_context else "No relevant results found"

    # Semantic Search

    def _strict_semantic_search(self, collection, query, top_k=3, vendor_filter=None):
        return self._retrieve_relevant_context(
            self._query_document_embeddings(collection, query, top_k, vendor_filter),
            collection
        )

    def _keyword_semantic_hybrid_search(self, collection, keyword_index, query, top_k=3, vendor_filter=None):
        vector_results = self._query_document_embeddings(collection, query, top_k, vendor_filter)
        keyword_results = self._query_keyword_index(collection, keyword_index, query, top_k, vendor_filter)
        combined_results = self._reciprocal_rank_fusion(vector_results, keyword_results, top_k)
        return self._retrieve_relevant_context(combined_results, collection)

    def search_rowdy25(self, query, top_k=3, hybrid=True):
        if hybrid:
            return self._keyword_semantic_hybrid_search(self.code_collection, self.bm25_code, query, top_k)
        return self._strict_semantic_search(self.code_collection, query, top_k)

    def search_external_docs(self, query, top_k=3, vendor_filter=None, hybrid=True):
        if hybrid:
            return self._keyword_semantic_hybrid_search(self.external_docs_collection, self.bm25_external_docs,
                                                        query, top_k, vendor_filter)
        return self._strict_semantic_search(self.external_docs_collection, query, top_k, vendor_filter)

    # Claude wrapper

    async def system_guardrail(self, user_query):
        # TODO: RUN THE FIRST CHECK OF RELEVANCE + ONE QUESTION ONLY
        pass

    async def _retrieve_robot_code_context(self, user_query):
        system_prompt = (f"You are a research assistant. Analyze the user query and generate a list of 3 precise "
                         f"technical search queries for a code search engine. Respond in the form '[query1, query2, "
                         f"query3]', and limit your queries to less than 6 words per query. "
                         f"Here is the list of classes that might be useful to include in your queries: "
                         f"RobotContainer, Main, IntakeCommand, DirectMoveToPoseCommand, "
                         f"PathfindToPoseAvoidingReefCommand, DriveCommand, ElevatorCommand, WristCommand, "
                         f"PivotCommand, SearchForObjectCommand, FollowPathRequiringAlgaeCommand, Lights, Localizer, "
                         f"LocalizerSim, LocalizationTelemetry, Wrist, WristTelemetry, ElevatorTelemetry, Elevator, "
                         f"Pivot, PivotTelemetry, IntakeTelemetry, Intake, Swerve, SwerveSim, Song, SwerveTelemetry, "
                         f"RobotIdentity, RobotPoses, Constants, CompConstants, TestConstants, DefaultConstants, "
                         f"SimConstants, PhoenixProfiledPIDController, EquationUtil, PhotonUtil, QuestNavUtil, "
                         f"LimelightUtil, Elastic, DoubleTrueTrigger, EstimatedRobotPose, GravityGainsCalculator, "
                         f"MacAddress, RotationUtil, MultipleChooser, ProfiledExpEndController, FieldUtil, SysID, "
                         f"AutoTrigger, AutoEventLooper, AutoManager, Pathfinder, RobotStates, Robot. "
                         f"Use prefix 'class' for your query if a particular Java "
                         f"class is mentioned or is very obviously what the user is asking about.")

        message: list[MessageParam] = [{"role": "user", "content": user_query}]
        response = await self.claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system_prompt,
            messages=message
        )

        text_block = next((block for block in response.content if isinstance(block, TextBlock)), None)
        if text_block:
            print(text_block.text)
            return self.search_rowdy25(text_block.text)
        else:
            return "No relevant results found"

    async def run_agentic_query(self, user_query):
        # TODO: CREATE CONVERSATION HISTORY
        system_prompt = ("You are an expert FRC programmer. Answer the user's question concisely (50 words). "
                         "The broader the question, the less specific you can be in your answer. "
                         "Only use each tool once. Do not strive for perfection. "
                         "Use the available tools to answer questions about the codebase, "
                         "hardware, and API details. "
                         "For searching Rowdy25 code, you must generate up to 3 precise technical search queries "
                         "for the string query argument that is in the form '[query1, query2, query3]', with each "
                         "query limited to less than 6 words per query. Be clever with your queries. "
                         "Here is the list of classes that might be useful to include in your queries: "
                         "RobotContainer, Main, IntakeCommand, DirectMoveToPoseCommand, "
                         "PathfindToPoseAvoidingReefCommand, DriveCommand, ElevatorCommand, WristCommand, "
                         "PivotCommand, SearchForObjectCommand, FollowPathRequiringAlgaeCommand, Lights, Localizer, "
                         "LocalizerSim, LocalizationTelemetry, Wrist, WristTelemetry, ElevatorTelemetry, Elevator, "
                         "Pivot, PivotTelemetry, IntakeTelemetry, Intake, Swerve, SwerveSim, Song, SwerveTelemetry, "
                         "RobotIdentity, RobotPoses, Constants, CompConstants, TestConstants, DefaultConstants, "
                         "SimConstants, PhoenixProfiledPIDController, EquationUtil, PhotonUtil, QuestNavUtil, "
                         "LimelightUtil, Elastic, DoubleTrueTrigger, EstimatedRobotPose, GravityGainsCalculator, "
                         "MacAddress, RotationUtil, MultipleChooser, ProfiledExpEndController, FieldUtil, SysID, "
                         "AutoTrigger, AutoEventLooper, AutoManager, Pathfinder, RobotStates, Robot. "
                         "Use prefix 'class' for your query if a particular Java "
                         "class is mentioned or is very obviously what the user is asking about.")

        tool_schema: list[ToolParam] = [
            {
                "name": "search_rowdy25",
                "description": "Searches the Rowdy25 codebase. Use this when you need to answer questions about "
                               "certain implementation details, such as how a certain feature of the robot works, "
                               "or how a specific command is built.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The string query argument that is in the form '[query1, query2, query3]', "
                                           "with each query limited to less than 6 words per query. Here is the list "
                                           "of classes that might be useful to include in the queries: "
                                           "RobotContainer, Main, IntakeCommand, DirectMoveToPoseCommand, "
                                           "PathfindToPoseAvoidingReefCommand, DriveCommand, ElevatorCommand, "
                                           "WristCommand, PivotCommand, SearchForObjectCommand, "
                                           "FollowPathRequiringAlgaeCommand, Lights, Localizer, LocalizerSim, "
                                           "LocalizationTelemetry, Wrist, WristTelemetry, ElevatorTelemetry, Elevator, "
                                           "Pivot, PivotTelemetry, IntakeTelemetry, Intake, Swerve, SwerveSim, Song, "
                                           "SwerveTelemetry, RobotIdentity, RobotPoses, Constants, CompConstants, "
                                           "TestConstants, DefaultConstants, SimConstants, "
                                           "PhoenixProfiledPIDController, EquationUtil, PhotonUtil, QuestNavUtil, "
                                           "LimelightUtil, Elastic, DoubleTrueTrigger, EstimatedRobotPose, "
                                           "GravityGainsCalculator, MacAddress, RotationUtil, MultipleChooser, "
                                           "ProfiledExpEndController, FieldUtil, SysID, AutoTrigger, AutoEventLooper, "
                                           "AutoManager, Pathfinder, RobotStates, Robot. "
                                           "Use prefix 'class' for a query if a particular Java "
                                           "class is mentioned or is very obviously what the user is asking about."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "The number of document chunks (results) to return, depending on how large "
                                           "or broad in scope the query is. Use 1 for specific API checks, 2 for "
                                           "standard queries, and 3 for broad conceptual questions.",
                            "minimum": 1,
                            "maximum": 3
                        },
                    },
                    "required": ["query"]
                },
            },
            {
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
                            "description": "The number of document chunks (results) to return, depending on how large "
                                           "or broad in scope the query is. Use 1 for specific API checks, 2 for "
                                           "standard queries, and 3 for broad conceptual questions.",
                            "minimum": 1,
                            "maximum": 3
                        },
                        "vendor_filter": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["WPILib", "DogLog", "PhotonLib", "Phoenix6", "PathPlanner",
                                         "REVLib", "ReduxLib"]
                            },
                            "description": "Optional list of vendors to search by. Leave empty to search all vendors."}
                    },
                    "required": ["query"]
                }
            }
        ]

        conversation_history: list[MessageParam] = [{"role": "user", "content": user_query}]

        while True:
            print("Sending prompt to Claude...")
            response = await self.claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                tools=tool_schema,
                system=system_prompt,
                messages=conversation_history
            )

            if response.stop_reason == "end_turn":
                thinking_blocks = [block for block in response.content if isinstance(block, ThinkingBlock)]
                if thinking_blocks:
                    print(f"\nRetrieved {len(thinking_blocks)} thinking blocks:\n")
                    for i, block in enumerate(thinking_blocks):
                        print(f"Block {i + 1}:\n{block.thinking}\n\n")
                else:
                    print("\nNo extended thinking blocks in this turn.\n")

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
                    if tool_block and tool_block.name == "search_rowdy25":
                        query = tool_block.input.get("query")
                        top_k = tool_block.input.get("top_k", 3)

                        if not isinstance(query, str):
                            tool_result = "Invalid tool input: expected 'query' to be a string. Retry the tool."
                            print("Failed searching for Rowdy25 docs, invalid inputs.")
                        else:
                            if not isinstance(top_k, int):
                                top_k = 3
                            tool_result = self.search_rowdy25(query, top_k)
                            print(f"\nSearching for Rowdy25 docs!\n"
                                  f"Query: {query}\n"
                                  f"Top K results (order of context): {top_k}\n"
                                  f"Extracted Rowdy25 material:\n"
                                  f"{tool_result}\n\n")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": tool_result
                        })

                    if tool_block and tool_block.name == "search_external_docs":
                        query = tool_block.input.get("query")
                        top_k = tool_block.input.get("top_k", 3)
                        vendor_filter = tool_block.input.get("vendor_filter")

                        if not isinstance(query, str):
                            tool_result = "Invalid tool input: expected 'query' to be a string. Retry the tool."
                            print("Failed searching for Rowdy25 docs, invalid inputs.")
                        else:
                            if not isinstance(top_k, int):
                                top_k = 3

                            if not isinstance(vendor_filter, list):
                                vendor_filter = None
                            else:
                                vendor_filter = [vendor for vendor in vendor_filter if isinstance(vendor, str)]
                            tool_result = self.search_external_docs(query, top_k, vendor_filter)
                            print(f"\nSearching for Rowdy25 docs!\n"
                                  f"Query: {query}\n"
                                  f"Top K results (order of context): {top_k}\n"
                                  f"Vendor filter: {vendor_filter if vendor_filter else "All"}\n"
                                  f"Extracted Rowdy25 material:\n"
                                  f"{tool_result}\n\n")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": tool_result
                        })
                conversation_history.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    test = AgenticRAGService()
    # robot_context = asyncio.run(test.search_rowdy25("What does RobotStates do?")) # BAD (13 cents)
    # robot_context = asyncio.run(test.run_agentic_query(
    #     "How does the robot pathfind avoiding the reef using commands?" # BAD (8 cents)
    # ))
    # robot_context = asyncio.run(test.run_agentic_query( # ACCEPTABLE (6 cents)
    #     "Can you clarify what the SameSide method calculates when pathfinding avoiding the reef?"
    # ))
    # robot_context = asyncio.run(test.run_agentic_query(
    #     "What does two convenience composites mean for fields atReefAlgaeState and atAutoScoreState?" # GOOD (1 cent)
    # ))
    # print(robot_context)
    docs_context = asyncio.run(test.run_agentic_query(
        "What is different between PathPlanner PID and WPILib PID?"
    ))
    print(docs_context)
