from typing import Any

from chromadb.errors import InternalError


def query_document_embeddings(collection, query_embedding, top_k=3, vendor_filter=None):
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
    except InternalError:
        raise RuntimeError(
            "Local Chroma vector database index could not be loaded. Usually this means './code_doc_embedding/db' "
            "is corrupted or was created with an incompatible Chroma version. Stop the bot, move the database (db) "
            "to a backup location, and rebuild it using code_doc_embedding/embedder.py.")
    return _retrieve_relevant_context(results, collection)


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
