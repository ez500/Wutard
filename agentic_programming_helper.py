import anthropic
import chromadb
from sentence_transformers import SentenceTransformer


with open('claude_key', 'r') as f:
    api_key = f.readline().strip()


def search_vector_databases(query, collection_name="external_docs", top_k=3): # TODO: VENDOR FILTER
    # TODO: LOAD ONCE IN BOT START
    model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
    client = chromadb.PersistentClient(path="./code_doc_embedding/db")
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved_context = ""
    if results['documents'] and results['documents'][0]: # TODO: SOURCE INJECTION
        for i, doc in enumerate(results['documents'][0]):
            file_id = results['ids'][0][i]
            retrieved_context += f"\n--- Start of snippet from {file_id} ---\n"
            retrieved_context += doc
            retrieved_context += "\n--- End of snippet ---\n"
    return retrieved_context if retrieved_context else "No relevant documentation found"


def system_guardrail(user_query):
    # TODO: RUN THE FIRST CHECK OF RELEVANCE + ONE QUESTION ONLY
    pass


def run_agentic_query(user_query):
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = (f"You are an expert FRC programmer. Answer the user's question concisely using the "
                     f"following snippets from our team's codebase: "
                     f"{search_vector_databases(user_query, collection_name='rowdy25_codebase')}")

    tool_schema = [{
        "name": "search_vector_databases",
        "description": "Searches the WPILib and vendor dependency documentation. Use this when you need"
                       "specific API details, hardware characteristics, framework constraints, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to search for, e.g., "
                                                           "'PathPlanner AutoBuilder constructors'"},
                "collection_name": {"type": "string", "description": "The name of the collection to search in "
                                                                     "(usually 'external_docs')"},
                "top_k": {"type": "integer", "description": "The number of results to return, depending on how large "
                                                            "or broad in scope the query is"}
            },
            "required": ["query"]
        }
    }]

    print("Sending prompt to Claude")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        tools=tool_schema,
        messages=[
            {"role": "user", "content": ("I am writing code for our FRC robot. Why might our CANSparkMax be "
                                         "stuttering? What are its current limits?")}
        ]
    )

    if response.stop_reason == "tool_use":
        print(f"External documentation required, stopping generation")
        tool_block = next(block for block in response.content if block["type"] == "tool_use")

        if tool_block.name == "search_vector_databases":
            query_arg = tool_block.input["query"]

            tool_result = search_vector_databases(query_arg)
            print(f"Extracted documentation: {tool_result}")


if __name__ == "__main__":
    context = search_vector_databases("What does RobotStates do?", collection_name="rowdy25_codebase")
    print(context)
