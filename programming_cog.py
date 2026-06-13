import anthropic
from discord.ext import commands
import chromadb
from sentence_transformers import SentenceTransformer


with open('claude_key', 'r') as f:
    api_key = f.readline().strip()


def retrieve_relevant_context(results):
    retrieved_context = ""
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            file_id = results['ids'][0][i]
            source = results['metadatas'][0][i].get('source', 'Unknown')

            retrieved_context += f"\n--- Start of snippet {file_id} (from {source}) ---\n"
            retrieved_context += doc
            retrieved_context += "\n--- End of snippet ---\n"
    return retrieved_context if retrieved_context else "No relevant documentation found"


class Programming(commands.Cog):
    def __init__(self, client):
        self.client = client

        self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
        self.embedding_model.max_seq_length = 2048
        self.chroma_client = chromadb.PersistentClient(path="./code_doc_embedding/db")
        self.code_collection = self.chroma_client.get_collection("rowdy25_codebase")
        self.external_docs_collection = self.chroma_client.get_collection("external_docs")

    def search_code_rag(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.code_collection.query(query_embeddings=query_embedding, n_results=top_k)
        return retrieve_relevant_context(results)

    def search_external_docs_rag(self, query, top_k=3, vendor_filter=None):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.external_docs_collection.query(query_embeddings=query_embedding, n_results=top_k)
        return retrieve_relevant_context(results, vendor_filter)

    def system_guardrail(self, user_query):
        # TODO: RUN THE FIRST CHECK OF RELEVANCE + ONE QUESTION ONLY
        pass

    def run_agentic_query(self, user_query):
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (f"You are an expert FRC programmer. Answer the user's question concisely using the "
                         f"following snippets from our team's codebase: "
                         f"{self.search_code_rag(user_query)}")

        tool_schema = [{
            "name": "search_external_docs_rag",
            "description": "Searches the WPILib and vendor dependency documentation. Use this when you need"
                           "specific API details, hardware characteristics, framework constraints, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search for, e.g., "
                                                               "'PathPlanner AutoBuilder constructors'"},
                    "top_k": {"type": "integer",
                              "description": "The number of results to return, depending on how large "
                                             "or broad in scope the query is (default is 3)"},
                    "vendor_filter": {"type": "string",
                                      "description": "Optional vendor to filter by (choose from WPILib, DogLog,"
                                                     "PhotonLib, Phoenix6, PathPlanner, REVLib, ReduxLib)"}
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

                tool_result = self.search_code_rag(query_arg)
                print(f"Extracted documentation: {tool_result}")

    @commands.hybrid_command(name="test", help="Responds with a test message.")
    async def test(self, ctx):
        await ctx.send("Test message.")


async def setup(client):
    await client.add_cog(Programming(client))


if __name__ == "__main__":
    test = Programming(None)
    context = test.search_code_rag("What does RobotStates do?", collection_name="rowdy25_codebase")
    print(context)
