import os

import anthropic
import chromadb
from anthropic.types import ToolParam, MessageParam, ToolUseBlock, TextBlock, ThinkingBlock
from discord.ext import commands
from sentence_transformers import SentenceTransformer

from rag_util import query_document_embeddings

with open('claude_key', 'r') as f:
    api_key = f.readline().strip()


class Programming(commands.Cog):
    def __init__(self, client):
        self.client = client

        self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)
        self.embedding_model.max_seq_length = 2048

        base_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_db_path = os.path.join(base_dir, "code_doc_embedding", "db")

        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.code_collection = self.chroma_client.get_collection("rowdy25_codebase")
        self.external_docs_collection = self.chroma_client.get_collection("external_docs")

        self.claude_client = anthropic.Anthropic(api_key=api_key)

    def embed_query(self, query):
        return self.embedding_model.encode([query]).tolist()

    def search_rowdy25(self, query):
        return query_document_embeddings(self.code_collection, self.embed_query(query))

    def search_external_docs(self, query, top_k=3, vendor_filter=None):
        return query_document_embeddings(self.external_docs_collection, self.embed_query(query), top_k, vendor_filter)

    def system_guardrail(self, user_query):
        # TODO: RUN THE FIRST CHECK OF RELEVANCE + ONE QUESTION ONLY
        pass

    def run_agentic_query(self, user_query):
        # TODO: CREATE CONVERSATION HISTORY
        system_prompt = (f"You are an expert FRC programmer. Answer the user's question concisely using the "
                         f"following snippets from our team's codebase: "
                         f"{self.search_rowdy25(self.embed_query(user_query))}")

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
            response = self.claude_client.messages.create(
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

    @commands.Cog.listener()
    async def on_message(self, message):
        client_user = self.client.user
        if client_user is None:
            return
        if message.author == client_user:
            return
        if message.author.bot:
            return

        if message.channel.id == 1292666640256991282 or message.channel.id == 1013977098370699305:
            await message.channel.send(self.run_agentic_query(message.content))


async def setup(client):
    await client.add_cog(Programming(client))


if __name__ == "__main__":
    test = Programming(None)
    # context = test.search_code_rag("What does RobotStates do?")
    context = test.search_code_rag("How does the robot path find around the reef using commands?")
    print(context)
