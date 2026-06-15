import discord
from discord.ext import commands

from rag_service import AgenticRAGService


class Programming(commands.Cog):
    def __init__(self, client, rag_service):
        self.client = client
        self.rag_service = rag_service

    @commands.Cog.listener()
    async def on_message(self, message):
        client_user = self.client.user
        if client_user is None:
            return
        if message.author == client_user:
            return
        if message.author.bot:
            return

        target_channels = [1292666640256991282, 1013977098370699305]

        is_target_channel = message.channel.id in target_channels
        is_target_thread = isinstance(message.channel, discord.Thread) and message.channel.parent_id in target_channels

        if not (is_target_channel or is_target_thread):
            return

        filter_words = ["woodard", "chris", "rowdy", "rowdy25", "bot", "code", "coding", "java"]
        if not any(word in message.content.lower() for word in filter_words) and client_user not in message.mentions:
            return

        print(f"User {message.author} registered query: {message.content}\n")
        if is_target_thread:
            conversation_history = []
            async for past_msg in message.channel.history(limit=20, oldest_first=True):
                if past_msg.is_system():
                    continue

                role = "assistant" if past_msg.author == self.client.user else "user"
                conversation_history.append({"role": role, "content": past_msg.clean_content})

            _, output_code, response = (
                await self.rag_service.run_agentic_query_with_system_guardrail(conversation_history)
            )
            print(f"Evaluated as: {output_code}")
            if "GOOD" in output_code:
                async with (message.channel.typing()):
                    approved_query_response = await self.rag_service.run_agentic_query(response)
                    print(approved_query_response + "\n")
                    await message.channel.send(approved_query_response)
            elif response:
                print(response + "\n")
                await message.channel.send(response)
        else:
            title, output_code, response = await self.rag_service.register_agentic_query_with_system_guardrail(
                message.content
            )
            print(f"Evaluated as: {output_code}")
            if title:
                try:
                    async with ((message.channel.typing())):
                        approved_query_response = await self.rag_service.run_agentic_query(response)
                        print(approved_query_response + "\n")
                        new_thread = await message.create_thread(name=title, auto_archive_duration=60)
                        await new_thread.send(approved_query_response)
                except discord.HTTPException as e:
                    print(f"Failed to create thread: {e}\n")
                    await message.channel.send("I am not able to create threads. I guess I can't help. Not super!")
            elif response:
                print(response + "\n")
                await message.channel.send(response)


async def setup(client):
    await client.add_cog(Programming(client, AgenticRAGService()))
