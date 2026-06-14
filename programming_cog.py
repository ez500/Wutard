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

        if is_target_thread:
            async with (message.channel.typing()):
                conversation_history = []

                async for past_msg in message.channel.history(limit=20, oldest_first=True):
                    if past_msg.is_system():
                        continue

                    role = "assistant" if past_msg.author == self.client.user else "user"
                    conversation_history.append({"role": role, "content": past_msg.clean_content})
                _, response = (
                    await self.rag_service.continue_convo_with_system_guardrail(conversation_history)
                )
                print(response)
                await message.channel.send(response)
        else:
            async with message.channel.typing():
                title, response = await self.rag_service.register_query_with_system_guardrail(message.content)
                print(response)
                if title:
                    try:
                        new_thread = await message.create_thread(name=title, auto_archive_duration=60)
                        await new_thread.send(response)
                    except discord.HTTPException as e:
                        print(f"Failed to create thread: {e}")
                        await message.channel.send("I am not able to create threads. I guess I can't help. Not super!")
                else:
                    await message.channel.send(response)


async def setup(client):
    await client.add_cog(Programming(client, AgenticRAGService()))
