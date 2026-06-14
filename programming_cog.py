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

        if message.channel.id == 1292666640256991282 or message.channel.id == 1013977098370699305:
            await message.channel.send(self.rag_service.run_agentic_query(message.content))


async def setup(client):
    await client.add_cog(Programming(client, AgenticRAGService()))
