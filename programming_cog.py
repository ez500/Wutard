import discord
from discord.ext import commands

from rag_service import AgenticRAGService


class Programming(commands.Cog):
    def __init__(self, client, rag_service):
        self.client = client
        self.rag_service = rag_service

    @commands.hybrid_command(name='programming_help', description='get briefed on how the llm feature should be used')
    async def programming_help(self, ctx):
        await ctx.send("Ah, how's it goin' ladies! I spent some time reading on FRC docs and Rowdy25. If you have any "
                       "questions feel free to head to <#1292666640256991282> and ask me for help. Make sure to get "
                       "my attention and ask your programming question.")

    @commands.hybrid_command(name='toggle_llm_model', description='toggle between the cheap and normal llm models')
    async def toggle_llm_model(self, ctx):
        if ctx.author.id == 434430979075997707:
            self.rag_service.premium_agent = not self.rag_service.premium_agent
            await ctx.send(f"LLM model toggled to {'premium' if self.rag_service.premium_agent else 'cheap'}.")
        else:
            await ctx.send("Ah, super! I'm not gonna let you do that. You're not my boss! Please go sit down. "
                           "That's life!")

    @commands.Cog.listener()
    async def on_message(self, message):
        client_user = self.client.user
        if client_user is None:
            return
        if message.author == client_user:
            return
        if message.author.bot:
            return

        valid_channels = [1292666640256991282, 1013977098370699305]
        is_valid_channel = message.channel.id in valid_channels
        is_valid_thread = isinstance(message.channel, discord.Thread) and message.channel.parent_id in valid_channels
        is_listening_thread = False
        if is_valid_thread:
            async for past_msg in message.channel.history(limit=5):
                if past_msg.author == client_user:
                    is_listening_thread = True
                    break
        is_mentioned = client_user in message.mentions or is_listening_thread

        if not (is_valid_channel or is_valid_thread):
            return

        filter_words = ["woodard", "chris", "rowdy", "rowdy25", "bot", "code", "coding", "java"]
        if not any(word in message.content.lower() for word in filter_words) and not is_mentioned:
            return

        print(f"User {message.author} registered query: {message.content}\n")
        if is_valid_thread:
            conversation_history = []
            async for past_msg in message.channel.history(limit=20, oldest_first=True):
                if past_msg.is_system():
                    continue

                role = "assistant" if past_msg.author == self.client.user else "user"
                conversation_history.append({"role": role, "content": past_msg.clean_content})

            _, output_code, response = (
                await self.rag_service.run_system_guardrail(conversation_history[-1]["content"], is_mentioned)
            )
            print(f"Evaluated as: {output_code}")
            if "GOOD" in output_code:
                async with (message.channel.typing()):
                    approved_query_response = await self.rag_service.run_agentic_query(conversation_history)
                    print(approved_query_response + "\n")
                    await message.channel.send(approved_query_response)
            elif response:
                print(response + "\n")
                await message.channel.send(response)
        else:
            title, output_code, response = await self.rag_service.run_system_guardrail(
                message.content, is_mentioned
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
