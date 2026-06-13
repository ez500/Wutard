from discord.ext import commands


class Programming(commands.Cog):
    def __init__(self, client):
        self.client = client

    @commands.hybrid_command(name="test", help="Responds with a test message.")
    async def test(self, ctx):
        await ctx.send("Test message.")


async def setup(client):
    await client.add_cog(Programming(client))
