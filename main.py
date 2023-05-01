import discord
from discord.ext import commands
from discord.ext import tasks

import asyncio
import pathlib as pl

with open('client.token', 'r') as f:
    token = f.readline().strip()

client = commands.Bot(command_prefix='super ', help_command=None, intents=discord.Intents.all())


@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.idle, activity=discord.Game(name='Ahh..Super!'))
    print('Logged in')


async def main():
    await client.start(token)


@client.command()
async def help(ctx):
    await ctx.send('Sorry, I can\'t help you. Ah, that\'s life. I think <@377907768071553024> over there can though.')
    print("Hello")


if __name__ == '__main__':
    asyncio.run(main())
