import discord
from discord.ext import commands
from discord.ext import tasks

import random
import asyncio
import pathlib as pl


with open('client.token', 'r') as f:
    token = f.readline().strip()


client = commands.Bot(command_prefix='super ', help_command=None, intents=discord.Intents.all())


@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.idle, activity=discord.Game(name='Ahh..Super!'))
    print('Logged in')


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.author.bot:
        return
    if random.randint(0, 1) == 0:
        await message.channel.send('Ah, super!')

    await client.process_commands(message)


@client.command(name='help', description='ask for help')
async def help(ctx):
    await ctx.send('Sorry, I can\'t help you. Ah, that\'s life. I think <@377907768071553024> over there can though.')


@client.command(name='leave', description='leave', aliases=['closeofbusiness'])
async def closeofbusiness(ctx):
    await ctx.send('Take care!')


@client.command(name='start', description='start class', aliases=['startofclass'])
async def startofclass(ctx):
    await ctx.send('It is a super day! Ah, that’s life. Who’s ready for the bell ringer? I surely am. It’s a three! '
                   'Who got A? Who got B? Who got C? Cs get degrees. Ah, super. '
                   'The answer at the back of the book is C. Alright! Lecture time. Unit 10 iteration! '
                   'But first, better today than yesterday!')


@tasks.loop(minutes=2)
async def super_every_two_minutes():
    await client.wait_until_ready()
    await client.get_channel(1013977098370699305).send('Ah, super!')


async def main():
    super_every_two_minutes.start()
    await client.start(token)


if __name__ == '__main__':
    asyncio.run(main())
