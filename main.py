import discord
from discord import Guild, Message
from discord.ext import commands
from discord.ext import tasks

import random
import datetime
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
async def on_guild_join(guild: Guild):
    channels = guild.channels
    for c in channels:
        if c.type == discord.ChannelType.text:
            channel = c.id
            await client.get_channel(channel).send('Hello! How\'s it goin\'? I\'m Mr. Woodard, and I am your computer '
                                                   'teacher. As the year begins, I lecture a lot, and as it goes on, I '
                                                   'lecture a little. Everyone ready for the bell ringer? Ah, super!')
            break


@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return
    if message.author.bot:
        return
    if message.content.__contains__(client.user.mention):
        try:
            await message.channel.send(f'Ah, {message.author.mention}, hello! Hey there! How are you?')

            def check(m):
                return m.channel == message.channel and m.author == message.author

            msg = await client.wait_for('message', check=check, timeout=20.0)
            await message.channel.send('Yes,')
        except asyncio.exceptions.TimeoutError:
            await message.channel.send('Ah, you ignored me. That\'s life!')

    elif random.randint(0, 1) == 0:
        await message.channel.send('Ah, super!')

    await client.process_commands(message)


@client.command(name='help', description='ask for help')
async def help(ctx):
    await ctx.send('Sorry, I can\'t help you. Ah, that\'s life. I think <@377907768071553024> over there can though.', allowed_mentions=discord.AllowedMentions(everyone=False, users=False, roles=False))


@client.command(name='leave', description='leave', aliases=['closeofbusiness'])
async def closeofbusiness(ctx):
    await ctx.send('Take care!')


@client.command(name='start', description='start class', aliases=['startofclass'])
async def startofclass(ctx):
    await ctx.send('It is a super day! Ah, that’s life. Who’s ready for the bell ringer? I surely am. It’s a three! '
                   'Who got A? Who got B? Who got C? Cs get degrees. Ah, super. '
                   'The answer at the back of the book is C. Alright! Lecture time. Unit 10 iteration! '
                   'But first, better today than yesterday!')


@tasks.loop(time=datetime.time(hour=18, minute=30))
async def super_every_two_minutes():
    await client.wait_until_ready()
    await client.get_channel(1013977098370699305).send('Ah, super!')


async def main():
    super_every_two_minutes.start()
    await client.start(token)


if __name__ == '__main__':
    asyncio.run(main())
