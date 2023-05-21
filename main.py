import discord
from discord.ext import commands
from discord.ext import tasks

import random
import datetime
import asyncio


with open('client.token', 'r') as f:
    token = f.readline().strip()


client = commands.Bot(command_prefix='super ', help_command=None, intents=discord.Intents.all())


@client.event
async def on_ready():
    await client.change_presence(status=discord.Status.idle, activity=discord.Game(name='Ahh..Super!'))
    print('Logged in')


@client.event
async def on_guild_join(guild):
    channels = guild.channels
    for c in channels:
        if c.type == discord.ChannelType.text:
            channel = c.id
            await client.get_channel(channel).send('Hello! How\'s it goin\'? I\'m Mr. Woodard, and I am your computer '
                                                   'teacher. As the year begins, I lecture a lot, and as it goes on, I '
                                                   'lecture a little. Everyone ready for the bell ringer? Ah, super!')
            break


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.author.bot:
        return

    if client.user.mention in message.content:
        try:
            await message.channel.send(f'Ah, {message.author.mention}, hello! Hey there! How are you?')
            await client.wait_for('message',
                                  check=lambda m: m.channel == message.channel and m.author == message.author,
                                  timeout=20.0)
            await message.channel.send('Yes,')
        except asyncio.exceptions.TimeoutError:
            await message.channel.send('Ah, you ignored me. That\'s life!')

    elif random.randint(0, 19) == 0:
        if message.channel != 1013977098370699305:
            if random.randint(0, 5) == 0:
                await message.channel.send('Ah, super!')
        else:
            await message.channel.send('Ah, super!')

    await client.process_commands(message)


@client.command(name='help', description='ask for help')
async def help(ctx):
    await ctx.send('Sorry, I can\'t help you. Ah, that\'s life. I think <@377907768071553024> over there can though.',
                   allowed_mentions=discord.AllowedMentions(everyone=False, users=False, roles=False))


@client.command(name='leave', description='leave', aliases=['closeofbusiness'])
async def closeofbusiness(ctx):
    await ctx.send('Take care!')


@client.command(name='start', description='start class', aliases=['startofdice'])
async def startofclass(ctx):
    if random.randint(0, 1) == 0:
        message = 'Hello! How\'s it going. '
    else:
        if datetime.datetime.now().hour < 5:
            time = 'night'
        elif datetime.datetime.now().hour < 12:
            time = 'morning'
        elif datetime.datetime.now().hour < 18:
            time = 'afternoon'
        elif datetime.datetime.now().hour < 21:
            time = 'evening'
        else:
            time = 'night'
        message = f'Good {time}! '
    phrases = ['It is a super day! ', 'Ah, that\'s life. ', 'Who\'s ready for the bell ringer? I surely am. ',
               'How\'s it goin\', ladies? ', 'Ah, super! ', 'Alright! ', 'Go on in and sit down! ']
    for i in range(random.randint(1, 7)):
        current = phrases
        index = random.randint(0, len(current) - 1)
        message += current[index]
        del current[index]
    message = message[0:-1]
    await ctx.send(message)


@client.command(name='dice', description='roll dice', aliases=['rollthedice'])
async def rollthedice(ctx):
    message = f'Everyone ready for the bell ringer? Time to roll the dice! It\'s a {random.randint(1, 6)}! Who got A? '
    index = random.randint(1, 5)
    answers = ['A', 'B', 'C', 'D', 'E']
    answer = ['Benchmarking', 'Crowd-sourcing', 'The back of the book', 'The textbook', 'CodeHS']
    if index > 1:
        message += 'Who got B? '
    if index > 2:
        message += 'Who got C? '
    if index > 3:
        message += 'Who got D? '
    if index > 4:
        message += 'Who got E? '
    message += f'{random.choice(answer)} says the answer is {answers[index - 1]}! '
    message += 'Super!'
    await ctx.send(message)


@client.command(name='lecture', description='it\'s lecture time', aliases=['lecturetime'])
async def lecturetime(ctx):
    lectures = ['Unit 1 Primitive Types', 'Unit 2 Using Objects', 'Unit 3 Boolean Expression and if Statements',
                'Unit 4 Iteration', 'Unit 5 Writing Classes', 'Unit 6 Array', 'Unit 7 ArrayList', 'Unit 8 2DArray',
                'Unit 9 Inheritance', 'Unit 10 Recursion']
    lessons = ['What all computer user face?', 'The power to be proactive', 'You if seen this, but are stressed',
               'Which RDP does being proactive fall under?', '"Hey! Look what Zog do!"', 'How to tame email',
               'You have a knowledge of time management, but no an understanding', 'Benchmarking successful people',
               'How to manage files', 'The power of Win-Win', 'Win - Win - Teams', 'Why this is powerful?',
               'Lack of planning = easy jobs are easy', 'Lack of planning = hard projects get hard']
    message = f'Alright! Lecture time. {random.choice(lectures)}! But first, better today then yesterday! ' \
              f'{random.choice(lessons)}'
    await ctx.send(message)


@tasks.loop(time=datetime.time(hour=18, minute=30))
async def super_every_two_minutes():
    await client.wait_until_ready()
    await client.get_channel(1013977098370699305).send('Ah, super!')


async def main():
    super_every_two_minutes.start()
    await client.start(token)


if __name__ == '__main__':
    asyncio.run(main())
