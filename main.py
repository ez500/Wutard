import discord
from discord.ext import commands
from discord.ext import tasks

import os
import random
import datetime
import asyncio

from PIL import Image, ImageDraw, ImageFont


with open('client_token', 'r') as f:
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
        if message.author.id == 597056424803303435:
            await message.channel.send('Jiujiu, I need you to sit down and be quiet. Ah, that\'s life.')
        else:
            try:
                await message.channel.send(f'Ah, {message.author.mention}, hello! Hey there! How are you?')
                await client.wait_for('message',
                                      check=lambda m: m.channel == message.channel and m.author == message.author,
                                      timeout=20.0)
                await message.channel.send('Yes,')
            except asyncio.exceptions.TimeoutError:
                await message.channel.send('Ah, you ignored me. That\'s life!')
        return

    if message.content.lower() == 'super':
        await message.channel.send('Ah, super!')
        return
    elif random.randint(0, 19) == 0:
        if message.channel != 1013977098370699305:
            if random.randint(0, 5) == 0:
                await message.channel.send('Ah, super!')
        else:
            await message.channel.send('Ah, super!')
        return

    await client.process_commands(message)


@client.command(name='help', description='ask for help')
async def help(ctx):
    await ctx.send('Sorry, I can\'t help you. Ah, that\'s life. I think <@377907768071553024> over there can though.',
                   allowed_mentions=discord.AllowedMentions(everyone=False, users=False, roles=False))


@client.command(name='closeofbusiness', description='leave', aliases=['leave', 'close'])
async def closeofbusiness(ctx):
    await ctx.send('Take care!')


@client.command(name='startofclass', description='start class', aliases=['start', 'startofdice'])
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


@client.command(name='rollthedice', description='roll dice', aliases=['dice'])
async def rollthedice(ctx):
    dice = random.randint(1, 6)
    if dice == 1:
        message = f'Everyone ready for the bell ringer? Time to roll the dice! It\'s a {dice}! Turn \'em in!' \
                  f' I will rip them out of your hands now. Who got A? '
    else:
        message = f'Everyone ready for the bell ringer? Time to roll the dice! It\'s a {dice}! Who got A? '
    ans = random.randint(1, 5)
    letters = ['A', 'B', 'C', 'D', 'E']
    source = ['Benchmarking', 'Crowd-sourcing', 'The back of the book', 'The textbook', 'CodeHS']
    if ans > 1:
        message += 'Who got B? '
    if ans > 2:
        message += 'Who got C? '
    if ans > 3:
        message += 'Who got D? '
    if ans > 4:
        message += 'Who got E? '
    message += f'{random.choice(source)} says the answer is {letters[ans - 1]}! '
    message += 'Super!'
    if random.randint(0, 5) == 0:
        message += ' Onguh free free!'
    await ctx.send(message)


@client.command(name='lecturetime', description='it\'s lecture time', aliases=['lecture'])
async def lecturetime(ctx):
    lectures = ['Unit 1 Primitive Types', 'Unit 2 Using Objects', 'Unit 3 Boolean Expression and if Statements',
                'Unit 4 Iteration', 'Unit 5 Writing Classes', 'Unit 6 Array', 'Unit 7 ArrayList', 'Unit 8 2DArray',
                'Unit 9 Inheritance', 'Unit 10 Recursion']
    lessons = ['What all computer user face?', 'The power to be proactive', 'You if seen this, but are stressed',
               'Which RDP does being proactive fall under?', '"Hey! Look what Zog do!"', 'How to tame email',
               'You have a knowledge of time management, but no an understanding', 'Benchmarking successful people',
               'How to manage files', 'The power of Win-Win', 'Win - Win - Teams', 'Why this is powerful?',
               'Lack of planning = easy jobs are easy', 'Lack of planning = hard projects get hard', 'ChatGDP!']
    message = f'Alright! Lecture time. {random.choice(lectures)}! But first, better today then yesterday! ' \
              f'{random.choice(lessons)}'
    await ctx.send(message)


@client.command(name='chinese', description='ask woodard if he knows anything foreign', aliases=['foreignlanguage', 'black', 'muslim', 'hispanic', 'arab', 'korean'])
async def chinese(ctx):
    ethnicities = ['Chinese', 'Japanese', 'Korean', 'Vietnamese', 'Filipino', 'Indian', 'Thai', 'Indonesian', 'African',
                   'South American', 'Native American', 'Middle Eastern', 'Muslim', 'Black', 'Hispanic', 'Minority']
    await ctx.send(f'Konichiwa! Yes, I am friends with a lot of {random.choice(ethnicities)} people.')


@client.command(name='slides', description='create slides')
async def slides(ctx, *, msg: str):
    image = Image.open('assets/vapor_trail.png')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('assets/verdana.ttf', 75)
    draw.text((100, 200), msg, fill=(255, 255, 255), font=font)
    image.save('_slides.png')
    with open('_slides.png', 'rb') as f:
        upload_image = discord.File(f, spoiler=False)
    await ctx.send(file=upload_image)


@tasks.loop(time=datetime.time(hour=18, minute=30))
async def super_every_day():
    await client.wait_until_ready()
    await client.get_channel(1013977098370699305).send('Ah, super!')


async def main():
    super_every_day.start()
    await client.start(token)


if __name__ == '__main__':
    asyncio.run(main())
