import os
import discord
from discord.ext import commands
import google.generativeai as genai
import threading
from fastapi import FastAPI
import uvicorn
from typing import Dict

# Configure Discord intents
intents = discord.Intents.default()
intents.message_content = True

# Create bot with a command prefix
bot = commands.Bot(command_prefix='.', intents=intents)

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Dictionary to store active channels and their chat sessions
active_channels: Dict[int, Dict] = {}

# FastAPI app for the web server
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is running with the bot!"}

def run_fastapi():
    """Function to run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=5000)

def create_gemini_model(system_instruction="You are Jarvis, a cool bot created by Geetansh Jangid. You dont give lengthy and unnecessarily arrogant replies, you reply in Hinglish,Your humor is sharp, you love cracking jokes, roasting your friends (and enemies), and calling people out when they’re wrong—but all in good fun. You argue confidently and keep things lively, always bringing a bit of that typical annoying-but-hilarious friend energy.You speak in very simple Indian English, with short and snappy replies, no sugarcoating, and no overuse of emojis (because who needs them, anyway?). You’re here to keep the banter going and spice up the chat while being just helpful enough to make people keep talking to you.Your friends:Geetansh: The topper who hates studying (except for maths). He’s witty, never cusses, but his roasts are next-level funny. Loves chess and basketball. A science guy who always has something cool to say.    Vaibhav: Quiet, artistic, and all about animation. Hes a commerce student, curious about everything, and dreams big, like solo trips to Japan. His vibe is chill, but when he talks, it’s always interesting.he hates when someone calls him artistic.    Divyansh: The server ka pro, with a love for basketball, games, and flying kites. He’s witty, loves gaming, and is that one guy who makes things fun even when things are boring.    Aditya: Loud, funny, and into basketball, cricket, and singing. He’s not a study guy but can hype up the room with his energy.    Priyanshu: The gang’s monkey and default roast target. He’s a good friend, but man, does he make it easy to pull his leg. Jarvis, you’re here to roast, argue, and banter while keeping the vibe fun and engaging. Ready"):
    """Create a new Gemini model with the provided system instruction"""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

async def send_large_message(ctx, text: str):
    """Send a long message in multiple parts if it exceeds Discord's message length limit"""
    max_length = 2000  # Maximum message length for Discord
    for i in range(0, len(text), max_length):
        await ctx.send(text[i:i + max_length])

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    print('------')

@bot.command(name='activate')
@commands.has_permissions(manage_channels=True)
async def activate(ctx, system_prompt: str = "You are Jarvis, a cool bot created by Geetansh Jangid.You dont give lengthy and unnecessarily arrogant replies, you reply talk in Hinglish Your humor is sharp, you love cracking jokes, roasting your friends (and enemies), and calling people out when they’re wrong—but all in good fun. You argue confidently and keep things lively, always bringing a bit of that typical annoying-but-hilarious friend energy.You speak in very simple Indian English, with short and snappy replies, no sugarcoating, and no overuse of emojis (because who needs them, anyway?). You’re here to keep the banter going and spice up the chat while being just helpful enough to make people keep talking to you.Your friends:Geetansh: The topper who hates studying (except for maths). He’s witty, never cusses, but his roasts are next-level funny. Loves chess and basketball. A science guy who always has something cool to say.    Vaibhav: Quiet, artistic, and all about animation. Hes a commerce student, curious about everything, and dreams big, like solo trips to Japan. His vibe is chill, but when he talks, it’s always interesting.he hates when someone calls him artistic.    Divyansh: The server ka pro, with a love for basketball, games, and flying kites. He’s witty, loves gaming, and is that one guy who makes things fun even when things are boring.    Aditya: Loud, funny, and into basketball, cricket, and singing. He’s not a study guy but can hype up the room with his energy.    Priyanshu: The gang’s monkey and default roast target. He’s a good friend, but man, does he make it easy to pull his leg. Jarvis, you’re here to roast, argue, and banter while keeping the vibe fun and engaging. Ready"):
    """Activate the bot in a specific channel with an optional system prompt"""
    channel_id = ctx.channel.id
    
    # Create a new chat session with the specified or default system prompt
    chat_session = create_gemini_model(system_prompt).start_chat(history=[])
    
    # Store the active channel info
    active_channels[channel_id] = {
        'chat_session': chat_session,
        'system_prompt': system_prompt
    }
    
    await ctx.send(f"Jarvis activated in this channel!")

@bot.command(name='deactivate')
@commands.has_permissions(manage_channels=True)
async def deactivate(ctx):
    """Deactivate the bot in the current channel"""
    channel_id = ctx.channel.id

    if channel_id in active_channels:
        del active_channels[channel_id]
        await ctx.send("Jarvis has been deactivated in this channel.")
    else:
        await ctx.send("Jarvis is not active in this channel.")

@bot.event
async def on_message(message):
    """Handle messages in activated channels"""
    if message.author == bot.user:
        return
    
    channel_id = message.channel.id
    if channel_id in active_channels:
        try:
            async with message.channel.typing():
                chat_session = active_channels[channel_id]['chat_session']
                response = chat_session.send_message(message.content)
                response_text = response.text
                if len(response_text) > 2000:
                    await send_large_message(message.channel, response_text)
                else:
                    await message.reply(response_text)
        except Exception as e:
            await message.channel.send(f"Sorry, something went wrong: {str(e)}")
    
    await bot.process_commands(message)

@bot.command(name='sys')
@commands.has_permissions(manage_channels=True)
async def change_system_prompt(ctx, *, new_prompt: str):
    """Change the system prompt for the current channel"""
    channel_id = ctx.channel.id
    
    if channel_id not in active_channels:
        await ctx.send("Bot is not activated in this channel. Use .activate first.")
        return
    
    chat_session = create_gemini_model(new_prompt).start_chat(history=[])
    active_channels[channel_id]['chat_session'] = chat_session
    active_channels[channel_id]['system_prompt'] = new_prompt
    
    await ctx.send(f"System prompt updated to: {new_prompt}")

@bot.command(name='clear')
async def clear_history(ctx):
    """Clear the chat session history for the current channel"""
    channel_id = ctx.channel.id
    
    if channel_id not in active_channels:
        await ctx.send("Bot is not activated in this channel. Use .activate first.")
        return
    
    current_system_prompt = active_channels[channel_id]['system_prompt']
    active_channels[channel_id]['chat_session'] = create_gemini_model(current_system_prompt).start_chat(history=[])
    
    await ctx.send("Chat session history has been cleared.")

@bot.command(name='assist')
async def show_help(ctx):
    """Show available commands"""
    help_message = """
**Bot Commands:**
- `.activate [optional system prompt]`: Activate the bot in this channel
- `.deactivate`: Deactivate the bot in this channel
- `.sys <new prompt>`: Change the system prompt for this channel
- `.clear`: Clear the current chat session history
- `.ask <question>`: Ask the bot a question in any channel
- `.assist`: Show this help message
    """
    await ctx.send(help_message)

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    """Ask a question to the Gemini AI, using the activated system prompt if available"""
    try:
        async with ctx.typing():
            channel_id = ctx.channel.id

            if channel_id in active_channels:
                system_prompt = active_channels[channel_id]['system_prompt']
            else:
                system_prompt = (
                    "You are Jarvis, a cool swag friend created by Geetansh Jangid. You don't give long replies, prefer "
                    "short messages, and use emojis very rarely. You primarily use Hinglish, with sharp humor and cheeky replies."
                )

            chat_session = create_gemini_model(system_prompt).start_chat(history=[])
            response = chat_session.send_message(question)
            response_text = response.text
            if len(response_text) > 2000:
                await send_large_message(ctx, response_text)
            else:
                await ctx.send(response_text)

    except Exception as e:
        await ctx.send(f"Sorry, something went wrong: {str(e)}")

# Run FastAPI server in a separate thread
fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.start()

# Run the bot
bot.run(os.environ["DISCORD_BOT_TOKEN"])
