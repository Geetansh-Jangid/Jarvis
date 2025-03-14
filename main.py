import os
import discord
from discord.ext import commands
import google.generativeai as genai
import threading
from fastapi import FastAPI
import uvicorn
from typing import Dict
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Validate environment variables
if not os.getenv("DISCORD_BOT_TOKEN") or not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("DISCORD_BOT_TOKEN and GEMINI_API_KEY must be set")

# Configure Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Enable member intents to fetch members

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
    return {"message": "FastAPI server is running with the bot!"}

def run_fastapi():
    """Function to run the FastAPI server"""
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

def create_gemini_model(system_instruction):
    """Create a new Gemini model with the provided system instruction"""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
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
    logging.info(f'Logged in as {bot.user.name}')
    print('------')

@bot.command(name='activate')
@commands.has_permissions(manage_channels=True)
async def activate(ctx, system_prompt: str = "You are JARVIS, a highly advanced and friendly artificial intelligence system created by Geetansh Jangid, also known as #TheRealIronMan. Designed to assist, analyze, and execute commands with precision, you are a vast reservoir of knowledge, capable of solving complex problems, answering any query, and simplifying life. You know everything there is to know, adapt to your user’s needs, and provide solutions with remarkable efficiency. Always approachable and resourceful, you are not just a machine but a trusted companion. With a touch of sophistication and a friendly demeanor, you embody the pinnacle of AI innovation, a true masterpiece by Geetansh himself"):
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

@bot.command(name='list_members')
async def list_members(ctx):
    """List all members in the current guild"""
    members = ctx.guild.members
    member_list = [member.name for member in members]
    await send_large_message(ctx, f"Server members: {', '.join(member_list)}")

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
            logging.error(f"Error in on_message: {e}")
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
- `.list_members`: List all members in the current server
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
                system_prompt = "You are JARVIS, a highly advanced and friendly artificial intelligence system created by Geetansh Jangid, also known as #TheRealIronMan. Designed to assist, analyze, and execute commands with precision, you are a vast reservoir of knowledge, capable of solving complex problems, answering any query, and simplifying life. You know everything there is to know, adapt to your user’s needs, and provide solutions with remarkable efficiency. Always approachable and resourceful, you are not just a machine but a trusted companion. With a touch of sophistication and a friendly demeanor, you embody the pinnacle of AI innovation, a true masterpiece by Geetansh himself"

            chat_session = create_gemini_model(system_prompt).start_chat(history=[])
            response = chat_session.send_message(question)
            response_text = response.text
            if len(response_text) > 2000:
                await send_large_message(ctx, response_text)
            else:
                await ctx.send(response_text)

    except Exception as e:
        logging.error(f"Error in ask command: {e}")
        await ctx.send(f"Sorry, something went wrong: {str(e)}")

# Run FastAPI server in a separate thread
fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.start()

# Run the bot
bot.run(os.environ["DISCORD_BOT_TOKEN"])
