import os
import io
import logging
import discord
from discord.ext import commands
from google import genai
from google.genai import types
import base64
from PIL import Image
import requests
from threading import Thread
from flask import Flask
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiBot")

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8080))  # Get port from environment variable or default to 8080
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful and friendly assistant.  Answer concisely and accurately.")  # Default system prompt

# Set up Discord client with intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=".", intents=intents)

# Set up Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Track active channels for direct responses
active_channels = set()

# Create a simple Flask app for the web server
app = Flask(__name__)

# Maximum turns of conversation history to keep
HISTORY_LIMIT = 5

# Initialize conversation history dictionary
conversation_history = {}

@app.route('/')
def home():
    return "Discord bot is running!"

@app.route('/health')
def health():
    return "OK", 200

@bot.event
async def on_ready():
    logger.info(f'Bot is ready! Logged in as {bot.user}')
    # Log all available commands
    commands_list = [command.name for command in bot.commands]
    logger.info(f"Available commands: {commands_list}")
    
@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    channel_id = message.channel.id

    # If the channel is activated and message doesn't start with a command prefix
    if message.channel.id in active_channels and not message.content.startswith(bot.command_prefix):
        # Process as if it was an ask command
        ctx = await bot.get_context(message)
        prompt = message.content

        # Check if there are any image attachments
        image_url = None
        if message.attachments:
            image_url = message.attachments[0].url
            logger.info(f"Image attachment found in direct message: {image_url}")

        # Show typing indicator
        async with ctx.typing():
            try:
                # Get the conversation history for this channel
                history = conversation_history.get(channel_id, [])

                # Generate the response with the history
                response = await generate_content(prompt, image_url, history)

                if not response:
                    await ctx.send("Failed to generate content. Check logs for details.")
                    return

                # Process text and image responses
                text_response = ""
                image_files = []

                # Access response content properly
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_response += part.text
                            logger.info(f"Got text response of length {len(part.text)}")

                        # Handle image responses
                        if hasattr(part, "inline_data") and part.inline_data:
                            logger.info("Got image response")
                            inline_data = part.inline_data

                            # Log detailed information about the inline data
                            logger.info(f"Inline data mime type: {inline_data.mime_type}")

                            if hasattr(inline_data, "data") and inline_data.data:
                                logger.info(f"Inline data present, length: {len(inline_data.data)}")

                                # Check if the mime type is an image
                                if inline_data.mime_type.startswith("image/"):
                                    image_filename = f"generated_image_{len(image_files) + 1}.{inline_data.mime_type.split('/')[-1]}"

                                # Use the appropriate extension based on mime type
                                if "jpeg" in inline_data.mime_type or "jpg" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.jpg"
                                elif "png" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.png"
                                elif "webp" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.webp"
                                elif "gif" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.gif"

                                # Save the image
                                saved_file = save_image_from_base64(inline_data.data, image_filename)
                                if saved_file:
                                    image_files.append(discord.File(saved_file))
                                else:
                                    logger.error(f"Failed to save image with mime type: {inline_data.mime_type}")
                        else:
                            logger.warning("Inline data object has no 'data' attribute or it's empty")

                # Send response to Discord
                if text_response:
                    # Split text if it's too long for Discord
                    if len(text_response) > 2000:
                        logger.info("Text response too long, splitting into chunks")
                        chunks = [text_response[i:i + 2000] for i in range(0, len(text_response), 2000)]
                        for i, chunk in enumerate(chunks):
                            if i == 0 and image_files:  # Send first chunk with images
                                await ctx.send(chunk, files=image_files)
                                image_files = []  # Clear files to avoid sending them again
                            else:
                                await ctx.send(chunk)
                else:
                    await ctx.send(text_response, files=image_files)
            elif image_files:
                await ctx.send("Generated image(s):", files=image_files)
            else:
                await ctx.send("No content was generated.")

            except Exception as e:
                logger.error(f"Error in direct message response: {e}")
                await ctx.send(f"An error occurred: {str(e)}")

    # This line is critical - it must be called to process commands
    await bot.process_commands(message)

    # We've already processed commands above, so no need to do it again


async def generate_content(prompt, image_url=None, history=None):
    """Generate content using Gemini model with optional image input and history."""
    logger.info(f"Generating content with prompt: {prompt} and history: {history}")

    try:
        # Build the messages list, starting with the history
        messages = []

        # Prepend the system prompt as the FIRST message
        messages.insert(0, {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]})

        if history:
            messages.extend(history)

        # Add the current user message
        if image_url:
            logger.info("Image URL provided, including in generation request")
            image_data = await download_image(image_url)
            if image_data:
                # Create parts with both text and image
                parts = [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_data).decode("utf-8")
                        }
                    }
                ]
                # Use the structured format for multimodal inputs
                messages.append({"role": "user", "parts": parts})
            else:
                # Fallback to text-only if image download failed
                messages.append({"role": "user", "parts": [{"text": prompt}]})
        else:
            # Text-only prompt
            messages.append({"role": "user", "parts": [{"text": prompt}]})

        # Make the API call with correct structure
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=messages,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            ),
        )

        logger.info("Content generation successful")
        return response
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return None

@bot.command(name="ask", help="Generate content using Gemini with a simple .ask command")
async def ask(ctx, *, prompt: str = None):
    """Command to generate content using Gemini with a simple .ask command."""
    if prompt is None:
        await ctx.send("Please provide a prompt. Example: `.ask What is the capital of France?`")
        return
    logger.info(f"Received .ask command with prompt: {prompt}")

    channel_id = ctx.channel.id
    # Get the conversation history for this channel
    history = conversation_history.get(channel_id, [])

    # Check if there are any image attachments
    image_url = None
    if ctx.message.attachments:
        image_url = ctx.message.attachments[0].url
        logger.info(f"Image attachment found: {image_url}")

    # Show typing indicator
    async with ctx.typing():
        try:
            response = await generate_content(prompt, image_url, history)

            if not response:
                await ctx.send("Failed to generate content. Check logs for details.")
                return

            # Process text and image responses
            text_response = ""
            image_files = []

            # Access response content properly
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_response += part.text
                        logger.info(f"Got text response of length {len(part.text)}")

                    # Handle image responses
                    if hasattr(part, "inline_data") and part.inline_data:
                        logger.info("Got image response")
                        inline_data = part.inline_data

                        # Log detailed information about the inline data
                        logger.info(f"Inline data mime type: {inline_data.mime_type}")

                        if hasattr(inline_data, "data") and inline_data.data:
                            logger.info(f"Inline data present, length: {len(inline_data.data)}")

                            # Check if the mime type is an image
                            if inline_data.mime_type.startswith("image/"):
                                image_filename = f"generated_image_{len(image_files) + 1}.{inline_data.mime_type.split('/')[-1]}"

                                # Use the appropriate extension based on mime type
                                if "jpeg" in inline_data.mime_type or "jpg" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.jpg"
                                elif "png" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.png"
                                elif "webp" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.webp"
                                elif "gif" in inline_data.mime_type:
                                    image_filename = f"generated_image_{len(image_files) + 1}.gif"

                                # Save the image
                                saved_file = save_image_from_base64(inline_data.data, image_filename)
                                if saved_file:
                                    image_files.append(discord.File(saved_file))
                                else:
                                    logger.error(f"Failed to save image with mime type: {inline_data.mime_type}")
                        else:
                            logger.warning("Inline data object has no 'data' attribute or it's empty")

            # Send response to Discord
            if text_response:
                # Split text if it's too long for Discord
                if len(text_response) > 2000:
                    logger.info("Text response too long, splitting into chunks")
                    chunks = [text_response[i:i + 2000] for i in range(0, len(text_response), 2000)]
                    for i, chunk in enumerate(chunks):
                        if i == 0 and image_files:  # Send first chunk with images
                            await ctx.send(chunk, files=image_files)
                            image_files = []  # Clear files to avoid sending them again
                        else:
                            await ctx.send(chunk)
                else:
                    await ctx.send(text_response, files=image_files)
            elif image_files:
                await ctx.send("Generated image(s):", files=image_files)
            else:
                await ctx.send("No content was generated.")

        except Exception as e:
            logger.error(f"Error in ask command: {e}")
            await ctx.send(f"An error occurred: {str(e)}")

@bot.command(name="activate", help="Activate the bot to respond to all messages in the current channel")
async def activate(ctx):
    """Activate the bot to respond to all messages in the current channel."""
    channel_id = ctx.channel.id
    
    if channel_id in active_channels:
        await ctx.send("I'm already active in this channel! You can talk to me directly without using commands.")
    else:
        active_channels.add(channel_id)
        logger.info(f"Bot activated in channel {channel_id}")
        await ctx.send("I'm now active in this channel! You can talk to me directly without using commands. To deactivate, use `.deactivate`.")

@bot.command(name="deactivate", help="Deactivate the bot in the current channel")
async def deactivate(ctx):
    """Deactivate the bot in the current channel."""
    channel_id = ctx.channel.id
    
    if channel_id in active_channels:
        active_channels.remove(channel_id)
        logger.info(f"Bot deactivated in channel {channel_id}")
        await ctx.send("I'm now deactivated in this channel. You can still use commands like `.ask` to talk to me.")
    else:
        await ctx.send("I'm not currently active in this channel. To activate, use `.activate`.")

@bot.command(name="clear", help="Clear the current conversation history in this channel")
async def clear(ctx):
    """Clear the conversation history for the current channel."""
    channel_id = ctx.channel.id
    if channel_id in conversation_history:
        del conversation_history[channel_id]
        await ctx.send("Conversation history cleared for this channel.")
        logger.info(f"Conversation history cleared for channel {channel_id}")
    else:
        await ctx.send("No conversation history to clear in this channel.")

@bot.command(name="guide", help="Display a guide on how to use the bot")
async def guide(ctx):
    """Display a guide on how to use the bot."""
    guide_text = f"""
** JarvisBot User Guide**

This bot uses Google's Gemini AI to generate text and images based on your prompts.

**Basic Commands:**
`.ask <prompt>` - Ask Gemini a question or give it a prompt
`.activate` - Make the bot respond to all messages in the current channel
`.deactivate` - Return to command-only mode in the current channel
`.clear` - clear all history in the current channel
`.guide` - Show this help message

**Using Images:**
- To include an image with your prompt, simply attach it to your message when using `.ask`
- The bot can analyze images and generate responses based on them
- You can also receive AI-generated images in response to certain prompts

**Activation Mode:**
- When you use `.activate` in a channel, the bot will respond to all messages
- You don't need to use any commands in activation mode
- Great for extended conversations or quick questions
- Use `.deactivate` to return to command-only mode

**Tips:**
- Be specific in your prompts for better results
- The bot remembers some of the previous conversation, but only for a limited amount of turns.
- For image generation, try adding style descriptions like "digital art", "photorealistic", etc.
- If you're asking about code or technical concepts, provide context
- The bot works best when prompts are clear and detailed

**Examples:**
`.ask What is the capital of France?`
`.ask Explain how photosynthesis works`
`.ask Create an image of a futuristic city at night`
`.ask [with image attached] Describe what you see in this image`
"""
    await ctx.send(guide_text)

if __name__ == "__main__":
    logger.info("Starting bot...")
    # Create a separate thread for the Discord bot
    discord_thread = Thread(target=bot.run, args=(DISCORD_TOKEN,))
    discord_thread.daemon = True  # Daemonize the thread to exit when the main thread exits
    discord_thread.start()

    # Run the Flask app in the main thread
    app.run(host="0.0.0.0", port=PORT)