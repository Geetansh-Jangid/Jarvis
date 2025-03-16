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
                response = await generate_content(prompt, image_url)
                
                # Rest of your function...
                
            except Exception as e:
                logger.error(f"Error in direct message response: {e}")
                await ctx.send(f"An error occurred: {str(e)}")
    
    # This line is critical - it must be called to process commands
    await bot.process_commands(message)
    
    # We've already processed commands above, so no need to do it again

async def download_image(url):
    """Download an image from a URL and return it as bytes."""
    logger.info(f"Downloading image from {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return None

def save_image_from_base64(data, filename="generated_image.png"):
    """Save image data (either base64 string or raw bytes) to a file."""
    try:
        logger.info(f"Attempting to save image data (type: {type(data)}, length: {len(data)})")
        
        # Determine if we're dealing with actual base64 string or raw bytes
        if isinstance(data, str):
            # This is a base64 string - decode it
            logger.info("Processing as base64 string")
            
            # Fix padding issues if any
            padding_needed = len(data) % 4
            if padding_needed:
                data += '=' * (4 - padding_needed)
                logger.info(f"Added {4 - padding_needed} padding characters to base64 string")
            
            # Remove any potential data URL prefix
            if ',' in data:
                data = data.split(',', 1)[1]
                logger.info("Removed data URL prefix from base64 string")
            
            try:
                image_data = base64.b64decode(data)
                logger.info(f"Successfully decoded base64 data to binary (size: {len(image_data)} bytes)")
            except Exception as e:
                logger.error(f"Failed to decode base64 string: {e}")
                return None
        elif isinstance(data, bytes):
            # This appears to be raw binary data already
            logger.info("Processing as raw binary data")
            image_data = data
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            return None
            
        # Verify we have enough data for an image
        if len(image_data) < 100:
            logger.error(f"Data too small to be an image: {len(image_data)} bytes")
            return None
            
        # Now try to save the image using PIL
        bytes_io = io.BytesIO(image_data)
        bytes_io.seek(0)
        
        try:
            # Try to open as an image
            image = Image.open(bytes_io)
            image_format = image.format
            logger.info(f"Successfully identified image format: {image_format}")
            
            # If format is None, try to infer from filename or default to PNG
            if not image_format:
                image_format = filename.split('.')[-1].upper() if '.' in filename else 'PNG'
                logger.info(f"No format detected, using extension from filename: {image_format}")
            
            # Save the image
            image.save(filename, format=image_format)
            logger.info(f"Image saved as {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to process as image with PIL: {e}")
            
            # Fallback to direct file save
            try:
                logger.info("Attempting direct file save as fallback")
                with open(filename, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Direct save successful: {filename}")
                return filename
            except Exception as e2:
                logger.error(f"Direct file save failed: {e2}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None

async def generate_content(prompt, image_url=None):
    """Generate content using Gemini model with optional image input."""
    logger.info(f"Generating content with prompt: {prompt}")
    
    try:
        # Default structure for text-only prompts
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
                contents = {"role": "user", "parts": parts}
            else:
                # Fallback to text-only if image download failed
                contents = {"role": "user", "parts": [{"text": prompt}]}
        else:
            # Text-only prompt
            contents = {"role": "user", "parts": [{"text": prompt}]}
        
        # Make the API call with correct structure
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
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
    
    # Check if there are any image attachments
    image_url = None
    if ctx.message.attachments:
        image_url = ctx.message.attachments[0].url
        logger.info(f"Image attachment found: {image_url}")
    
    # Show typing indicator
    async with ctx.typing():
        try:
            response = await generate_content(prompt, image_url)
            
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
                    chunks = [text_response[i:i+2000] for i in range(0, len(text_response), 2000)]
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
            
            # Clean up temporary files
            for file in image_files:
                try:
                    os.remove(file.filename)
                    logger.info(f"Removed temporary file {file.filename}")
                except Exception as e:
                    logger.error(f"Error removing file {file.filename}: {e}")
                    
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

@bot.command(name="guide", help="Display a guide on how to use the bot")
async def guide(ctx):
    """Display a guide on how to use the bot."""
    guide_text = """
** JarvisBot User Guide**

This bot uses Google's Gemini AI to generate text and images based on your prompts.

**Basic Commands:**
`.ask <prompt>` - Ask Gemini a question or give it a prompt
`.activate` - Make the bot respond to all messages in the current channel
`.deactivate` - Return to command-only mode in the current channel
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
    
    # Define function to run Discord bot
    def run_discord_bot():
        bot.run(DISCORD_TOKEN)
    
    # Start Discord bot in a separate thread
    bot_thread = Thread(target=run_discord_bot)
    bot_thread.start()
    
    # Run Flask web server with proper host binding
    # '0.0.0.0' allows external connections (necessary for cloud platforms)
    logger.info(f"Starting web server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
