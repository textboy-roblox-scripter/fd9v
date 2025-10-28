import discord
from discord import app_commands
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch, asyncio, io
from aiohttp import web

DISCORD_TOKEN = "MTQzMjY0MzcwMTc5MjA0NzIwNw.GGijlh.8gD9e0hN2woM5jIgx-Y7S0MfmGUoCYR9-gHgic"
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
MAX_TOKENS = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
img_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

intents = discord.Intents.default()
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)
memory = {}

async def generate_text(prompt, user_id):
    context = memory.get(user_id, "")
    full_prompt = f"{context}\nUser: {prompt}\nAI:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.7)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.split("AI:")[-1].strip()
    memory[user_id] = "\n".join((context + f"\nUser:{prompt}\nAI:{reply}").splitlines()[-6:])
    return reply

@tree.command(name="ask", description="Ask the AI something")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    reply = await generate_text(question, interaction.user.id)
    await interaction.followup.send(reply)

@tree.command(name="imagine", description="Generate an image from text")
async def imagine(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    image = img_pipe(prompt).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    await interaction.followup.send(file=discord.File(buf, filename="result.png"))

@tree.command(name="clear", description="Clear chat memory")
async def clear(interaction: discord.Interaction):
    memory.pop(interaction.user.id, None)
    await interaction.response.send_message("Memory cleared.")

@tree.command(name="help", description="Show available commands")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(title="Paradise AI Help", color=0x5865F2)
    embed.add_field(name="/ask <message>", value="Chat with the AI", inline=False)
    embed.add_field(name="/imagine <prompt>", value="Create an image from text", inline=False)
    embed.add_field(name="/clear", value="Reset your conversation", inline=False)
    embed.add_field(name="/about", value="Learn about this bot", inline=False)
    await interaction.response.send_message(embed=embed)

@tree.command(name="about", description="Information about this bot")
async def about(interaction: discord.Interaction):
    reply = "I am **Paradise AI**, a large language model coded by **SoloXAssasin** to help you. Use `/help` to see available commands."
    await interaction.response.send_message(reply)

@bot.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {bot.user}")

# ---------------------- Website ----------------------

async def index(request):
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Paradise AI</title>
        <style>
            body { margin:0; font-family: 'Poppins', sans-serif; background:linear-gradient(135deg,#1e1f29,#2b2d42,#3a0ca3); color:white; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; }
            h1 { font-size:3rem; margin-bottom:0.5rem; }
            p { font-size:1.2rem; opacity:0.8; }
            a.button { margin-top:1rem; text-decoration:none; background:#5865F2; padding:0.8rem 1.5rem; border-radius:0.5rem; color:white; transition:0.3s; }
            a.button:hover { background:#4752C4; }
            footer { position:absolute; bottom:10px; font-size:0.9rem; opacity:0.6; }
        </style>
    </head>
    <body>
        <h1>🤖 Paradise AI is Online</h1>
        <p>Your Discord AI bot is running and ready.</p>
        <a href="https://discord.com" class="button">Join Discord</a>
        <footer>Powered by SoloXAssasin • Paradise AI</footer>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")

app = web.Application()
app.add_routes([web.get("/", index)])

async def start_all():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    await bot.start(DISCORD_TOKEN)

asyncio.run(start_all())
