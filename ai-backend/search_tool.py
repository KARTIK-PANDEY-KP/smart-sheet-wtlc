from langchain_core.tools import tool
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

perplexity_client = AsyncOpenAI(
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY")  # Load from environment variable
)

@tool
async def web_search(query: str) -> str:
    """Use Perplexity to search the web for real-time info."""
    # Add instruction to include URLs
    enhanced_query = f"{query}\n\nPlease include source URLs for any information provided."
    
    response = await perplexity_client.chat.completions.create(
        model="sonar-reasoning-pro",
        messages=[{"role": "user", "content": enhanced_query}],
        stream=False,
    )
    return response.choices[0].message.content
