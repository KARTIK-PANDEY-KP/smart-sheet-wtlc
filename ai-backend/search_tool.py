from langchain_core.tools import tool
from openai import AsyncOpenAI

perplexity_client = AsyncOpenAI(
    base_url="https://api.perplexity.ai",
    api_key="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"  # Replace with actual key
)

@tool
async def web_search(query: str) -> str:
    """Use Perplexity to search the web for real-time info."""
    response = await perplexity_client.chat.completions.create(
        model="sonar-reasoning-pro",
        messages=[{"role": "user", "content": query}],
        stream=False,
    )
    return response.choices[0].message.content
