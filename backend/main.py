import os
import json
from dotenv import load_dotenv
import httpx
import asyncio
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone
 # ─── LOAD .env ───────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
TABLE_API_URL = os.getenv("TABLE_API_URL", "http://localhost:8000/api/table/current")

# Check if keys are available
if not PERPLEXITY_API_KEY:
    print("WARNING: PERPLEXITY_API_KEY is not set. Web search functionality will not work.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. API calls to OpenAI will fail.")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is not set. Interview search functionality will not work.")

async def fetch_table_data():
    """Fetch current table data from the API"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            print(f"Fetching table data from {TABLE_API_URL}")
            response = await client.get(TABLE_API_URL)
            
            if response.status_code == 200:
                data = response.json()
                if "tableData" in data:
                    print("Successfully fetched table data")
                    return data
                else:
                    print("Warning: Response did not contain expected 'tableData' field")
                    return None
            else:
                print(f"Error fetching table data: Status {response.status_code}")
                return None
    except Exception as e:
        print(f"Exception while fetching table data: {str(e)}")
        return None

def create_global_system_prompt():
    """Create the global system prompt that will be used for all requests"""
    base_prompt = "" # enter anything here that needs to be globally sent to all the OpenAI calls
    
    # This function will be called synchronously but we'll have the table data cached
    # so it's not an issue
    return base_prompt

_cached_table_data = None
_last_table_fetch_time = 0

async def get_system_prompt_with_table_data():
    """Get the system prompt, optionally with table data if available"""
    global _cached_table_data, _last_table_fetch_time
    
    # Check if we need to refresh the cache (every 60 seconds)
    current_time = asyncio.get_event_loop().time()
    if current_time - _last_table_fetch_time > 60 or _cached_table_data is None:
        _cached_table_data = await fetch_table_data()
        _last_table_fetch_time = current_time
    
    base_prompt = ""
    
    # If we have table data, add it to the prompt
    if _cached_table_data and "tableData" in _cached_table_data:
        table_data = _cached_table_data["tableData"]
        if table_data and len(table_data) > 0:
            # Create a markdown table representation
            table_str = "Here is the current table data you can reference:\n\n"
            
            # Add headers
            if table_data and len(table_data) > 0:
                headers = table_data[0].keys()
                table_str += "| " + " | ".join(headers) + " |\n"
                table_str += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                
                # Add rows
                for row in table_data:
                    table_str += "| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |\n"
            
            # Add the table data to the prompt
            return f"{base_prompt}\n\n{table_str}\n\nRefer to this table data when answering questions about student information."
    
    # Return the base prompt if no table data
    return base_prompt

# Initialize Pinecone client
try:
    print(f"Initializing Pinecone with API key (first 8 chars): {PINECONE_API_KEY[:8]}... and environment: {PINECONE_ENV}")
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    
    print("Pinecone client initialized. Attempting to connect to index 'cq-transcripts-1'...")
    
    pinecone_index = pc.Index(
        name="cq-transcripts-1",
        pool_threads=50,
        connection_pool_maxsize=50,
    )
    
    # Test the connection by listing namespaces (collections)
    try:
        print("Testing Pinecone connection...")
        # Attempt to do a simple operation to verify the connection works
        describe_response = pinecone_index.describe_index_stats()
        print(f"Pinecone connection successful. Index stats: {describe_response}")
        
        # Get a list of namespaces
        namespaces = describe_response.get("namespaces", {})
        print(f"Available namespaces: {list(namespaces.keys())}")
        print("Pinecone initialized successfully.")
    except Exception as e:
        print(f"WARNING: Pinecone connection test failed: {str(e)}")
        # Still continue as the index might be valid
except Exception as e:
    print(f"ERROR initializing Pinecone: {str(e)}")
    pinecone_index = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def web_search(query, stream_callback=None):
    """Perform a web search using Perplexity API with optional streaming."""
    print(f"Starting web search for query: '{query}'")
    
    # Check if PERPLEXITY_API_KEY is available
    if not PERPLEXITY_API_KEY:
        print("PERPLEXITY_API_KEY is not set. Using fallback search.")
        if stream_callback and callable(stream_callback):
            await stream_callback("Web search API key not found. Using alternative search method...")
        return await fallback_search(query, stream_callback)
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": "sonar-reasoning-pro",
        "messages": [{"role": "user", "content": query}],
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # Increase timeout to 2 minutes
            print(f"Making Perplexity API request for query: {query}")
            print(f"Using API key (first 8 chars): {PERPLEXITY_API_KEY[:8]}...")
            
            try:
                print("Attempting to connect to Perplexity API...")
                response = await client.post(
                    PERPLEXITY_URL,
                    headers=headers,
                    json=body
                )
                print(f"Perplexity API call completed. Response status: {response.status_code}")
                
                # Dump headers for debugging
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"Response JSON keys: {data.keys()}")
                        
                        # Dump entire response for debugging (truncated)
                        response_dump = str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data)
                        print(f"Response dump: {response_dump}")
                        
                        if "choices" not in data or len(data["choices"]) == 0:
                            error_message = f"Unexpected response format: No choices in response. Full response: {data}"
                            print(error_message)
                            if stream_callback and callable(stream_callback):
                                await stream_callback("Error: Unexpected response format from search API")
                            return error_message
                        
                        result = data["choices"][0]["message"]["content"]
                        print(f"Perplexity search result (truncated): {result[:100]}...")
                        
                        # If streaming callback is provided, send chunks gradually
                        if stream_callback and callable(stream_callback):
                            # Split the result into sentences for more natural streaming
                            sentences = re.split(r'(?<=[.!?])\s+', result)
                            
                            # Stream each sentence with a small delay
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if sentence:
                                    # Further break down long sentences into smaller chunks
                                    chunk_size = 30  # characters per chunk
                                    for i in range(0, len(sentence), chunk_size):
                                        chunk = sentence[i:i+chunk_size]
                                        await stream_callback(chunk)
                                        await asyncio.sleep(0.1)  # Small delay for more natural streaming
                        
                        return result
                    except json.JSONDecodeError as e:
                        error_message = f"Error decoding JSON response: {str(e)}. Response content: {response.text[:500]}"
                        print(error_message)
                        if stream_callback and callable(stream_callback):
                            await stream_callback("Error: Invalid response from search API")
                        return error_message
                    except KeyError as e:
                        error_message = f"Key error in response: {str(e)}. Response content: {response.json()}"
                        print(error_message)
                        if stream_callback and callable(stream_callback):
                            await stream_callback(f"Error: Missing data in API response: {str(e)}")
                        return error_message
                else:
                    error_message = f"Error in web search: Status {response.status_code}"
                    if response.status_code == 401:
                        error_message += " - Unauthorized. Check your API key."
                    elif response.status_code == 403:
                        error_message += " - Forbidden. API key may be invalid or expired."
                    else:
                        try:
                            error_message += f", Response: {response.text}"
                        except Exception:
                            error_message += " (Could not read response body)"
                    
                    print(error_message)
                    
                    # If there's an error when streaming, use fallback search
                    print("Web search API failed. Using fallback search.")
                    if stream_callback and callable(stream_callback):
                        await stream_callback("Web search encountered an error. Using alternative search method...")
                    
                    return await fallback_search(query, stream_callback)
            except httpx.TimeoutException:
                error_message = "Search request timed out after 120 seconds"
                print(f"ERROR: {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback("Web search timed out. Using alternative search method...")
                return await fallback_search(query, stream_callback)
            except httpx.RequestError as exc:
                error_message = f"An error occurred while requesting from Perplexity API: {exc}"
                print(f"HTTPX REQUEST ERROR: {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Web search connection error. Using alternative search method... Error: {str(exc)[:50]}")
                return await fallback_search(query, stream_callback)
            except Exception as e:
                # Catch any other exceptions during the API call or initial processing
                error_message = f"Unexpected exception during Perplexity API call or initial response handling: {str(e)}"
                print(f"UNEXPECTED ERROR (during API call phase): {error_message}")
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Unexpected web search error. Using alternative search method... Error: {str(e)[:50]}")
                return await fallback_search(query, stream_callback)
                
    except Exception as e:
        error_message = f"Outer exception in web search: {str(e)}" # Renamed for clarity
        print(f"OUTER EXCEPTION (web_search): {error_message}")
        
        # If there's an exception when streaming, use fallback search
        if stream_callback and callable(stream_callback):
            await stream_callback(f"Web search error: {str(e)[:30]}... Using alternative search method...")
        
        return await fallback_search(query, stream_callback)

async def interview_search(query, company_name="innabox", stream_callback=None):
    """Search for interview transcripts using Pinecone."""
    print(f"Starting interview search for query: '{query}' in company: '{company_name}'")
    
    if not pinecone_index:
        error_message = "Pinecone is not initialized. Cannot perform interview search."
        print(error_message)
        
        # If there's an error when streaming, send a simplified error message
        if stream_callback and callable(stream_callback):
            await stream_callback("Error: Pinecone is not initialized")
            
        return error_message
    
    try:
        # Send initial message if streaming
        if stream_callback:
            await stream_callback(f"Searching interview transcripts for: {query}")
            await stream_callback("\nGenerating embeddings...")
        
        # Create embedding for query
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        embed_body = {
            "model": "text-embedding-3-small",
            "input": query
        }
        
        print(f"Generating embeddings for query: '{query}'")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            embed_response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=embed_body
            )
            
            print(f"Embedding API response status: {embed_response.status_code}")
            
            if embed_response.status_code != 200:
                error_message = f"Error generating embeddings: {embed_response.text}"
                print(error_message)
                
                # If there's an error when streaming, send a simplified error message
                if stream_callback and callable(stream_callback):
                    await stream_callback(f"Error: Unable to generate embeddings (status {embed_response.status_code})")
                    
                return error_message
            
            embed_data = embed_response.json()
            vector = embed_data["data"][0]["embedding"]
            
            # Log the vector dimensions to ensure it's valid
            print(f"Generated embedding vector of dimension: {len(vector)}")
            
            # Query Pinecone with the embedding
            if stream_callback:
                await stream_callback("\nSearching Pinecone index...")
            
            print(f"Querying Pinecone index with namespace: '{company_name}'")
            
            # Execute the Pinecone query
            response = pinecone_index.query(
                vector=vector,
                top_k=3,
                include_metadata=True,
                namespace=company_name
            )
            
            # Log the number of matches
            print(f"Pinecone query returned {len(response.matches)} matches")
            
            # Process and format results
            all_results = []
            formatted_results = ""
            
            if len(response.matches) == 0:
                no_results_message = f"No results found for '{query}' in company '{company_name}'"
                print(no_results_message)
                if stream_callback:
                    await stream_callback(f"\n{no_results_message}")
                return no_results_message
            
            for i, match in enumerate(response.matches):
                meta = match.metadata or {}
                
                # Pick the first available snippet field
                snippet = None
                for field in ("text", "snippet", "content", "transcript", "body"):
                    if field in meta:
                        snippet = meta[field]
                        break
                if snippet is None:
                    snippet = "<no snippet available>"
                
                # Pick a filename (or fallback to 'source')
                filename = meta.get("filename", meta.get("source", "unknown"))
                
                # Format the result
                formatted_match = f"[{i+1}] [{filename}]\n{snippet}\n\n"
                all_results.append({
                    "filename": filename,
                    "content": snippet,
                    "score": match.score
                })
                
                formatted_results += formatted_match
                
                # Stream this match if callback provided
                if stream_callback:
                    await stream_callback(f"\n[{i+1}] [{filename}]\n")
                    
                    # Stream the snippet in chunks for more natural flow
                    chunk_size = 40
                    snippet_chunks = [snippet[i:i+chunk_size] for i in range(0, len(snippet), chunk_size)]
                    for chunk in snippet_chunks:
                        await stream_callback(chunk)
                        await asyncio.sleep(0.05)
                    
                    await stream_callback("\n\n")
            
            # Return the full formatted results
            return formatted_results
            
    except Exception as e:
        error_message = f"Exception in interview search: {str(e)}"
        print(error_message)
        
        # If there's an exception when streaming, send a simplified error message
        if stream_callback and callable(stream_callback):
            await stream_callback(f"Error occurred during interview search: {str(e)[:50]}")
            
        return error_message

async def fallback_search(query, stream_callback=None):
    """Perform a fallback search using OpenAI when Perplexity fails."""
    print(f"Using fallback search for query: '{query}'")
    
    if not OPENAI_API_KEY:
        error_message = "OpenAI API key not available for fallback search."
        print(error_message)
        if stream_callback and callable(stream_callback):
            await stream_callback("Error: Unable to perform fallback search.")
        return error_message
    
    try:
        # Craft a prompt that asks for a factual response about the query
        prompt = f"""Please provide a factual, up-to-date summary about: {query}
        
Focus on providing accurate information only. Include key facts and figures if relevant.
If the information would require very recent data that might not be in your training data,
please mention that limitation.
        
Respond in a concise, informative manner without any filler text or disclaimers."""
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant providing factual information. Always address the user as KARTIK in your responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Keep temperature low for more factual responses
            "max_tokens": 800
        }
        
        if stream_callback:
            await stream_callback("Using AI to generate information about your query...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("Making OpenAI API request for fallback search")
            response = await client.post(
                OPENAI_URL,
                headers=headers,
                json=body
            )
            
            print(f"OpenAI API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                print(f"Fallback search result (truncated): {result[:100]}...")
                
                # Send each paragraph separately for more natural streaming
                if stream_callback and callable(stream_callback):
                    paragraphs = result.split("\n\n")
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            # Further chunk the paragraph for smoother streaming
                            chunk_size = 40
                            for i in range(0, len(paragraph), chunk_size):
                                chunk = paragraph[i:i+chunk_size]
                                await stream_callback(chunk)
                                await asyncio.sleep(0.05)
                            
                            # Add a newline between paragraphs
                            await stream_callback("\n\n")
                
                # Add a note about where the information came from
                note = "\n\nNote: This information was generated using AI and may not include the very latest developments."
                if stream_callback:
                    await stream_callback(note)
                
                return result + note
            else:
                error_message = f"Error in fallback search: Status {response.status_code}"
                print(error_message)
                if stream_callback:
                    await stream_callback("Unable to retrieve information from fallback search.")
                return error_message
    except Exception as e:
        error_message = f"Exception in fallback search: {str(e)}"
        print(error_message)
        if stream_callback:
            await stream_callback("An error occurred during the fallback search.")
        return error_message

@app.post("/api/my-custom-chat")
async def custom_chat(request: Request):
    payload = await request.json()
    system_prompt = payload.get("system", "")
    tools = payload.get("tools", [])
    user_messages = payload.get("messages", [])
    
    # Add default tools if not present
    if not tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information about any topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "interview_search",
                    "description": "Search through interview transcripts for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant interview transcript segments."
                            },
                            "company_name": {
                                "type": "string",
                                "description": "The company namespace to search within. Defaults to 'innabox'."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    # Convert message format from content array to simple text for OpenAI
    openai_messages = []
    
    # Get our enhanced system prompt with table data
    enhanced_system_prompt = await get_system_prompt_with_table_data()
    
    # Add system message - use enhanced prompt or user-provided one
    if system_prompt:
        # Combine our enhanced prompt with user-provided one
        openai_messages.append({"role": "system", "content": f"{enhanced_system_prompt}\n\n{system_prompt}"})
    else:
        openai_messages.append({"role": "system", "content": enhanced_system_prompt})
    
    # Convert user messages to OpenAI format
    for msg in user_messages:
        role = msg.get("role", "user")
        content_parts = msg.get("content", [])
        
        # Extract text parts and join them
        text_content = ""
        for part in content_parts:
            if part.get("type") == "text":
                text_content += part.get("text", "")
        
        openai_messages.append({"role": role, "content": text_content})

    async def custom_stream():
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "gpt-4o",
            "messages": openai_messages,  # We've already added the system message above
            "stream": True,
            # "max_tokens": 500  # Limit maximum response length
        }
        
        # Add tools if provided
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Create a reference to the yield function for callbacks
        output_queue = asyncio.Queue()
        
        async def yield_to_queue(content):
            await output_queue.put(content)
        
        # Process task
        async def process_api():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    OPENAI_URL,
                    headers=headers,
                    json=body
                ) as resp:
                    if resp.status_code != 200:
                        try:
                            # For streaming responses, we shouldn't access .text() directly
                            # Instead, read the response content manually
                            content = b""
                            async for chunk in resp.aiter_bytes():
                                content += chunk
                            error_text = content.decode('utf-8', errors='replace')
                        except Exception as e:
                            error_text = f"Error reading response (status {resp.status_code}): {str(e)}"
                        
                        error_text = error_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                        await yield_to_queue(f"0:\"{error_text}\"\n")
                        return
                    
                    # Variables to track tool calls
                    current_tool_calls = []
                    
                    async for line in resp.aiter_lines():
                        if line and line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            # Skip [DONE]
                            if line == "[DONE]":
                                continue
                            
                            try:
                                # Parse JSON response
                                data = json.loads(line)
                                
                                # Extract content delta if available
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    index = choice.get("index", 0)
                                    
                                    # Handle tool calls
                                    if "delta" in choice and "tool_calls" in choice["delta"]:
                                        tool_call_delta = choice["delta"]["tool_calls"]
                                        
                                        # Initialize tool calls list if this is the first delta
                                        if len(current_tool_calls) == 0 and len(tool_call_delta) > 0:
                                            for _ in range(len(tool_call_delta)):
                                                current_tool_calls.append({
                                                    "id": None,
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""}
                                                })
                                        
                                        # Update tool calls with delta information
                                        for i, call_delta in enumerate(tool_call_delta):
                                            if i < len(current_tool_calls):
                                                # Update tool call ID
                                                if "id" in call_delta:
                                                    current_tool_calls[i]["id"] = call_delta["id"]
                                                
                                                # Update function information
                                                if "function" in call_delta:
                                                    if "name" in call_delta["function"]:
                                                        current_tool_calls[i]["function"]["name"] = call_delta["function"]["name"]
                                                    if "arguments" in call_delta["function"]:
                                                        current_tool_calls[i]["function"]["arguments"] += call_delta["function"]["arguments"]
                                    
                                    # Execute tool calls when complete
                                    if choice.get("finish_reason") == "tool_calls" and len(current_tool_calls) > 0:
                                        # Create a new messages array with the previous messages
                                        new_messages = openai_messages.copy()
                                        
                                        # Add the assistant's message with the tool calls
                                        new_messages.append({
                                            "role": "assistant",
                                            "tool_calls": current_tool_calls
                                        })
                                        
                                        # Process each tool call
                                        for tool_call in current_tool_calls:
                                            function_name = tool_call["function"]["name"]
                                            
                                            if function_name == "web_search":
                                                try:
                                                    arguments = json.loads(tool_call["function"]["arguments"])
                                                    query = arguments.get("query", "")
                                                    
                                                    # Send initial search message - special format tag for tool start
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    
                                                    # Now send the actual search message
                                                    # search_message = f"Searching for {query}..."
                                                    # search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    # await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    
                                                    # Execute web search with streaming
                                                    async def stream_callback(chunk):
                                                        print(f"Web search chunk: {chunk[:50]}..." if len(chunk) > 50 else f"Web search chunk: {chunk}")
                                                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                        await yield_to_queue(f"{index}:\"{escaped_chunk}\"\n")

                                                        # Add a small delay to ensure chunks are properly sent and processed
                                                        await asyncio.sleep(0.05)
                                                    
                                                    # Now send the actual search message
                                                    search_message = f"Searching for {query}..."
                                                    search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    await asyncio.sleep(0.1)
                                                    
                                                    search_result = await web_search(query, stream_callback=stream_callback)
                                                    print(f"Web search complete. Result length: {len(search_result)}")
                                                    
                                                    # Check if the result is an error message or empty
                                                    if search_result.startswith("Error") or "error" in search_result.lower() or len(search_result.strip()) < 20:
                                                        print(f"Web search failed or returned minimal results: {search_result}")
                                                        
                                                        # Provide a useful fallback message to the user
                                                        fallback_message = f"I couldn't retrieve current search results for '{query}'. This might be due to API limitations. Please try again later or rephrase your query."
                                                        
                                                        # Stream the fallback message
                                                        await stream_callback(fallback_message)
                                                        
                                                        # Use the fallback message as the search result
                                                        search_result = fallback_message
                                                    
                                                    # Add the function result to messages
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": search_result
                                                    })
                                                    
                                                    # Send the tool end marker
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                                except Exception as e:
                                                    # Handle errors in function execution
                                                    error_message = f"Error executing web_search: {str(e)}"
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": error_message
                                                    })
                                                    
                                                    # Send error with tool start/end tags
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{error_message}\"\n")
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                            elif function_name == "interview_search":
                                                try:
                                                    arguments = json.loads(tool_call["function"]["arguments"])
                                                    query = arguments.get("query", "")
                                                    company_name = arguments.get("company_name", "innabox")
                                                    
                                                    # Send initial search message
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    
                                                    search_message = f"Searching interview transcripts for '{query}' in company '{company_name}'..."
                                                    search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    
                                                    # Execute interview search with streaming
                                                    async def stream_callback(chunk):
                                                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                        await yield_to_queue(f"{index}:\"{escaped_chunk}\"\n")
                                                    
                                                    search_result = await interview_search(query, company_name, stream_callback=stream_callback)
                                                    
                                                    # Add the function result to messages
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": search_result
                                                    })
                                                    
                                                    # Send the tool end marker
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                                except Exception as e:
                                                    # Handle errors in function execution
                                                    error_message = f"Error executing interview_search: {str(e)}"
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": error_message
                                                    })
                                                    
                                                    # Send error with tool start/end tags
                                                    await yield_to_queue(f"{index}:\"<<TOOL_START>>\"\n")
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{error_message}\"\n")
                                                    await yield_to_queue(f"{index}:\"<<TOOL_END>>\"\n")
                                        
                                        # Call OpenAI again with the results of the function call
                                        second_response = await client.post(
                                            OPENAI_URL,
                                            headers=headers,
                                            json={
                                                "model": "gpt-4o",
                                                "messages": new_messages,  # We now have the system message in new_messages already
                                                "stream": True,
                                                "temperature": 0.7,  # Add temperature to ensure valid responses
                                                "max_tokens": 1000  # Increase max_tokens to ensure complete responses
                                            }
                                        )
                                        
                                        # Add a delay before starting the second response streaming
                                        # This gives the UI time to process the search results
                                        await asyncio.sleep(0.5)
                                        
                                        # Add a separator to indicate the transition from search to AI response
                                        separator = "\n\nBased on the search results, here's my response:"
                                        separator = separator.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                        await yield_to_queue(f"{index}:\"{separator}\"\n")
                                        await asyncio.sleep(0.5)
                                        
                                        if second_response.status_code == 200:
                                            # Stream the second response
                                            finished_streaming = False
                                            async for second_line in second_response.aiter_lines():
                                                if second_line and second_line.startswith("data: "):
                                                    second_line = second_line[6:]
                                                    
                                                    if second_line == "[DONE]":
                                                        # Mark that we've finished streaming the response
                                                        finished_streaming = True
                                                        break  # Exit the streaming loop
                                                    
                                                    try:
                                                        second_data = json.loads(second_line)
                                                        
                                                        if "choices" in second_data and len(second_data["choices"]) > 0:
                                                            second_choice = second_data["choices"][0]
                                                            second_index = second_choice.get("index", 0)
                                                            
                                                            # Get content from delta
                                                            if "delta" in second_choice and "content" in second_choice["delta"]:
                                                                content = second_choice["delta"]["content"]
                                                                if content:
                                                                    # Format as index:"content" with newline escaping
                                                                    escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                                    await yield_to_queue(f"{second_index}:\"{escaped_content}\"\n")
                                                                    # Add a small delay between tokens for smoother streaming
                                                                    await asyncio.sleep(0.02)
                                                    except json.JSONDecodeError:
                                                        # Skip invalid JSON
                                                        continue
                                        else:
                                            try:
                                                # For streaming responses, we shouldn't access .text() directly
                                                # Instead, read the response content manually
                                                content = b""
                                                async for chunk in second_response.aiter_bytes():
                                                    content += chunk
                                                error_text = content.decode('utf-8', errors='replace')
                                            except Exception as e:
                                                error_text = f"Error reading response (status {second_response.status_code}): {str(e)}"
                                            
                                            error_text = error_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"0:\"{error_text}\"\n")
                                    
                                    # Get content from delta (for normal responses)
                                    elif "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Format as index:"content" with newline escaping
                                            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"{index}:\"{escaped_content}\"\n")
                                            # Add a small delay between tokens for smoother streaming
                                            await asyncio.sleep(0.02)
                                    
                                    # If finish_reason is stop, send [DONE] to terminate stream
                                    if choice.get("finish_reason") == "stop":
                                        break
                                    
                            except json.JSONDecodeError:
                                # Skip invalid JSON
                                continue
        
        # Start the processing task
        task = asyncio.create_task(process_api())
        
        # Yield from the queue as items become available
        try:
            last_message_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    item = await asyncio.wait_for(output_queue.get(), timeout=1.0)  # Reduce timeout for faster detection of completion
                    yield item
                    output_queue.task_done()
                    
                    # Update the last message time
                    last_message_time = asyncio.get_event_loop().time()
                    
                    # If this is the [DONE] marker, break the loop to close the connection
                    if item.strip() == "0:\"done\"":
                        # Give a little time for client to process the [DONE] marker
                        await asyncio.sleep(0.1)
                        break
                except asyncio.TimeoutError:
                    # Check if we've been idle too long (15 seconds)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_message_time > 50:
                        # Force completion after 15 seconds of inactivity
                        break
                        
                    # Check if the task is done
                    if task.done():
                        break
        finally:
            # Make sure to clean up the task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        # No need for an additional [DONE] at the end, as we've already sent it

    return StreamingResponse(
        custom_stream(),
        media_type="text/plain",
        headers={"Connection": "close"}  # Explicitly tell client to close the connection
    )

@app.get("/api/test-perplexity")
async def test_perplexity():
    """Test endpoint to verify Perplexity API connectivity."""
    if not PERPLEXITY_API_KEY:
        return {"status": "error", "message": "PERPLEXITY_API_KEY not set"}
    
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "sonar-reasoning-pro",
            "messages": [{"role": "user", "content": "What day is it today?"}],
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
                return {
                    "status": "success",
                    "message": "Perplexity API is working",
                    "response_code": response.status_code,
                    "result_sample": result[:100] + "..." if len(result) > 100 else result
                }
            else:
                return {
                    "status": "error",
                    "message": f"Perplexity API returned status code {response.status_code}",
                    "response": response.text
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception while testing Perplexity API: {str(e)}"
        }

@app.post("/api/update-perplexity-key")
async def update_perplexity_key(request: Request):
    """Update the Perplexity API key at runtime."""
    global PERPLEXITY_API_KEY
    
    try:
        data = await request.json()
        new_key = data.get("api_key")
        
        if not new_key:
            return {"status": "error", "message": "No API key provided"}
        
        # Store the old key to revert if testing fails
        old_key = PERPLEXITY_API_KEY
        
        # Update the key
        PERPLEXITY_API_KEY = new_key
        
        # Test the new key
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": "sonar-reasoning-pro",
            "messages": [{"role": "user", "content": "Test message"}],
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Perplexity API key updated and verified",
                    "key_preview": f"{PERPLEXITY_API_KEY[:8]}..."
                }
            else:
                # Revert to the old key if the new one doesn't work
                PERPLEXITY_API_KEY = old_key
                return {
                    "status": "error",
                    "message": f"New API key validation failed with status code {response.status_code}",
                    "response": response.text
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception while updating Perplexity API key: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)