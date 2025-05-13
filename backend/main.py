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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_4KT4q5_NgcrXDWLU9SBSJij9SNuPiH8b8qXcAQrkk3RCyE9KkGtGV4bwyRimrktkdEWjqG")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Check if keys are available
if not PERPLEXITY_API_KEY:
    print("WARNING: PERPLEXITY_API_KEY is not set. Web search functionality will not work.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. API calls to OpenAI will fail.")
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is not set. Interview search functionality will not work.")

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
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": "sonar-reasoning-pro",
        "messages": [{"role": "user", "content": query}],
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"Making Perplexity API request for query: {query}")
            response = await client.post(
                PERPLEXITY_URL,
                headers=headers,
                json=body
            )
            
            print(f"Perplexity API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
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
            else:
                error_message = f"Error in web search: Status {response.status_code}, Response: {response.text}"
                print(error_message)
                
                # If there's an error when streaming, send a simplified error message
                if stream_callback and callable(stream_callback):
                    simple_error = f"Error {response.status_code} occurred when searching."
                    await stream_callback(simple_error)
                    
                return error_message
    except Exception as e:
        error_message = f"Exception in web search: {str(e)}"
        print(error_message)
        
        # If there's an exception when streaming, send a simplified error message
        if stream_callback and callable(stream_callback):
            simple_error = f"Error occurred when searching: {str(e)[:50]}"
            await stream_callback(simple_error)
            
        return error_message

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
    
    # Add system message if provided
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    
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
            "messages": openai_messages,
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
                        error_text = await resp.text()
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
                                                    search_message = f"Searching for {query}..."
                                                    search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    
                                                    # Execute web search with streaming
                                                    async def stream_callback(chunk):
                                                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                        await yield_to_queue(f"{index}:\"{escaped_chunk}\"\n")
                                                    
                                                    search_result = await web_search(query, stream_callback=stream_callback)
                                                    
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
                                                "messages": new_messages,
                                                "stream": True
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
                                                        # Send our own [DONE] marker when the stream is complete
                                                        await yield_to_queue(f"0:\"done\"\n")
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
                                            error_text = await second_response.text()
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
                                        await yield_to_queue(f"0:\"done\"\n")
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
                    item = await asyncio.wait_for(output_queue.get(), timeout=5.0)  # Reduce timeout for faster detection of completion
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
                    if current_time - last_message_time > 15:
                        # Force completion after 15 seconds of inactivity
                        yield f"0:\"done\"\n"
                        break
                        
                    # Check if the task is done
                    if task.done():
                        # Send a final [DONE] if we didn't already
                        yield f"0:\"done\"\n"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)