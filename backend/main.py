import os
import json
from dotenv import load_dotenv
import httpx
import asyncio
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ─── LOAD .env ───────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Check if keys are available
if not PERPLEXITY_API_KEY:
    print("WARNING: PERPLEXITY_API_KEY is not set. Web search functionality will not work.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. API calls to OpenAI will fail.")

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
                                escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                await stream_callback(escaped_chunk)
                                await asyncio.sleep(0.1)  # Small delay for more natural streaming
                
                return result
            else:
                error_message = f"Error in web search: Status {response.status_code}, Response: {response.text}"
                print(error_message)
                return error_message
    except Exception as e:
        error_message = f"Exception in web search: {str(e)}"
        print(error_message)
        return error_message

@app.post("/api/my-custom-chat")
async def custom_chat(request: Request):
    payload = await request.json()
    system_prompt = payload.get("system", "")
    tools = payload.get("tools", [])
    user_messages = payload.get("messages", [])
    
    # Add default web search tool if not present
    if not tools:
        tools = [{
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
        }]
    
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
            "stream": True
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
                        await yield_to_queue(f"error:\"{error_text}\"\n")
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
                                                    
                                                    # Send initial search message
                                                    search_message = f"Searching for {query}..."
                                                    search_message = search_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    await yield_to_queue(f"{index}:\"{search_message}\"\n")
                                                    
                                                    # Create a streaming callback for Perplexity
                                                    async def stream_callback(chunk):
                                                        await yield_to_queue(f"{index}:\"{chunk}\"\n")
                                                    
                                                    # Execute web search with streaming
                                                    search_result = await web_search(query, stream_callback=stream_callback)
                                                    
                                                    # Add the function result to messages
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": search_result
                                                    })
                                                except Exception as e:
                                                    # Handle errors in function execution
                                                    error_message = f"Error executing web_search: {str(e)}"
                                                    error_message = error_message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                                    new_messages.append({
                                                        "role": "tool",
                                                        "tool_call_id": tool_call["id"],
                                                        "content": error_message
                                                    })
                                                    await yield_to_queue(f"{index}:\"{error_message}\"\n")
                                        
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
                                                        await yield_to_queue("[DONE]\n")
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
                                                                
                                                                # Check if this is the end of the response
                                                                # if second_choice.get("finish_reason") is not None:
                                                                #     # Send an explicit [DONE] marker
                                                                #     await yield_to_queue("[DONE]\n")
                                                    except json.JSONDecodeError:
                                                        # Skip invalid JSON
                                                        continue
                                        else:
                                            error_text = await second_response.text()
                                            error_text = error_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"error:\"{error_text}\"\n")
                                    
                                    # Get content from delta (for normal responses)
                                    elif "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Format as index:"content" with newline escaping
                                            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                                            await yield_to_queue(f"{index}:\"{escaped_content}\"\n")
                                            # Add a small delay between tokens for smoother streaming
                                            await asyncio.sleep(0.02)
                                    
                            except json.JSONDecodeError:
                                # Skip invalid JSON
                                continue
        
        # Start the processing task
        task = asyncio.create_task(process_api())
        
        # Yield from the queue as items become available
        try:
            while True:
                try:
                    item = await asyncio.wait_for(output_queue.get(), timeout=30.0)  # Reduce timeout for faster detection of completion
                    yield item
                    output_queue.task_done()
                    
                    # If this is the [DONE] marker, break the loop to close the connection
                    if item.strip() == "[DONE]":
                        # Give a little time for client to process the [DONE] marker
                        await asyncio.sleep(0.1)
                        break
                except asyncio.TimeoutError:
                    # Check if the task is done
                    if task.done():
                        # Send a final [DONE] if we didn't already
                        yield "[DONE]\n"
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
