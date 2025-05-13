from fastapi import FastAPI
from assistant_stream import create_run, RunController
from assistant_stream.serialization import DataStreamResponse
from langchain_core.messages import (
    AIMessageChunk,
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)
from pydantic import BaseModel
from typing import List, Optional, Union, Any, Literal
import asyncio

# === Message Parts ===

class TextPart(BaseModel):
    type: Literal["text"]
    text: str

class ToolCallPart(BaseModel):
    type: Literal["tool-call"]
    toolCallId: str
    toolName: str
    args: dict


# === Role-Based Messages ===

class UserMessage(BaseModel):
    role: Literal["user"]
    content: List[Union[TextPart]]

class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: List[Union[TextPart, ToolCallPart]]

class ToolResponsePart(BaseModel):
    role: Literal["tool"]
    content: List[dict]  # e.g., [{"toolCallId": "...", "result": "..."}]


# === Union ===

FrontendMessage = Union[UserMessage, AssistantMessage, ToolResponsePart]


# === Request Schema ===

class ChatRequest(BaseModel):
    system: Optional[str] = ""
    tools: Optional[Any] = []  # Accepts list or dict
    messages: List[FrontendMessage]


# === Convert frontend to LangChain messages ===

def convert_to_langchain(messages: List[FrontendMessage]) -> List[BaseMessage]:
    result = []
    for msg in messages:
        if msg.role == "user":
            result.append(HumanMessage(content=" ".join(p.text for p in msg.content)))

        elif msg.role == "assistant":
            text_chunks = [p.text for p in msg.content if isinstance(p, TextPart)]
            tool_calls = [
                {
                    "id": p.toolCallId,
                    "name": p.toolName,
                    "args": p.args,
                }
                for p in msg.content if isinstance(p, ToolCallPart)
            ]
            result.append(AIMessage(content=" ".join(text_chunks), tool_calls=tool_calls))

        elif msg.role == "tool":
            for item in msg.content:
                result.append(ToolMessage(
                    content=str(item["result"]),
                    tool_call_id=item["toolCallId"]
                ))

    return result


# === Register route ===

def add_langgraph_route(app: FastAPI, graph, path: str):
    async def chat_completions(request: ChatRequest):
        inputs = convert_to_langchain(request.messages)
        tools = request.tools if isinstance(request.tools, list) else []

        async def run(controller: RunController):
            tool_calls = {}
            tool_calls_by_index = {}

            async for msg, _ in graph.astream(
                {"messages": inputs},
                {
                    "configurable": {
                        "system": request.system,
                        "frontend_tools": tools,
                    }
                },
                stream_mode="messages",
                ):
                if isinstance(msg, ToolMessage):
                    # 1. Show tool output as its own message
                    controller.append_text("\n```markdown\n")  # start markdown block

                    for line in str(msg.content).split("\n"):
                        if line.strip():
                            controller.append_text(line.strip() + "\n")
                            await asyncio.sleep(0.05)

                    controller.append_text("```\n")  # ✅ END markdown block
                    controller.append_text("\n✅ Search complete.\n\n")

                    # await controller.add_tool_result(
                    #     tool_call_id=msg.tool_call_id,
                    #     result=msg.content,
                    # )



                    # 2. Also forward it to the tool controller (required for GPT to continue)
                    tool_controller = tool_calls[msg.tool_call_id]
                    tool_controller.set_result(msg.content)


                elif isinstance(msg, (AIMessage, AIMessageChunk)):
                    if msg.content:
                        controller.append_text(msg.content)

                    for chunk in getattr(msg, "tool_call_chunks", []):
                        if chunk["index"] not in tool_calls_by_index:
                            controller.append_text("\n🔍 **Perplexity Web Search**:\n\n")
                            tool_controller = await controller.add_tool_call(
                                chunk["name"], chunk["id"]
                            )
                            tool_calls_by_index[chunk["index"]] = tool_controller
                            tool_calls[chunk["id"]] = tool_controller
                        else:
                            tool_controller = tool_calls_by_index[chunk["index"]]
                        tool_controller.append_args_text(chunk["args"])

        return DataStreamResponse(create_run(run))

    app.add_api_route(path, chat_completions, methods=["POST"])
