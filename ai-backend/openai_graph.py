from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from agent_state import AgentState
from search_tool import web_search
from langgraph.prebuilt import ToolNode
from langchain.agents import Tool
import os

model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

tools = [web_search]
agent_with_tools = model.bind_tools(tools)

async def call_model(state, config):
    system = config["configurable"]["system"]
    messages = [SystemMessage(content=system)] + state["messages"]
    response = await agent_with_tools.ainvoke(messages)
    return {"messages": response}

async def run_tools(input, config, **kwargs):
    tool_node = ToolNode(tools=tools)
    return await tool_node.ainvoke(input, config, **kwargs)

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", run_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools", END})
graph.add_edge("tools", "agent")
openai_perplexity_graph = graph.compile()
