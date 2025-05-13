from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai_graph import openai_perplexity_graph
from add_langgraph_route import add_langgraph_route

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom chat endpoint
add_langgraph_route(app, openai_perplexity_graph, "/api/my-custom-chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
