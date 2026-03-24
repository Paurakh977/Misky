import json
import base64
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model_path = hf_hub_download(
    repo_id="LiquidAI/LFM2.5-VL-1.6B-GGUF",
    filename="LFM2.5-VL-1.6B-Q4_0.gguf"
)

llm = Llama(
    model_path=model_path,
    n_gpu_layers=0,
    n_threads=4,
    n_ctx=2048,
    verbose=False
)
print("Model ready!")

@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class ChatMessage(BaseModel):
    role: str
    content: str | list

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@app.post("/chat")
async def chat(request: ChatRequest):
    def generate():
        messages_dict = [msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() for msg in request.messages]
        stream = llm.create_chat_completion(
            messages=messages_dict,
            max_tokens=4000,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)