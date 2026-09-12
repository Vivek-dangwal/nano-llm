import os
import sys
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from engine.core import MiniLLMEngine

app = FastAPI(title="Mini LLM Inference Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[Server] Initializing MiniLLMEngine...")
engine = MiniLLMEngine(quantize=True)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512

@app.get("/")
def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "SmolLM2-135M-Instruct (INT8 Quantized)",
        "engine": "Custom MiniLLMEngine with KV-Cache"
    }

@app.post("/generate/stream")
def generate_stream(request: ChatRequest):
    def event_generator():
        # Pass conversation history down to engine
        history = [{"role": m.role, "content": m.content} for m in request.messages]
        for token in engine.generate_chat_stream(history, max_new_tokens=request.max_tokens):
            yield token

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port)