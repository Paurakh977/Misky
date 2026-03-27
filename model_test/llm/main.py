"""
LFM2.5-VL FastAPI Chat — powered by llama-server subprocess
============================================================
Requirements:
    pip install fastapi uvicorn httpx

Run:
    python main.py
Then open: http://localhost:8000
"""

import uvicorn
import base64
import subprocess
import threading
import time
import socket
import sys
import json
import httpx
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_DIR = Path(r"C:\Users\V16\Downloads\llama-b8533-bin-win-cpu-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"

MODEL_HF = "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0"
LLAMA_PORT = 8181
CTX_SIZE = 32768
MODEL_LOAD_TIMEOUT = 180

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="LFM2.5-VL Chat")

messages: list[dict] =[]
_server_proc: subprocess.Popen | None = None


# ── llama-server lifecycle ────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

def _start_llama_server() -> None:
    global _server_proc

    if not LLAMA_SERVER.exists():
        print(f"\n[ERROR] llama-server.exe not found at:\n  {LLAMA_SERVER}\n")
        sys.exit(1)

    cmd =[
        str(LLAMA_SERVER),
        "-hf", MODEL_HF,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "-c", str(CTX_SIZE),
        "-n", "4012",
        "-t", "4",
        "--image-max-tokens", "64",
        "--temp", "0.1",
        "--min-p", "0.15",
        "--repeat-penalty", "1.05",
        "--log-disable",
    ]

    print(f"[llama-server] Starting: {' '.join(cmd)}\n")
    _server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def _pipe_logs():
        for line in _server_proc.stdout:
            print(f"[llama-server] {line.rstrip()}")

    threading.Thread(target=_pipe_logs, daemon=True).start()

    print(f"[startup] Waiting for llama-server on port {LLAMA_PORT}...")
    deadline = time.time() + MODEL_LOAD_TIMEOUT
    while time.time() < deadline:
        if _port_open(LLAMA_PORT):
            print("[startup] ✓ llama-server is ready!\n")
            return
        time.sleep(2)

    print(f"[ERROR] llama-server did not become ready within {MODEL_LOAD_TIMEOUT}s.\n")
    sys.exit(1)


@app.on_event("startup")
async def _app_startup() -> None:
    t = threading.Thread(target=_start_llama_server, daemon=True)
    t.start()
    t.join()


@app.on_event("shutdown")
async def _app_shutdown() -> None:
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        print("[shutdown] llama-server terminated.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = HTML_CONTENT.replace("{{MODEL_NAME}}", MODEL_HF)
    html = html.replace("{{CTX_SIZE}}", str(CTX_SIZE))
    return html


@app.delete("/history")
async def clear_history():
    global messages
    messages =[]
    return {"status": "cleared"}


@app.post("/chat")
async def chat_endpoint(
    request: Request,
    prompt: str = Form(""),
    image: UploadFile = File(None),
):
    global messages

    content: list[dict] =[]
    if image and image.filename:
        image_bytes = await image.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = image.content_type or "image/jpeg"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    text = prompt.strip() or "(describe the image)"
    content.append({"type": "text", "text": text})

    messages.append({
        "role": "user",
        "content": content if (image and image.filename) else text,
    })

    async def stream_generator():
        global messages
        full_response = ""
        
        payload = {
            "model": "lfm2.5",
            "messages": messages,
            "max_tokens": 4012,
            "temperature": 0.1,
            "stream": True,
            "return_progress": True,      
            "timings_per_token": True     
        }

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload) as resp:
                    async for line in resp.aiter_lines():
                        
                        if await request.is_disconnected():
                            print("\n[INFO] Client Aborted! Terminating llama-server task.\n")
                            break
                        
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]" or not data_str:
                                yield f"data: {data_str}\n\n"
                                continue
                            
                            try:
                                data = json.loads(data_str)
                                
                                # 1. Catch live prompt evaluation progress payload (Input)
                                if "prompt_progress" in data:
                                    pp = data["prompt_progress"]
                                    total = pp.get("total") or 0
                                    processed = pp.get("processed") or 0
                                    
                                    time_ms = pp.get("time_ms")
                                    time_ms = float(time_ms) if time_ms is not None else 0.0
                                    
                                    prompt_sec = time_ms / 1000.0
                                    prompt_speed = (processed / prompt_sec) if prompt_sec > 0 else 0.0
                                    
                                    yield f"data: {json.dumps({'input_progress': {'processed': processed, 'total': total, 'time': round(prompt_sec, 2), 'speed': round(prompt_speed, 2)}})}\n\n"
                                
                                # 2. Catch native llama.cpp timings payload per token (Output)
                                if "timings" in data:
                                    t = data["timings"]
                                    
                                    prompt_ms = t.get("prompt_ms")
                                    prompt_ms = float(prompt_ms) if prompt_ms is not None else 0.0
                                    
                                    prompt_n = t.get("prompt_n")
                                    prompt_n = int(prompt_n) if prompt_n is not None else 0
                                    
                                    pred_ms = t.get("predicted_ms")
                                    pred_ms = float(pred_ms) if pred_ms is not None else 0.0
                                    
                                    pred_n = t.get("predicted_n")
                                    pred_n = int(pred_n) if pred_n is not None else 0
                                    
                                    prompt_sec = prompt_ms / 1000.0
                                    pred_sec = pred_ms / 1000.0
                                    
                                    # cached tokens kept by llama-server (previous context cached in memory)
                                    cache_n = t.get("cache_n")
                                    cache_n = int(cache_n) if cache_n is not None else 0
                                    
                                    # Fallbacks for `null` speeds returned natively
                                    prompt_speed = t.get("prompt_per_second")
                                    if prompt_speed is None:
                                        prompt_speed = (prompt_n / prompt_sec) if prompt_sec > 0 else 0.0
                                        
                                    pred_speed = t.get("predicted_per_second")
                                    if pred_speed is None:
                                        pred_speed = (pred_n / pred_sec) if pred_sec > 0 else 0.0
                                        
                                    total_ctx = prompt_n + cache_n + pred_n
                                    stats = {
                                      "prompt_n": prompt_n,
                                      "cache_n": cache_n,
                                      "prompt_time": round(prompt_sec, 2),
                                      "prompt_speed": round(float(prompt_speed), 2),
                                      "predicted_n": pred_n,
                                      "predicted_time": round(pred_sec, 2),
                                      "predicted_speed": round(float(pred_speed), 2),
                                      "total_ctx": total_ctx,
                                      "total_time": round(prompt_sec + pred_sec, 2)
                                    }
                                    yield f"data: {json.dumps({'stats': stats})}\n\n"
                                    
                                # 3. Extract Standard OAI Text Chunk
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {}).get("content", "")
                                    if delta:
                                        full_response += delta
                                        yield f"data: {json.dumps({'text': delta})}\n\n"

                            except json.JSONDecodeError:
                                pass

        except asyncio.CancelledError:
            print("\n[INFO] Request cancelled.\n")
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            if full_response:
                messages.append({"role": "assistant", "content": full_response})
            yield "data: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LFM2.5-VL · Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:        #0a0a0c;
    --surface:   #111116;
    --border:    #1e1e28;
    --accent:    #7b61ff;
    --accent2:   #00e5c3;
    --text:      #e2e2f0;
    --muted:     #5a5a78;
    --user-bg:   #16161f;
    --bot-bg:    #0e0e18;
    --danger:    #ff4f6b;
    --radius:    10px;
    --font-mono: 'JetBrains Mono', monospace;
    --font-disp: 'Syne', sans-serif;
  }
  html, body { height: 100%; background: var(--bg); color: var(--text);
    font-family: var(--font-mono); font-size: 14px; line-height: 1.65; }
  .app { display: grid; grid-template-rows: 56px 1fr auto; height: 100vh;
    max-width: 860px; margin: 0 auto; }
  header { display: flex; align-items: center; justify-content: space-between;
    padding: 0 20px; border-bottom: 1px solid var(--border);
    background: var(--bg); position: sticky; top: 0; z-index: 10; }
  .logo { font-family: var(--font-disp); font-weight: 800; font-size: 16px;
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px; }
  .logo-dot { width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent2); box-shadow: 0 0 8px var(--accent2);
    animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.85); } }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .model-tag { font-size: 11px; color: var(--muted); background: var(--surface);
    border: 1px solid var(--border); padding: 3px 8px; border-radius: 4px; }
  .btn-clear { font-family: var(--font-mono); font-size: 11px;
    background: transparent; border: 1px solid var(--border); color: var(--muted);
    padding: 4px 10px; border-radius: 4px; cursor: pointer; transition: .2s; }
  .btn-clear:hover { border-color: var(--danger); color: var(--danger); }
  #chat { overflow-y: auto; padding: 24px 20px; display: flex;
    flex-direction: column; gap: 4px; scroll-behavior: smooth; }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .msg { display: flex; flex-direction: column; max-width: 88%; animation: fadeUp .25s ease both; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  .msg.user { align-self: flex-end; }
  .msg.bot  { align-self: flex-start; width: 100%; }
  .msg-label { font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 4px; padding: 0 4px; }
  .msg.user .msg-label { text-align: right; color: var(--accent); }
  .msg.bot  .msg-label { color: var(--accent2); }
  .bubble { padding: 12px 16px; border-radius: var(--radius); white-space: pre-wrap; word-break: break-word; line-height: 1.7; display: inline-block; }
  .msg.user .bubble { background: var(--user-bg); border: 1px solid var(--border); border-bottom-right-radius: 2px; }
  .msg.bot  .bubble { background: var(--bot-bg); border: 1px solid #1a1a2e; border-bottom-left-radius: 2px; display: flex; flex-direction: column; }
  
  /* LIVE STATS PILL - Fixed Flex Alignment */
  .stats-bar { 
    display: flex; gap: 10px 18px; margin-top: 14px; padding: 10px 14px;
    background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 6px; 
    font-size: 11px; color: var(--muted); align-items: center; flex-wrap: wrap; 
    font-family: var(--font-mono); user-select: none; width: fit-content; line-height: 1;
  }
  .stats-item { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
  .status-badge { color: #f1c40f; display: flex; align-items: center; gap: 6px; font-weight: bold; }
  .stats-detail { opacity: 0.6; font-size: 10px; margin-left: 2px; }

  .cursor::after { content: '▋'; color: var(--accent2); animation: blink .7s step-start infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .msg-image { max-width: 220px; max-height: 160px; border-radius: 6px; margin-bottom: 8px; object-fit: cover; border: 1px solid var(--border); display: block; }
  .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; color: var(--muted); pointer-events: none; user-select: none; }
  .empty-state h2 { font-family: var(--font-disp); font-size: 22px; font-weight: 800; background: linear-gradient(135deg,var(--accent),var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .input-area { border-top: 1px solid var(--border); background: var(--bg); padding: 14px 20px 20px; display: flex; flex-direction: column; gap: 10px; }
  #preview-strip { display: none; align-items: center; gap: 10px; padding: 8px 10px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
  #preview-strip.visible { display: flex; }
  #img-preview { width: 48px; height: 48px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
  #preview-name { flex: 1; font-size: 11px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .btn-remove-img { background: transparent; border: none; color: var(--muted); cursor: pointer; font-size: 16px; padding: 2px 6px; transition: .2s; }
  .btn-remove-img:hover { color: var(--danger); }
  .prompt-row { display: flex; gap: 8px; align-items: flex-end; }
  .btn-attach { flex-shrink: 0; width: 40px; height: 40px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--muted); cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center; transition: .2s; }
  .btn-attach:hover { border-color: var(--accent); color: var(--accent); }
  #prompt { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); font-family: var(--font-mono); font-size: 13px; padding: 10px 14px; resize: none; min-height: 40px; max-height: 140px; outline: none; transition: .2s; overflow-y: auto; }
  #prompt:focus { border-color: var(--accent); }
  
  /* Buttons */
  .action-btn { flex-shrink: 0; height: 40px; padding: 0 18px; border-radius: var(--radius); font-family: var(--font-mono); font-size: 13px; font-weight: 500; cursor: pointer; transition: .2s; white-space: nowrap; }
  .btn-send { background: var(--accent); border: none; color: #fff; }
  .btn-send:hover:not(:disabled) { opacity: .85; }
  .btn-send:disabled { opacity: .35; cursor: not-allowed; }
  
  .btn-stop { background: transparent; border: 1px solid var(--danger); color: var(--danger); display: none; }
  .btn-stop:hover { background: rgba(255, 79, 107, 0.1); }
  
  .hint { font-size: 10px; color: var(--muted); text-align: center; letter-spacing: .03em; }
  @media (max-width: 600px) { .model-tag { display: none; } .msg { max-width: 96%; } .stats-bar { flex-direction: column; align-items: flex-start; gap: 8px; } }
</style>
</head>
<body>
<div class="app">

  <header>
    <div class="logo"><span class="logo-dot"></span>LFM2.5-VL</div>
    <div class="header-right">
      <span class="model-tag">1.6B · CPU · Vision ✓</span>
      <button class="btn-clear" onclick="clearHistory()">clear history</button>
    </div>
  </header>

  <div id="chat">
    <div class="empty-state" id="empty">
      <h2>LFM2.5-VL Chat</h2>
      <p>Send a message or attach an image to begin</p>
    </div>
  </div>

  <div class="input-area">
    <div id="preview-strip">
      <img id="img-preview" src="" alt="preview" />
      <span id="preview-name"></span>
      <button class="btn-remove-img" onclick="removeImage()">✕</button>
    </div>

    <div class="prompt-row">
      <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFile(event)" />
      <button class="btn-attach" onclick="document.getElementById('file-input').click()">📎</button>
      <textarea id="prompt" rows="1" placeholder="Ask anything…" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      
      <button class="action-btn btn-send" id="send-btn" onclick="sendMessage()">Send ↑</button>
      <button class="action-btn btn-stop" id="stop-btn" onclick="abortGeneration()">Stop 🛑</button>
    </div>
    <div class="hint">Enter to send · Shift+Enter for new line · images optional</div>
  </div>

</div>

<script>
  const MODEL_NAME = "{{MODEL_NAME}}";
  const CTX_SIZE   = {{CTX_SIZE}};
  
  let selectedFile = null;
  let isStreaming  = false;
  let abortController = null;
  let statsInterval = null;

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;
    document.getElementById('img-preview').src = URL.createObjectURL(file);
    document.getElementById('preview-name').textContent = file.name;
    document.getElementById('preview-strip').classList.add('visible');
  }

  function removeImage() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-strip').classList.remove('visible');
    document.getElementById('img-preview').src = '';
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function scrollBottom() {
    const chat = document.getElementById('chat');
    chat.scrollTop = chat.scrollHeight;
  }

  function appendMessage(role, text, imgSrc) {
    document.getElementById('empty').style.display = 'none';
    const chat  = document.getElementById('chat');
    const wrap  = document.createElement('div');
    wrap.className = `msg ${role}`;
    
    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = role === 'user' ? 'you' : 'assistant';
    wrap.appendChild(label);
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    if (imgSrc) {
      const img = document.createElement('img');
      img.src = imgSrc;
      img.className = 'msg-image';
      bubble.appendChild(img);
    }
    
    const textNode = document.createElement('span');
    textNode.textContent = text;
    bubble.appendChild(textNode);
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollBottom();
    return { bubble, textNode };
  }

  function abortGeneration() {
    if (abortController) {
      abortController.abort(); // Triggers FastAPI `is_disconnected()`
    }
  }

  async function sendMessage() {
    if (isStreaming) return;
    const promptEl = document.getElementById('prompt');
    const prompt   = promptEl.value.trim();
    if (!prompt && !selectedFile) return;

    const imgSrc = selectedFile ? URL.createObjectURL(selectedFile) : null;
    appendMessage('user', prompt, imgSrc);

    promptEl.value = '';
    promptEl.style.height = 'auto';
    const fileSnap = selectedFile;
    removeImage();

    isStreaming = true;
    
    document.getElementById('send-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'block';

    const { bubble, textNode } = appendMessage('bot', '', null);
    textNode.classList.add('cursor');

    // Setup Live Stats HTML Element block
    const statsDiv = document.createElement('div');
    statsDiv.className = 'stats-bar';
    
    const idPrefix = Date.now();
    
    // Structuring Input & Output rows cleanly
    statsDiv.innerHTML = `
      <div class="stats-item" style="color:#e2e2f0;font-weight:bold;">📦 ${MODEL_NAME}</div>
      <div class="status-badge" id="s-status-${idPrefix}">⏳ Reading...</div>
      
      <div class="stats-item" style="color: #a89f91;">
        📥 In: <span id="s-prompt-tokens-${idPrefix}">0</span>t
        <span class="stats-detail">(<span id="s-prompt-time-${idPrefix}">0.0</span>s @ <span id="s-prompt-speed-${idPrefix}">0.00</span> t/s)</span>
      </div>
      
      <div class="stats-item" style="color: #9cdcfe;">
        📤 Out: <span id="s-gen-tokens-${idPrefix}">0</span>t
        <span class="stats-detail">(<span id="s-gen-time-${idPrefix}">0.0</span>s @ <span id="s-gen-speed-${idPrefix}">0.00</span> t/s)</span>
      </div>
      
      <div class="stats-item">⏱ Total: <span id="s-time-${idPrefix}">0.0</span>s</div>
      <div class="stats-item" id="s-ctx-${idPrefix}">Ctx: ?/${CTX_SIZE}</div>
    `;
    bubble.appendChild(statsDiv);

    let requestStartTime = Date.now();
    let generationStartTime = null;

    // A lightweight interval purely to keep the "Total Time" smooth 
    // All actual metrics (Tokens, Speeds, Inputs) are passed direct from backend
    statsInterval = setInterval(() => {
      let totalTime = ((Date.now() - requestStartTime) / 1000).toFixed(1);
      const timeEl = document.getElementById(`s-time-${idPrefix}`);
      if(timeEl) timeEl.textContent = totalTime;
    }, 100);

    const form = new FormData();
    form.append('prompt', prompt);
    if (fileSnap) form.append('image', fileSnap);

    abortController = new AbortController();

    try {
      const resp = await fetch('/chat', { 
        method: 'POST', 
        body: form,
        signal: abortController.signal 
      });
      
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let botText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6).trim();
          
          if (dataStr === '[DONE]') {
             clearInterval(statsInterval);
             document.getElementById(`s-status-${idPrefix}`).innerHTML = '✓ Done';
             document.getElementById(`s-status-${idPrefix}`).style.color = 'var(--muted)';
             break;
          }
          if (!dataStr) continue;

          try {
            const payload = JSON.parse(dataStr);
            
            // 1. Live Context Evaluation payload from Backend!
            if (payload.input_progress) {
                let p = payload.input_progress;
                
                // Keep the fraction display 50/120 while reading so user sees exactly how far along it is
                document.getElementById(`s-prompt-tokens-${idPrefix}`).textContent = `${p.processed}/${p.total}`;
                document.getElementById(`s-prompt-time-${idPrefix}`).textContent = p.time.toFixed(1);
                document.getElementById(`s-prompt-speed-${idPrefix}`).textContent = p.speed.toFixed(2);
            }
            
            // 2. Exact Output Timings (Arrives EVERY token now because of timings_per_token=True)
            if (payload.stats) {
                let s = payload.stats;
                
                // If this is the very first generation token, switch the UI badge to "Generating"
                if (s.predicted_n > 0 && !generationStartTime) {
                    generationStartTime = Date.now();
                    document.getElementById(`s-status-${idPrefix}`).innerHTML = '⚡ Generating';
                    document.getElementById(`s-status-${idPrefix}`).style.color = 'var(--accent2)';
                }
                
                 // Update PROMPT stats (Include cached tokens so Input = prompt + cache)
                 if (s.prompt_n > 0 || s.cache_n > 0) {
                   const totalInput = (s.prompt_n || 0) + (s.cache_n || 0);
                   document.getElementById(`s-prompt-tokens-${idPrefix}`).textContent = totalInput;
                   document.getElementById(`s-prompt-time-${idPrefix}`).textContent = (s.prompt_time || 0).toFixed(1);
                   document.getElementById(`s-prompt-speed-${idPrefix}`).textContent = (s.prompt_speed || 0).toFixed(2);
                 }
                
                // Update GENERATION stats
                if (s.predicted_n > 0) {
                    document.getElementById(`s-gen-tokens-${idPrefix}`).textContent = s.predicted_n;
                    document.getElementById(`s-gen-time-${idPrefix}`).textContent = s.predicted_time.toFixed(1);
                    document.getElementById(`s-gen-speed-${idPrefix}`).textContent = s.predicted_speed.toFixed(2);
                }
                
                // Context Limit Display (now includes cached tokens)
                const cached = (s.cache_n || 0);
                let trueCtxTokens = (s.prompt_n || 0) + cached + (s.predicted_n || 0);
                let trueCtxPercent = ((trueCtxTokens / CTX_SIZE) * 100).toFixed(1);
                document.getElementById(`s-ctx-${idPrefix}`).textContent = `Ctx: ${trueCtxTokens}/${CTX_SIZE} (${trueCtxPercent}%)`;
            }

            // 3. Text output stream
            if (payload.text) {
              botText += payload.text;
              textNode.textContent = botText;
              scrollBottom();
            } 
            
            else if (payload.error) {
               textNode.textContent += `\\n⚠ Error: ${payload.error}`;
               clearInterval(statsInterval);
               document.getElementById(`s-status-${idPrefix}`).innerHTML = '⚠ Error';
               document.getElementById(`s-status-${idPrefix}`).style.color = 'var(--danger)';
            }
          } catch(e) {}
        }
      }
    } catch (err) {
      clearInterval(statsInterval);
      if (err.name === 'AbortError') {
        textNode.textContent += ` [Stopped]`;
        const st = document.getElementById(`s-status-${idPrefix}`);
        if(st) { st.innerHTML = '🛑 Stopped'; st.style.color = 'var(--danger)'; }
      } else {
        textNode.textContent += `\\n⚠ Request Failed: ${err.message}`;
      }
    } finally {
      clearInterval(statsInterval);
      textNode.classList.remove('cursor');
      isStreaming = false;
      document.getElementById('stop-btn').style.display = 'none';
      document.getElementById('send-btn').style.display = 'block';
      promptEl.focus();
    }
  }

  async function clearHistory() {
    if (isStreaming) return;
    await fetch('/history', { method: 'DELETE' });
    [...document.querySelectorAll('.msg')].forEach(el => el.remove());
    document.getElementById('empty').style.display = '';
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)