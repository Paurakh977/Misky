"""
LFM2.5-VL FastAPI Chat — powered by llama-server subprocess
============================================================
WORKS WITH IMAGES because it uses llama-server (not llama-cpp-python),
which correctly loads the multimodal projector (mmproj) automatically
via the -hf flag.

Requirements:
    pip install fastapi uvicorn openai

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
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

# Path to your llama-server.exe (same folder as llama-cli.exe)
LLAMA_DIR = Path(r"C:\Users\V16\Downloads\llama-b8533-bin-win-cpu-x64")
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"

# The VL model — -hf flag auto-downloads model + mmproj from HuggingFace
MODEL_HF = "LiquidAI/LFM2.5-VL-1.6B-GGUF:Q4_0"

# llama-server runs on this port (your FastAPI runs on 8000)
LLAMA_PORT = 8181

# How long to wait for llama-server to finish loading the model (seconds)
# The first run downloads the model (~1 GB) so give it plenty of time
MODEL_LOAD_TIMEOUT = 180

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="LFM2.5-VL Chat")

# OpenAI client pointing at the local llama-server
# timeout=600 — image inference on CPU can take several minutes
oai_client = OpenAI(
    base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1",
    api_key="not-needed",
    timeout=600.0,
)

# In-memory conversation history
messages: list[dict] = []

# The llama-server subprocess handle
_server_proc: subprocess.Popen | None = None


# ── llama-server lifecycle ────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    """Return True if something is listening on 127.0.0.1:<port>."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_llama_server() -> None:
    global _server_proc

    if not LLAMA_SERVER.exists():
        print(
            f"\n[ERROR] llama-server.exe not found at:\n  {LLAMA_SERVER}\n"
            "Make sure the path in LLAMA_DIR points to your llama.cpp folder.\n"
        )
        sys.exit(1)

    cmd = [
        str(LLAMA_SERVER),
        "-hf",  MODEL_HF,          # auto-downloads model + mmproj
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "-c", "4096",               # context length
        "-n", "512",                # max tokens per reply
        "-t", "4",                  # CPU threads (raise if you have more cores)
        "--image-max-tokens", "64", # CRITICAL: limits image tokens for CPU speed
        "--temp", "0.1",
        "--min-p", "0.15",
        "--repeat-penalty", "1.05",
        "--log-disable",            # quieter output; remove to see full logs
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

    # Pipe server logs to our console in a background thread
    def _pipe_logs():
        for line in _server_proc.stdout:
            print(f"[llama-server] {line.rstrip()}")

    threading.Thread(target=_pipe_logs, daemon=True).start()

    # Wait until the port is open (model finished loading)
    print(
        f"[startup] Waiting for llama-server on port {LLAMA_PORT} "
        f"(timeout {MODEL_LOAD_TIMEOUT}s)…"
    )
    deadline = time.time() + MODEL_LOAD_TIMEOUT
    while time.time() < deadline:
        if _port_open(LLAMA_PORT):
            print("[startup] ✓ llama-server is ready!\n")
            return
        time.sleep(2)

    print(
        f"[ERROR] llama-server did not become ready within {MODEL_LOAD_TIMEOUT}s.\n"
        "Check the logs above for errors (download still in progress?).\n"
    )
    sys.exit(1)


@app.on_event("startup")
async def _app_startup() -> None:
    # Run blocking server startup in a thread so FastAPI's event loop isn't blocked
    t = threading.Thread(target=_start_llama_server, daemon=True)
    t.start()
    t.join()  # Block here until llama-server is ready


@app.on_event("shutdown")
async def _app_shutdown() -> None:
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        print("[shutdown] llama-server terminated.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT


@app.delete("/history")
async def clear_history():
    global messages
    messages = []
    return {"status": "cleared"}


@app.post("/chat")
async def chat_endpoint(
    prompt: str = Form(""),
    image: UploadFile = File(None),
):
    global messages

    # Build user message content
    content: list[dict] = []

    if image and image.filename:
        image_bytes = await image.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = image.content_type or "image/jpeg"
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )

    text = prompt.strip() or "(describe the image)"
    content.append({"type": "text", "text": text})

    # For text-only messages pass a plain string (keeps history compact)
    messages.append(
        {
            "role": "user",
            "content": content if (image and image.filename) else text,
        }
    )

    def stream_generator():
        global messages
        full_response = ""

        try:
            stream = oai_client.chat.completions.create(
                model="lfm2.5-vl-1.6b",
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                extra_body={"min_p": 0.15, "repetition_penalty": 1.05},
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    safe = delta.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

        except Exception as exc:
            yield f"data: ⚠ Error: {exc}\n\n"

        messages.append({"role": "assistant", "content": full_response})
        yield "data:[DONE]\n\n"

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

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
  }

  .header-right { display: flex; align-items: center; gap: 12px; }

  .model-tag { font-size: 11px; color: var(--muted); background: var(--surface);
    border: 1px solid var(--border); padding: 3px 8px; border-radius: 4px;
    letter-spacing: 0.04em; }

  .btn-clear { font-family: var(--font-mono); font-size: 11px;
    background: transparent; border: 1px solid var(--border); color: var(--muted);
    padding: 4px 10px; border-radius: 4px; cursor: pointer;
    transition: border-color .2s, color .2s; }
  .btn-clear:hover { border-color: var(--danger); color: var(--danger); }

  #chat { overflow-y: auto; padding: 24px 20px; display: flex;
    flex-direction: column; gap: 4px; scroll-behavior: smooth; }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg { display: flex; flex-direction: column; max-width: 88%;
    animation: fadeUp .25s ease both; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg.user { align-self: flex-end; }
  .msg.bot  { align-self: flex-start; }

  .msg-label { font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 4px; padding: 0 4px; }
  .msg.user .msg-label { text-align: right; color: var(--accent); }
  .msg.bot  .msg-label { color: var(--accent2); }

  .bubble { padding: 12px 16px; border-radius: var(--radius);
    white-space: pre-wrap; word-break: break-word; line-height: 1.7; }
  .msg.user .bubble { background: var(--user-bg); border: 1px solid var(--border);
    border-bottom-right-radius: 2px; }
  .msg.bot  .bubble { background: var(--bot-bg); border: 1px solid #1a1a2e;
    border-bottom-left-radius: 2px; }

  .cursor::after { content: '▋'; color: var(--accent2);
    animation: blink .7s step-start infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .msg-image { max-width: 220px; max-height: 160px; border-radius: 6px;
    margin-bottom: 8px; object-fit: cover; border: 1px solid var(--border);
    display: block; }

  .empty-state { flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px; color: var(--muted);
    pointer-events: none; user-select: none; }
  .empty-state h2 { font-family: var(--font-disp); font-size: 22px;
    font-weight: 800; background: linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .empty-state p { font-size: 12px; }

  .input-area { border-top: 1px solid var(--border); background: var(--bg);
    padding: 14px 20px 20px; display: flex; flex-direction: column; gap: 10px; }

  #preview-strip { display: none; align-items: center; gap: 10px; padding: 8px 10px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); }
  #preview-strip.visible { display: flex; }
  #img-preview { width: 48px; height: 48px; object-fit: cover;
    border-radius: 6px; border: 1px solid var(--border); }
  #preview-name { flex: 1; font-size: 11px; color: var(--muted); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }

  .btn-remove-img { background: transparent; border: none; color: var(--muted);
    cursor: pointer; font-size: 16px; line-height: 1; padding: 2px 6px;
    border-radius: 4px; transition: color .2s; }
  .btn-remove-img:hover { color: var(--danger); }

  .prompt-row { display: flex; gap: 8px; align-items: flex-end; }

  .btn-attach { flex-shrink: 0; width: 40px; height: 40px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--muted); cursor: pointer;
    font-size: 18px; display: flex; align-items: center; justify-content: center;
    transition: border-color .2s, color .2s; }
  .btn-attach:hover { border-color: var(--accent); color: var(--accent); }

  #prompt { flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--text);
    font-family: var(--font-mono); font-size: 13px; padding: 10px 14px;
    resize: none; min-height: 40px; max-height: 140px; outline: none;
    transition: border-color .2s; overflow-y: auto; }
  #prompt:focus { border-color: var(--accent); }
  #prompt::placeholder { color: var(--muted); }

  .btn-send { flex-shrink: 0; height: 40px; padding: 0 18px;
    background: var(--accent); border: none; border-radius: var(--radius);
    color: #fff; font-family: var(--font-mono); font-size: 13px; font-weight: 500;
    cursor: pointer; transition: opacity .2s, transform .1s; white-space: nowrap; }
  .btn-send:hover:not(:disabled) { opacity: .85; }
  .btn-send:active:not(:disabled) { transform: scale(.97); }
  .btn-send:disabled { opacity: .35; cursor: not-allowed; }

  .hint { font-size: 10px; color: var(--muted); text-align: center;
    letter-spacing: .03em; }

  @media (max-width: 600px) {
    .model-tag { display: none; }
    .msg { max-width: 96%; }
  }
</style>
</head>
<body>
<div class="app">

  <header>
    <div class="logo">
      <span class="logo-dot"></span>
      LFM2.5-VL
    </div>
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
      <button class="btn-remove-img" onclick="removeImage()" title="Remove image">✕</button>
    </div>

    <div class="prompt-row">
      <input type="file" id="file-input" accept="image/*"
             style="display:none" onchange="handleFile(event)" />

      <button class="btn-attach"
              onclick="document.getElementById('file-input').click()"
              title="Attach image">📎</button>

      <textarea id="prompt" rows="1" placeholder="Ask anything…"
                onkeydown="handleKey(event)"
                oninput="autoResize(this)"></textarea>

      <button class="btn-send" id="send-btn" onclick="sendMessage()">Send ↑</button>
    </div>

    <div class="hint">Enter to send · Shift+Enter for new line · images optional</div>
  </div>

</div>

<script>
  let selectedFile = null;
  let isStreaming  = false;

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
    document.getElementById('send-btn').disabled = true;

    const { bubble, textNode } = appendMessage('bot', '', null);
    bubble.classList.add('cursor');

    const form = new FormData();
    form.append('prompt', prompt);
    if (fileSnap) form.append('image', fileSnap);

    try {
      const resp = await fetch('/chat', { method: 'POST', body: form });
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
          const data = line.slice(6);
          if (data === '[DONE]') break;
          botText += data.replace(/\\\\n/g, '\\n');
          textNode.textContent = botText;
          scrollBottom();
        }
      }
    } catch (err) {
      textNode.textContent = `⚠ Error: ${err.message}`;
    } finally {
      bubble.classList.remove('cursor');
      isStreaming = false;
      document.getElementById('send-btn').disabled = false;
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

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)