import uvicorn
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Initialize FastAPI App
app = FastAPI(title="LFM2.5-VL Chat")

# 2. Load Model
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
print("Model ready! Server starting...")

# 3. State Management
messages =[]

# 4. Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the frontend HTML."""
    return HTML_CONTENT

@app.delete("/history")
async def clear_history():
    """Clears the chat history."""
    global messages
    messages =[]
    return {"status": "cleared"}

@app.post("/chat")
async def chat_endpoint(prompt: str = Form(""), image: UploadFile = File(None)):
    """Handles chat completions with optional images and streams the response."""
    global messages

    # Build the user's message content
    content =[]
    
    if image and image.filename:
        # Read and encode the uploaded image to Base64
        image_bytes = await image.read()
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = image.content_type or "image/jpeg"
        
        # Standard OpenAI/Llama.cpp vision format
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
        })
        
    if prompt:
        content.append({"type": "text", "text": prompt})

    # Append the incoming message to history
    # If it's just text, we can pass a string. If it includes an image, we pass the list.
    messages.append({
        "role": "user", 
        "content": content if len(content) > 1 or image else prompt
    })

    # Generator for Server-Sent Events (SSE) Streaming
    def stream_generator():
        global messages
        full_response = ""
        
        # Note: If passing images, llama-cpp-python usually requires a chat_handler configured 
        # with a multimodal projector (mmproj). If not configured, text chat will still work perfectly.
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=4000,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            text_chunk = delta.get("content", "")
            
            if text_chunk:
                full_response += text_chunk
                # Escape newlines so SSE doesn't prematurely split chunks
                safe_chunk = text_chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
        
        # Append assistant's final response to history
        messages.append({"role": "assistant", "content": full_response})
        yield "data:[DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# 5. The HTML UI (Embedded)
# Note: I slightly adjusted the JS line `botText += data.replace(/\\n/g, '\n');` 
# so the UI can properly render streaming newlines sent by the Python backend.
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
  /* ── Reset & Variables ─────────────────────────── */
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

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 14px;
    line-height: 1.65;
  }

  /* ── Layout ─────────────────────────────────────── */
  .app {
    display: grid;
    grid-template-rows: 56px 1fr auto;
    height: 100vh;
    max-width: 860px;
    margin: 0 auto;
  }

  /* ── Header ─────────────────────────────────────── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .logo {
    font-family: var(--font-disp);
    font-weight: 800;
    font-size: 16px;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .logo-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent2);
    box-shadow: 0 0 8px var(--accent2);
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .model-tag {
    font-size: 11px;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.04em;
  }

  .btn-clear {
    font-family: var(--font-mono);
    font-size: 11px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    transition: border-color 0.2s, color 0.2s;
  }
  .btn-clear:hover { border-color: var(--danger); color: var(--danger); }

  /* ── Chat Window ─────────────────────────────────── */
  #chat {
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    scroll-behavior: smooth;
  }

  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── Message Bubbles ─────────────────────────────── */
  .msg {
    display: flex;
    flex-direction: column;
    max-width: 88%;
    animation: fadeUp 0.25s ease both;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .msg.user  { align-self: flex-end; }
  .msg.bot   { align-self: flex-start; }

  .msg-label {
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
    padding: 0 4px;
  }

  .msg.user .msg-label { text-align: right; color: var(--accent); }
  .msg.bot  .msg-label { color: var(--accent2); }

  .bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.7;
    position: relative;
  }

  .msg.user .bubble {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-bottom-right-radius: 2px;
  }

  .msg.bot .bubble {
    background: var(--bot-bg);
    border: 1px solid #1a1a2e;
    border-bottom-left-radius: 2px;
  }

  /* Blinking cursor during stream */
  .cursor::after {
    content: '▋';
    color: var(--accent2);
    animation: blink 0.7s step-start infinite;
  }
  @keyframes blink {
    50% { opacity: 0; }
  }

  /* Image thumbnail inside message */
  .msg-image {
    max-width: 220px;
    max-height: 160px;
    border-radius: 6px;
    margin-bottom: 8px;
    object-fit: cover;
    border: 1px solid var(--border);
  }

  /* System / empty state */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--muted);
    pointer-events: none;
    user-select: none;
  }

  .empty-state h2 {
    font-family: var(--font-disp);
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .empty-state p { font-size: 12px; }

  /* ── Input Area ──────────────────────────────────── */
  .input-area {
    border-top: 1px solid var(--border);
    background: var(--bg);
    padding: 14px 20px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* Image preview strip */
  #preview-strip {
    display: none;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }

  #preview-strip.visible { display: flex; }

  #img-preview {
    width: 48px;
    height: 48px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }

  #preview-name {
    flex: 1;
    font-size: 11px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .btn-remove-img {
    background: transparent;
    border: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 2px 6px;
    border-radius: 4px;
    transition: color 0.2s;
  }
  .btn-remove-img:hover { color: var(--danger); }

  /* Prompt row */
  .prompt-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  .btn-attach {
    flex-shrink: 0;
    width: 40px; height: 40px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--muted);
    cursor: pointer;
    font-size: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.2s, color 0.2s;
  }
  .btn-attach:hover { border-color: var(--accent); color: var(--accent); }

  #prompt {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 10px 14px;
    resize: none;
    min-height: 40px;
    max-height: 140px;
    outline: none;
    transition: border-color 0.2s;
    overflow-y: auto;
  }
  #prompt:focus { border-color: var(--accent); }
  #prompt::placeholder { color: var(--muted); }

  .btn-send {
    flex-shrink: 0;
    height: 40px;
    padding: 0 18px;
    background: var(--accent);
    border: none;
    border-radius: var(--radius);
    color: #fff;
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    white-space: nowrap;
  }
  .btn-send:hover:not(:disabled) { opacity: 0.85; }
  .btn-send:active:not(:disabled) { transform: scale(0.97); }
  .btn-send:disabled { opacity: 0.35; cursor: not-allowed; }

  .hint {
    font-size: 10px;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.03em;
  }

  /* ── Responsive ──────────────────────────────────── */
  @media (max-width: 600px) {
    .model-tag { display: none; }
    .msg { max-width: 96%; }
  }
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header>
    <div class="logo">
      <span class="logo-dot"></span>
      LFM2.5-VL
    </div>
    <div class="header-right">
      <span class="model-tag">1.6B · CPU</span>
      <button class="btn-clear" onclick="clearHistory()">clear history</button>
    </div>
  </header>

  <!-- Chat -->
  <div id="chat">
    <div class="empty-state" id="empty">
      <h2>LFM2.5-VL Chat</h2>
      <p>Send a message or attach an image to begin</p>
    </div>
  </div>

  <!-- Input -->
  <div class="input-area">

    <!-- Image preview -->
    <div id="preview-strip">
      <img id="img-preview" src="" alt="preview" />
      <span id="preview-name"></span>
      <button class="btn-remove-img" onclick="removeImage()" title="Remove image">✕</button>
    </div>

    <!-- Prompt row -->
    <div class="prompt-row">
      <!-- Hidden file input -->
      <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFile(event)" />

      <button class="btn-attach" onclick="document.getElementById('file-input').click()" title="Attach image">
        📎
      </button>

      <textarea
        id="prompt"
        rows="1"
        placeholder="Ask anything…"
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"
      ></textarea>

      <button class="btn-send" id="send-btn" onclick="sendMessage()">Send ↑</button>
    </div>

    <div class="hint">Enter to send · Shift+Enter for new line · images optional</div>
  </div>

</div>

<script>
  // ── State ──────────────────────────────────────────
  let selectedFile = null;
  let isStreaming  = false;

  // ── File Handling ──────────────────────────────────
  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;
    const strip   = document.getElementById('preview-strip');
    const preview = document.getElementById('img-preview');
    const name    = document.getElementById('preview-name');
    preview.src   = URL.createObjectURL(file);
    name.textContent = file.name;
    strip.classList.add('visible');
  }

  function removeImage() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-strip').classList.remove('visible');
    document.getElementById('img-preview').src = '';
  }

  // ── UI Helpers ─────────────────────────────────────
  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function scrollBottom() {
    const chat = document.getElementById('chat');
    chat.scrollTop = chat.scrollHeight;
  }

  function appendMessage(role, text, imgSrc) {
    // Hide empty state
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
      const img    = document.createElement('img');
      img.src      = imgSrc;
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

  // ── Send ───────────────────────────────────────────
  async function sendMessage() {
    if (isStreaming) return;

    const promptEl = document.getElementById('prompt');
    const prompt   = promptEl.value.trim();
    if (!prompt && !selectedFile) return;

    // Show user message
    const imgSrc = selectedFile ? URL.createObjectURL(selectedFile) : null;
    appendMessage('user', prompt, imgSrc);

    // Clear inputs
    promptEl.value = '';
    promptEl.style.height = 'auto';
    const fileSnap = selectedFile;
    removeImage();

    // Lock UI
    isStreaming = true;
    document.getElementById('send-btn').disabled = true;

    // Show bot bubble with cursor
    const { bubble, textNode } = appendMessage('bot', '', null);
    bubble.classList.add('cursor');

    // Build form data
    const form = new FormData();
    form.append('prompt', prompt || '(describe the image)');
    if (fileSnap) form.append('image', fileSnap);

    try {
      const resp = await fetch('/chat', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = '';
      let botText   = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buffer.split('\\n');
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') break;
          // Unescape safe newlines provided by Python stream
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

  // ── Clear History ──────────────────────────────────
  async function clearHistory() {
    if (isStreaming) return;
    await fetch('/history', { method: 'DELETE' });
    const chat = document.getElementById('chat');
    // Remove all messages but keep empty state
    [...chat.querySelectorAll('.msg')].forEach(el => el.remove());
    document.getElementById('empty').style.display = '';
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)