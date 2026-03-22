import io, os, base64, tempfile, contextlib, logging, warnings
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import pynvml

pynvml.nvmlInit()
_gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


print("Loading Parakeet-TDT-0.6B-v2 …")
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
asr_model = asr_model.to(torch.float16)   # ~1.2 GB instead of 2.4 GB
asr_model = asr_model.cuda()
asr_model.eval()
print("✅ Model loaded on CUDA")

TRANSCRIPT_FILE = Path("transcriptions.txt")
app = FastAPI(title="Parakeet ASR")

class AudioPayload(BaseModel):
    audio: str

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/gpu")
async def gpu_stats():
    mem   = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
    util  = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
    temp  = pynvml.nvmlDeviceGetTemperature(_gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    name  = pynvml.nvmlDeviceGetName(_gpu_handle)
    torch_alloc   = torch.cuda.memory_allocated() / 1024**2    # MB
    torch_reserved= torch.cuda.memory_reserved()  / 1024**2    # MB
    return {
        "name":           name,
        "temp_c":         temp,
        "gpu_util_pct":   util.gpu,
        "mem_used_mb":    mem.used   / 1024**2,
        "mem_total_mb":   mem.total  / 1024**2,
        "mem_free_mb":    mem.free   / 1024**2,
        "torch_alloc_mb": torch_alloc,
        "torch_reserved_mb": torch_reserved,
    }

    
@app.post("/transcribe")
async def transcribe(payload: AudioPayload):
    raw = base64.b64decode(payload.audio)

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw)
        webm_path = f.name
    wav_path = webm_path.replace(".webm", ".wav")

    ret = os.system(f'ffmpeg -y -i "{webm_path}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet')
    if ret != 0:
        return JSONResponse({"error": "ffmpeg failed — run: winget install ffmpeg"}, status_code=500)

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    try:
        with torch.inference_mode(), \
             io.StringIO() as buf, \
             contextlib.redirect_stdout(buf), \
             contextlib.redirect_stderr(buf):
            results = asr_model.transcribe([wav_path], verbose=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        logging.disable(logging.NOTSET)
        for p in (webm_path, wav_path):
            try: os.remove(p)
            except: pass

    result = results[0]
    text = result.text if hasattr(result, "text") else str(result)
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")

    return {"text": text}