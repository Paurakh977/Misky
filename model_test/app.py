import io, os, base64, tempfile, logging, warnings
from pathlib import Path

import psutil
from faster_whisper import WhisperModel  # ✅ removed `audio` import
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    import pynvml
    pynvml.nvmlInit()
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    logging.warning("NVML not available: %s", e)
    _gpu_handle = None

try:
    import torch
except Exception:
    torch = None

print("Loading faster-whisper medium (INT8, CPU) …")

# asr_model = WhisperModel(
#     "medium",
#     device="cpu",
#     compute_type="int8",
#     cpu_threads=0,
#     num_workers=1,
# )

asr_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=0,      # 0 = auto (all cores)
    num_workers=1,
)
print("✅ Model ready")

TRANSCRIPT_FILE = Path("transcriptions.txt")
app = FastAPI(title="WhisperASR — small/CPU")


class AudioPayload(BaseModel):
    audio: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("wisper.html").read_text(encoding="utf-8")


@app.get("/stats")
async def system_stats():
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu  = psutil.cpu_percent(interval=None)
    per_core = psutil.cpu_percent(percpu=True)
    return {
        "ram_total_mb":  vm.total      / 1024**2,
        "ram_used_mb":   vm.used       / 1024**2,
        "ram_avail_mb":  vm.available  / 1024**2,
        "ram_pct":       vm.percent,
        "swap_total_mb": swap.total    / 1024**2,
        "swap_used_mb":  swap.used     / 1024**2,
        "swap_pct":      swap.percent,
        "cpu_pct":       cpu,
        "cpu_per_core":  per_core,
        "cpu_count":     psutil.cpu_count(logical=True),
    }


@app.get("/gpu")
async def gpu_stats():
    if _gpu_handle is None:
        return JSONResponse({"error": "NVML not available"}, status_code=501)
    name = temp = util_pct = None
    mem_used = mem_total = mem_free = None
    try:
        raw = pynvml.nvmlDeviceGetName(_gpu_handle)
        name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception as e:
        logging.warning("nvmlDeviceGetName failed: %s", e)
    try:
        temp = pynvml.nvmlDeviceGetTemperature(_gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception as e:
        logging.warning("nvmlDeviceGetTemperature failed: %s", e)
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
        util_pct = getattr(util, "gpu", None)
    except Exception as e:
        logging.warning("nvmlDeviceGetUtilizationRates failed: %s", e)
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
        mem_used  = mem.used  / 1024**2
        mem_total = mem.total / 1024**2
        mem_free  = mem.free  / 1024**2
    except Exception as e:
        logging.warning("nvmlDeviceGetMemoryInfo failed: %s", e)
    torch_alloc = torch_reserved = 0.0
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch_alloc    = torch.cuda.memory_allocated() / 1024**2
                torch_reserved = torch.cuda.memory_reserved()  / 1024**2
        except Exception as e:
            logging.warning("torch.cuda memory query failed: %s", e)
    return {
        "name": name, "temp_c": temp, "gpu_util_pct": util_pct,
        "mem_used_mb": mem_used, "mem_total_mb": mem_total, "mem_free_mb": mem_free,
        "torch_alloc_mb": torch_alloc, "torch_reserved_mb": torch_reserved,
    }


@app.post("/transcribe")
async def transcribe(payload: AudioPayload):
    raw = base64.b64decode(payload.audio)

    # ✅ Write WebM directly — no ffmpeg conversion needed
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw)
        webm_path = f.name

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    try:
        # segments, info = asr_model.transcribe(
        #     webm_path,
        #     beam_size=3,                        # ✅ 3 = 30% faster, barely less accurate than 5                   # ✅ skip auto-detect, lock primary language
        #     condition_on_previous_text=False,   # ✅ faster + prevents hallucination loops
        #     temperature=0,                      # ✅ no fallback retries = faster
        #     vad_filter=True,
        #     vad_parameters=dict(min_silence_duration_ms=500),
        # )


        segments, info = asr_model.transcribe(
        webm_path,
        beam_size=5,                        # max search paths
        best_of=5,                          # sample 5 candidates, pick best
        # condition_on_previous_text=True,    # use context for better coherence
        # temperature=[0.0, 0.2, 0.4, 0.6],  # retry with higher temp if uncertain
        # compression_ratio_threshold=2.4,    # aggressive hallucination filter
        # log_prob_threshold=-0.5,            # reject low-confidence segments
        # no_speech_threshold=0.6,            # better silence detection
        # vad_filter=True,
        # vad_parameters=dict(min_silence_duration_ms=300),
        # word_timestamps=True,               # forces more careful alignment
    )
        print(f"Detected: {info.language} ({info.language_probability:.0%})")
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        logging.disable(logging.NOTSET)
        try:
            os.remove(webm_path)
        except OSError:
            pass

    if text:
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    return {"text": text or "(no speech detected)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)