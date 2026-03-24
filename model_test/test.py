from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Download model
print("Loading model...")
model_path = hf_hub_download(
    repo_id="LiquidAI/LFM2.5-VL-1.6B-GGUF",
    filename="LFM2.5-VL-1.6B-Q4_0.gguf"
)
# repo_id="LiquidAI/LFM2.5-VL-1.6B-GGUF",
# 	filename="LFM2.5-VL-1.6B-BF16.gguf",
llm = Llama(
    model_path=model_path,
    n_gpu_layers=0,
    n_threads=4,
    n_ctx=2048,
    verbose=False
)

print("Model ready! Type 'quit' to exit.\n")



messages = []

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    print("Assistant: ", end="", flush=True)

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
            full_response += content

    print()  # newline after response
    messages.append({"role": "assistant", "content": full_response})