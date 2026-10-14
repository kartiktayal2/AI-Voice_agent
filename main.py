import asyncio
import time
import wave
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
from piper import PiperVoice

from rag import HybridRetriever, ask_ollama


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

# Resolve paths relative to this file, so the app can be started
# from another working directory.
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
AUDIO_DIR = BASE_DIR / "audio"
INPUT_AUDIO = AUDIO_DIR / "browser_input.webm"
OUTPUT_AUDIO = AUDIO_DIR / "browser_response.wav"

WHISPER_MODEL = "distil-medium.en"
PIPER_MODEL = BASE_DIR / "en_US-lessac-medium.onnx"

app = FastAPI(title="Loan Policy Voice Assistant")

# The frontend uses shared audio filenames. Serialize voice requests so
# two requests cannot overwrite each other's input/output audio files.
voice_lock = asyncio.Lock()


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper...")
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)
print("Whisper loaded.")


# ============================================================
# LOAD PIPER
# ============================================================

print("Loading Piper...")
piper_start = time.time()

if not PIPER_MODEL.exists():
    raise FileNotFoundError(
        f"Piper voice model not found: {PIPER_MODEL}. "
        "Update PIPER_MODEL to the path of your existing .onnx voice model."
    )

piper_voice = PiperVoice.load(str(PIPER_MODEL))
piper_load_time = time.time() - piper_start
print(f"Piper loaded in {piper_load_time:.2f} seconds.")


# ============================================================
# LOAD RAG
# ============================================================

print("\nLoading RAG system...")
rag_start = time.time()
rag_retriever = HybridRetriever()
rag_load_time = time.time() - rag_start
print(f"RAG system loaded in {rag_load_time:.2f} seconds.")
print("RAG ready.")


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# HOME PAGE AND HEALTH CHECK
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("Frontend template not found.", status_code=404)

    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "rag_loaded": rag_retriever is not None}


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(user_message: str) -> str:
    """
    Retrieve relevant chunks and return them as a labelled string.
    Returning a string keeps this compatible with rag.py implementations
    that process context using string operations such as splitlines().
    An empty string means no sufficiently relevant chunks were found.
    """
    print("\nSearching knowledge base...")
    results = rag_retriever.search(user_message)

    if not results:
        print("No sufficiently relevant information found for any question part.")
        return ""

    print("\nRetrieved Context:")
    context_parts = []

    for result in results:
        question_part = result.get("matched_question_part", user_message)
        score = result.get("rerank_score", 0.0)
        source = result.get("source", "unknown")
        text = result.get("text", "")

        print(f"\nQuestion part: {question_part}")
        print(f"Rerank Score: {score:.4f}")
        print(f"Source: {source}")
        print(text)

        # Keep labels in the context so the answer model can distinguish
        # which retrieved passage relates to which part of a compound query.
        context_parts.append(
            f"Question part: {question_part}\n"
            f"Source: {source}\n"
            f"Retrieved information:\n{text}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# TEXT CHAT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        return {"response": "Please enter a question."}

    print("\n" + "=" * 50)
    print("TEXT CHAT")
    print("USER:", message)

    # RAG timing
    rag_start = time.time()
    context = retrieve_context(message)
    rag_time = time.time() - rag_start

    # Always call ask_ollama, even if context is empty, so rag.py can
    # provide a conversational, question-specific refusal.
    ollama_start = time.time()
    ai_response = ask_ollama(message, context)
    ollama_time = time.time() - ollama_start

    print("\nAI:")
    print(ai_response)
    print(f"\nRAG time: {rag_time:.2f} seconds")
    print(f"Ollama time: {ollama_time:.2f} seconds")
    print("=" * 50)

    return {"response": ai_response}


# ============================================================
# VOICE INPUT
# ============================================================

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    async with voice_lock:
        return await _process_voice(file)


async def _process_voice(file: UploadFile):
    print("\n" + "=" * 50)
    print("VOICE INPUT RECEIVED")
    total_start = time.time()

    # --------------------------------------------------------
    # Save browser audio
    # --------------------------------------------------------
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_data = await file.read()

    print(f"Audio file size: {len(audio_data) / 1024:.2f} KB")

    if not audio_data:
        print("Empty audio upload received.")
        print("=" * 50)
        return {
            "user_text": "",
            "response": "I couldn't hear you. Please try again.",
            "audio": None,
        }

    INPUT_AUDIO.write_bytes(audio_data)
    print("Audio received.")

    # --------------------------------------------------------
    # Speech to text (VAD disabled, as requested)
    # --------------------------------------------------------
    whisper_start = time.time()
    print("\nTranscribing...")

    segments, info = whisper_model.transcribe(
        str(INPUT_AUDIO),
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    user_text = " ".join(segment.text.strip() for segment in segments).strip()
    whisper_time = time.time() - whisper_start

    print("\nUSER:")
    print(user_text)
    print(f"\nWhisper time: {whisper_time:.2f} seconds")

    if not user_text:
        total_time = time.time() - total_start
        print("\nNo speech detected.")
        print(f"Whisper: {whisper_time:.2f}s")
        print(f"TOTAL: {total_time:.2f}s")
        print("=" * 50)
        return {
            "user_text": "",
            "response": "I couldn't hear you. Please try again.",
            "audio": None,
        }

    # --------------------------------------------------------
    # RAG retrieval
    # --------------------------------------------------------
    rag_start = time.time()
    context = retrieve_context(user_text)
    rag_time = time.time() - rag_start

    # --------------------------------------------------------
    # Ollama / answer generation and refusal handling
    # --------------------------------------------------------
    ollama_start = time.time()
    ai_response = ask_ollama(user_text, context)
    ollama_time = time.time() - ollama_start

    print("\nAI:")
    print(ai_response)
    print(f"\nRAG time: {rag_time:.2f} seconds")
    print(f"Ollama time: {ollama_time:.2f} seconds")

    # --------------------------------------------------------
    # Text to speech
    # --------------------------------------------------------
    piper_start = time.time()
    print("\nGenerating voice...")

    with wave.open(str(OUTPUT_AUDIO), "wb") as wav_file:
        piper_voice.synthesize_wav(ai_response, wav_file)

    piper_time = time.time() - piper_start
    print("Voice generated.")
    print(f"Piper generation time: {piper_time:.2f} seconds")

    # --------------------------------------------------------
    # Detailed total processing time
    # --------------------------------------------------------
    total_time = time.time() - total_start

    print("\n" + "-" * 50)
    print(f"Whisper : {whisper_time:.2f}s")
    print(f"RAG     : {rag_time:.2f}s")
    print(f"Ollama  : {ollama_time:.2f}s")
    print(f"Piper   : {piper_time:.2f}s")
    print(f"TOTAL   : {total_time:.2f}s")
    print("-" * 50)
    print("=" * 50)

    return {
        "user_text": user_text,
        "response": ai_response,
        "audio": "/voice-response",
    }


# ============================================================
# RETURN GENERATED AUDIO
# ============================================================

@app.get("/voice-response")
def voice_response():
    if not OUTPUT_AUDIO.exists():
        return HTMLResponse("No generated audio is available yet.", status_code=404)

    return FileResponse(
        str(OUTPUT_AUDIO),
        media_type="audio/wav",
        filename="response.wav",
    )
