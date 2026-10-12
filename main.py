import time
import os
import wave

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel
from piper import PiperVoice

# IMPORTANT:
# RAG + Ollama are handled only by rag.py
from rag import HybridRetriever, ask_ollama


# ============================================================
# APP
# ============================================================

app = FastAPI()


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = "distil-small.en"

PIPER_MODEL = "en_US-lessac-medium.onnx"

INPUT_AUDIO = "audio/browser_input.webm"
OUTPUT_AUDIO = "audio/browser_response.wav"


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper...")

whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")


# ============================================================
# LOAD PIPER
# ============================================================

print("Loading Piper...")

piper_start = time.time()

piper_voice = PiperVoice.load(
    PIPER_MODEL
)

piper_load_time = time.time() - piper_start

print(
    f"Piper loaded in "
    f"{piper_load_time:.2f} seconds."
)


# ============================================================
# LOAD RAG
# ============================================================

print("\nLoading RAG system...")

rag_start = time.time()

rag_retriever = HybridRetriever()

rag_load_time = time.time() - rag_start

print(
    f"RAG system loaded in "
    f"{rag_load_time:.2f} seconds."
)

print("RAG ready.")


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    with open(
        "templates/index.html",
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# RAG
# ============================================================

def retrieve_context(user_message):

    print("\nSearching knowledge base...")

    results = rag_retriever.search(
        user_message
    )

    if not results:

        print(
            "No sufficiently relevant information found."
        )

        return None

    print("\nRetrieved Context:")

    context_parts = []

    for result in results:

        print(
            f"\nRerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            result["text"]
        )

        context_parts.append(
            result["text"]
        )

    context = "\n\n".join(
        context_parts
    )

    return context


# ============================================================
# TEXT CHAT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    print("\n" + "=" * 50)
    print("TEXT CHAT")
    print("USER:")
    print(request.message)

    # --------------------------------------------------------
    # RAG Retrieval
    # --------------------------------------------------------

    rag_start = time.time()

    context = retrieve_context(
        request.message
    )

    rag_time = time.time() - rag_start

    # --------------------------------------------------------
    # No Relevant Knowledge
    # --------------------------------------------------------

    if context is None:

        ai_response = (
            "I don't have that information."
        )

        print("\nAI:")
        print(ai_response)

        print(
            f"\nRAG time: "
            f"{rag_time:.2f} seconds"
        )

        print("=" * 50)

        return {
            "response": ai_response
        }

    # --------------------------------------------------------
    # Ollama
    #
    # ask_ollama comes directly from rag.py
    # --------------------------------------------------------

    ollama_start = time.time()

    ai_response = ask_ollama(
        request.message,
        context
    )

    ollama_time = time.time() - ollama_start

    print("\nAI:")
    print(ai_response)

    print(
        f"\nRAG time: "
        f"{rag_time:.2f} seconds"
    )

    print(
        f"Ollama time: "
        f"{ollama_time:.2f} seconds"
    )

    print("=" * 50)

    return {
        "response": ai_response
    }


# ============================================================
# VOICE
# ============================================================

@app.post("/voice")
async def voice(
    file: UploadFile = File(...)
):

    print("\n" + "=" * 50)
    print("VOICE INPUT RECEIVED")

    total_start = time.time()

    # --------------------------------------------------------
    # Save Browser Audio
    # --------------------------------------------------------

    audio_data = await file.read()

    print(
        f"Audio file size: "
        f"{len(audio_data) / 1024:.2f} KB"
    )

    os.makedirs(
        "audio",
        exist_ok=True
    )

    with open(
        INPUT_AUDIO,
        "wb"
    ) as audio_file:

        audio_file.write(
            audio_data
        )

    print("Audio received.")

    # --------------------------------------------------------
    # Speech → Text
    # --------------------------------------------------------

    whisper_start = time.time()

    print("\nTranscribing...")

    segments, info = whisper_model.transcribe(
        INPUT_AUDIO,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False
    )

    user_text = ""

    for segment in segments:

        user_text += (
            segment.text + " "
        )

    user_text = user_text.strip()

    whisper_time = (
        time.time() - whisper_start
    )

    print("\nUSER:")
    print(user_text)

    print(
        f"\nWhisper time: "
        f"{whisper_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # No Speech Detected
    # --------------------------------------------------------

    if not user_text:

        return {
            "user_text": "",
            "response": "I couldn't hear you.",
            "audio": None
        }

    # --------------------------------------------------------
    # RAG Retrieval
    # --------------------------------------------------------

    rag_start = time.time()

    context = retrieve_context(
        user_text
    )

    rag_time = (
        time.time() - rag_start
    )

    # --------------------------------------------------------
    # No Relevant Information
    # --------------------------------------------------------

    if context is None:

        ai_response = (
            "I don't have that information."
        )

        print("\nAI:")
        print(ai_response)

        # ----------------------------------------------------
        # Text → Speech
        # ----------------------------------------------------

        piper_start = time.time()

        print("\nGenerating voice...")

        with wave.open(
            OUTPUT_AUDIO,
            "wb"
        ) as wav_file:

            piper_voice.synthesize_wav(
                ai_response,
                wav_file
            )

        piper_time = (
            time.time() - piper_start
        )

        print("Voice generated.")

        print(
            f"Piper generation time: "
            f"{piper_time:.2f} seconds"
        )

        total_time = (
            time.time() - total_start
        )

        print("\n" + "-" * 50)

        print(
            f"Whisper : "
            f"{whisper_time:.2f}s"
        )

        print(
            f"RAG     : "
            f"{rag_time:.2f}s"
        )

        print(
            f"Piper   : "
            f"{piper_time:.2f}s"
        )

        print(
            f"TOTAL   : "
            f"{total_time:.2f}s"
        )

        print("-" * 50)
        print("=" * 50)

        return {
            "user_text": user_text,
            "response": ai_response,
            "audio": "/voice-response"
        }

    # --------------------------------------------------------
    # Ollama
    #
    # ask_ollama comes directly from rag.py
    # --------------------------------------------------------

    ollama_start = time.time()

    ai_response = ask_ollama(
        user_text,
        context
    )

    ollama_time = (
        time.time() - ollama_start
    )

    print("\nAI:")
    print(ai_response)

    print(
        f"\nOllama time: "
        f"{ollama_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Text → Speech
    # --------------------------------------------------------

    piper_start = time.time()

    print("\nGenerating voice...")

    with wave.open(
        OUTPUT_AUDIO,
        "wb"
    ) as wav_file:

        piper_voice.synthesize_wav(
            ai_response,
            wav_file
        )

    piper_time = (
        time.time() - piper_start
    )

    print("Voice generated.")

    print(
        f"Piper generation time: "
        f"{piper_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Total Processing Time
    # --------------------------------------------------------

    total_time = (
        time.time() - total_start
    )

    print("\n" + "-" * 50)

    print(
        f"Whisper : "
        f"{whisper_time:.2f}s"
    )

    print(
        f"RAG     : "
        f"{rag_time:.2f}s"
    )

    print(
        f"Ollama  : "
        f"{ollama_time:.2f}s"
    )

    print(
        f"Piper   : "
        f"{piper_time:.2f}s"
    )

    print(
        f"TOTAL   : "
        f"{total_time:.2f}s"
    )

    print("-" * 50)
    print("=" * 50)

    # --------------------------------------------------------
    # Return Result
    # --------------------------------------------------------

    return {
        "user_text": user_text,
        "response": ai_response,
        "audio": "/voice-response"
    }


# ============================================================
# RETURN GENERATED AUDIO
# ============================================================

@app.get("/voice-response")
def voice_response():

    return FileResponse(
        OUTPUT_AUDIO,
        media_type="audio/wav"
    )