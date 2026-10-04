import time
import os
import wave
import requests

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel
from piper import PiperVoice


app = FastAPI()


# ============================================================
# Configuration
# ============================================================

WHISPER_MODEL = "distil-small.en"

# Piper
PIPER_MODEL = "en_US-lessac-medium.onnx"

# Audio files
INPUT_AUDIO = "audio/browser_input.webm"
OUTPUT_AUDIO = "audio/browser_response.wav"

# Local Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:0.5b"


# ============================================================
# Conversation History
# ============================================================

# Stored only in RAM.
#
# Server starts  -> empty conversation
# During runtime -> conversation is remembered
# Server stops   -> history disappears

history = []


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = (
    "You are a customer support voice assistant. "
    "Give short, direct and conversational answers. "
    "Keep responses under 80 words unless the user asks "
    "for a detailed explanation. "
    "Do not repeat information unnecessarily."
)


# ============================================================
# Load Whisper
# ============================================================

print("Loading Whisper...")

whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")


# ============================================================
# Load Piper
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
# Request Model
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# Home Page
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
# Ollama
# ============================================================

def ask_ollama(user_message):

    global history

    # Add user message
    history.append({
        "role": "user",
        "content": user_message
    })

    # Keep only recent messages
    recent_history = history[-4:]

    print("\nAsking Ollama...")

    response = requests.post(
        OLLAMA_URL,

        json={
            "model": OLLAMA_MODEL,

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                *recent_history
            ],

            "stream": False,

            "options": {
                "num_predict": 100
            }
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    ai_response = data["message"]["content"].strip()

    # Save AI response
    history.append({
        "role": "assistant",
        "content": ai_response
    })

    return ai_response


# ============================================================
# Text Chat
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    print("\n" + "=" * 50)

    print("USER:")
    print(request.message)

    ai_response = ask_ollama(
        request.message
    )

    print("\nAI:")
    print(ai_response)

    print("=" * 50)

    return {
        "response": ai_response
    }


# ============================================================
# Voice
# ============================================================

@app.post("/voice")
async def voice(file: UploadFile = File(...)):

    print("\n" + "=" * 50)
    print("VOICE INPUT RECEIVED")

    total_start = time.time()


    # --------------------------------------------------------
    # Save Browser Audio
    # --------------------------------------------------------

    audio_data = await file.read()
    print(f"Audio file size: {len(audio_data) / 1024:.2f} KB")

    os.makedirs(
        "audio",
        exist_ok=True
    )

    with open(
        INPUT_AUDIO,
        "wb"
    ) as audio_file:

        audio_file.write(audio_data)

    print("Audio received.")


    # --------------------------------------------------------
    # Speech → Text
    # --------------------------------------------------------

    whisper_start = time.time()

    print("\nTranscribing...")

    segments, info = whisper_model.transcribe(

        INPUT_AUDIO,

        vad_filter=True,

        vad_parameters={
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400
        }
    )

    user_text = ""

    for segment in segments:

        user_text += segment.text + " "

    user_text = user_text.strip()

    whisper_time = time.time() - whisper_start

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
    # Ollama
    # --------------------------------------------------------

    ollama_start = time.time()

    ai_response = ask_ollama(
        user_text
    )

    ollama_time = time.time() - ollama_start

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

    # Piper was already loaded when the server started.
    # We reuse the same loaded model here.

    with wave.open(
        OUTPUT_AUDIO,
        "wb"
    ) as wav_file:

        piper_voice.synthesize_wav(
            ai_response,
            wav_file
        )

    piper_time = time.time() - piper_start

    print("Voice generated.")

    print(
        f"Piper generation time: "
        f"{piper_time:.2f} seconds"
    )


    # --------------------------------------------------------
    # Total Processing Time
    # --------------------------------------------------------

    total_time = time.time() - total_start

    print("\n" + "-" * 50)

    print(
        f"Whisper : "
        f"{whisper_time:.2f}s"
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
# Return Generated Audio
# ============================================================

@app.get("/voice-response")
def voice_response():

    return FileResponse(
        OUTPUT_AUDIO,
        media_type="audio/wav"
    )