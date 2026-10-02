import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel

import requests
import os
import subprocess


app = FastAPI()


# ============================================================
# Configuration
# ============================================================

WHISPER_MODEL = "base"

PIPER_EXE = r".venv\Scripts\piper.exe"

PIPER_MODEL = r"models\piper\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"audio\browser_input.webm"

OUTPUT_AUDIO = r"audio\browser_response.wav"

OLLAMA_URL = "http://ollama.railway.internal:11434/api/chat"

OLLAMA_MODEL = "qwen2.5:0.5b"


# ============================================================
# Conversation History
# ============================================================
#
# IMPORTANT:
# History is stored ONLY in RAM.
#
# When the server starts:
#     history = []
#
# During the current run:
#     conversation is remembered
#
# When the server stops:
#     history disappears
#
# When the server starts again:
#     completely new conversation
#

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


## ============================================================
# Ollama
# ============================================================

def ask_ollama(user_message):

    global history

    # Add user message to current RAM conversation
    history.append({
        "role": "user",
        "content": user_message
    })

    # --------------------------------------------------------
    # Keep only the latest 4 messages
    # --------------------------------------------------------
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

            # Limit response generation
            "options": {
                "num_predict": 100
            }
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    ai_response = data["message"]["content"].strip()

    # Save AI response in current RAM conversation
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


@app.post("/voice")
async def voice(file: UploadFile = File(...)):

    print("\n" + "=" * 50)

    print("VOICE INPUT RECEIVED")

    total_start = time.time()


    # --------------------------------------------------------
    # Save browser audio
    # --------------------------------------------------------

    audio_data = await file.read()

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

    print(f"\nWhisper time: {whisper_time:.2f} seconds")


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

    print(f"\nOllama time: {ollama_time:.2f} seconds")


    # --------------------------------------------------------
    # Text → Speech
    # --------------------------------------------------------

    piper_start = time.time()

    print("\nGenerating voice...")

    subprocess.run(
        [
            PIPER_EXE,
            "-m",
            PIPER_MODEL,
            "-f",
            OUTPUT_AUDIO
        ],
        input=ai_response,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True
    )

    piper_time = time.time() - piper_start

    print("Voice generated.")

    print(f"Piper time: {piper_time:.2f} seconds")


    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_time = time.time() - total_start

    print("\n" + "-" * 50)

    print(f"Whisper : {whisper_time:.2f}s")
    print(f"Ollama  : {ollama_time:.2f}s")
    print(f"Piper   : {piper_time:.2f}s")
    print(f"TOTAL   : {total_time:.2f}s")

    print("-" * 50)

    print("=" * 50)


    return {
        "user_text": user_text,
        "response": ai_response,
        "audio": "/voice-response"
    }


# ============================================================
# Voice Response
# ============================================================

@app.get("/voice-response")
def voice_response():

    return FileResponse(
        OUTPUT_AUDIO,
        media_type="audio/wav"
    )
    print("\nAI:")
    print(ai_response)


    # --------------------------------------------------------
    # Text → Speech
    # --------------------------------------------------------

    print("\nGenerating voice...")


    subprocess.run(

        [
            PIPER_EXE,

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_AUDIO
        ],

        input=ai_response,

        text=True,

        encoding="utf-8",

        errors="replace",

        check=True
    )


    print("Voice generated.")

    print("=" * 50)


    # --------------------------------------------------------
    # Return everything to frontend
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