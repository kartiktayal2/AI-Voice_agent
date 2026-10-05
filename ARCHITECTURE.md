# AI Voice Agent - Architecture

## Current Prototype Flow

Microphone
↓
Voice Activity Detection (Silero VAD)
↓
Speech-to-Text (Faster-Whisper)
↓
LLM Processing (Ollama)
↓
Text-to-Speech (Piper)
↓
Speaker

## Components

- **Silero VAD:** Detects when the user is speaking.
- **Faster-Whisper:** Converts speech into text.
- **Ollama:** Runs the local LLM.
- **Piper:** Converts generated text into speech.
- **FastAPI:** Provides the backend/API layer.

## Planned Architecture

Microphone / Telephony
↓
Voice Activity Detection
↓
Speech-to-Text
↓
Domain / Intent Detection
↓
RAG
↓
Local LLM
↓
Text-to-Speech
↓
Audio Response

## Future Components

- RAG with hybrid search
- ChromaDB vector database
- BM25 keyword search
- Twilio integration
- Docker deployment