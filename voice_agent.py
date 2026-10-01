import sounddevice as sd
import torch
import numpy as np
import wave
import subprocess
import winsound

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


# ============================================================
# 1. Load models
# ============================================================

print("Loading Whisper...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")

print("\nLoading Silero VAD...")

vad_model = load_silero_vad()

print("VAD loaded.")


# ============================================================
# 2. Record using microphone + VAD
# ============================================================

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_LIMIT = 1.5

print("\nListening...")
print("Speak now!")

audio_chunks = []

speaking = False
silence_time = 0


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE
) as stream:

    while True:

        audio_chunk, overflowed = stream.read(CHUNK_SIZE)

        audio_chunk = audio_chunk[:, 0]

        audio_tensor = torch.from_numpy(
            audio_chunk.copy()
        )

        speech_probability = vad_model(
            audio_tensor,
            SAMPLE_RATE
        ).item()


        # -----------------------------
        # Speech detected
        # -----------------------------

        if speech_probability > 0.5:

            if not speaking:
                print("\nSpeech detected!")

            speaking = True
            silence_time = 0

            audio_chunks.append(
                audio_chunk.copy()
            )


        # -----------------------------
        # Silence detected
        # -----------------------------

        elif speaking:

            audio_chunks.append(
                audio_chunk.copy()
            )

            silence_time += CHUNK_SIZE / SAMPLE_RATE

            if silence_time >= SILENCE_LIMIT:

                print("\nSpeech finished.")

                break


# ============================================================
# 3. Save microphone audio
# ============================================================

audio = np.concatenate(audio_chunks)

audio_file = "audio/mic_input.wav"

audio_int16 = (
    audio * 32767
).astype(np.int16)


with wave.open(audio_file, "wb") as wf:

    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(
        audio_int16.tobytes()
    )


print("\nAudio saved:")
print(audio_file)


# ============================================================
# 4. Transcribe with Whisper
# ============================================================

print("\nTranscribing...")

segments, info = whisper_model.transcribe(
    audio_file
)

user_text = ""

for segment in segments:

    user_text += segment.text


user_text = user_text.strip()


print("\nUser said:")
print(user_text)


# ============================================================
# 5. Send text to Ollama
# ============================================================

print("\nAsking Ollama...")

result = subprocess.run(
    [
        "ollama",
        "run",
        "llama3.2",
        user_text
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

ai_response = result.stdout.strip()


print("\nAI response:")
print(ai_response)


# ============================================================
# 6. Convert response to speech using Piper
# ============================================================

print("\nGenerating voice with Piper...")

piper_exe = r".venv\Scripts\piper.exe"

piper_model = (
    r"models\piper\en_US-lessac-medium.onnx"
)

output_audio = r"audio\response.wav"


subprocess.run(
    [
        piper_exe,
        "-m",
        piper_model,
        "-f",
        output_audio
    ],
    input=ai_response,
    text=True,
    encoding="utf-8",
    errors="replace",
    check=True
)


print("\nVoice generated:")
print(output_audio)


# ============================================================
# 7. Play response
# ============================================================

print("\nPlaying response...")

winsound.PlaySound(
    output_audio,
    winsound.SND_FILENAME
)

print("Done.")