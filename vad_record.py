import sounddevice as sd
import torch
import numpy as np
from silero_vad import load_silero_vad


# -----------------------------
# Settings
# -----------------------------

SAMPLE_RATE = 16000
CHUNK_SIZE = 512

SILENCE_LIMIT = 1.5


# -----------------------------
# Load VAD
# -----------------------------

print("Loading Silero VAD...")

model = load_silero_vad()

print("VAD loaded.")


# -----------------------------
# Start microphone
# -----------------------------

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

        speech_probability = model(
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

            audio_chunks.append(audio_chunk.copy())


        # -----------------------------
        # Silence detected
        # -----------------------------

        elif speaking:

            audio_chunks.append(audio_chunk.copy())

            silence_time += CHUNK_SIZE / SAMPLE_RATE

            if silence_time >= SILENCE_LIMIT:

                print("\nSpeech finished.")

                break


# -----------------------------
# Combine recorded audio
# -----------------------------

audio = np.concatenate(audio_chunks)

print("Recorded samples:", len(audio))
print("Recording duration:", len(audio) / SAMPLE_RATE, "seconds")


# -----------------------------
# Save recording
# -----------------------------

audio_int16 = (audio * 32767).astype(np.int16)

import wave

output_file = "audio/vad_record.wav"

with wave.open(output_file, "wb") as wf:

    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())


print("Saved:", output_file)