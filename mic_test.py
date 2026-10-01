import sounddevice as sd
import wave

OUTPUT_FILE = "audio/mic_test.wav"

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5

print("Recording for 5 seconds...")
print("Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16"
)

sd.wait()

print("Recording finished.")

with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())

print(f"Saved: {OUTPUT_FILE}")