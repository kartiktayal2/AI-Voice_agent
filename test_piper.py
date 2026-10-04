import time
import wave
from piper import PiperVoice

MODEL = "en_US-lessac-medium.onnx"
OUTPUT = "audio/piper_speed_test.wav"

print("Loading Piper...")
start = time.time()

voice = PiperVoice.load(MODEL)

print(f"Piper loaded in {time.time() - start:.2f} seconds")

text = "Hello, this is a test of the Piper text to speech system."

start = time.time()

with wave.open(OUTPUT, "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)

print(f"Speech generated in {time.time() - start:.2f} seconds")
print(f"Total time: {time.time() - start:.2f} seconds")
