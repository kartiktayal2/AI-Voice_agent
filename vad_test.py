import sounddevice as sd
import torch
from scipy.io.wavfile import write
from silero_vad import load_silero_vad, get_speech_timestamps


print("Loading Silero VAD...")

model = load_silero_vad()

print("VAD loaded.")


# -----------------------------
# Record audio
# -----------------------------

sample_rate = 16000
duration = 5

print("\nRecording for 5 seconds...")
print("Speak now!")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording finished.")


# -----------------------------
# Save recording
# -----------------------------

audio_file = "audio/vad_test.wav"

write(
    audio_file,
    sample_rate,
    audio
)

print("Saved:", audio_file)


# -----------------------------
# Convert audio to PyTorch tensor
# -----------------------------

wav = torch.from_numpy(audio).squeeze()


# -----------------------------
# Detect speech
# -----------------------------

speech_timestamps = get_speech_timestamps(
    wav,
    model,
    sampling_rate=sample_rate
)


print("\nSpeech timestamps:")
print(speech_timestamps)