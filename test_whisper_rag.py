import time

print("Loading RAG...")
from rag import HybridRetriever

retriever = HybridRetriever()
print("RAG loaded.")

print("Loading Whisper...")
from faster_whisper import WhisperModel

whisper_model = WhisperModel(
    "distil-small.en",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")

start = time.time()

segments, info = whisper_model.transcribe(
    "audio/browser_input.webm",
    language="en",
    beam_size=1,
    condition_on_previous_text=False,
    vad_filter=False
)

text = "".join(segment.text for segment in segments)

print("\nTEXT:", text)
print("TIME:", round(time.time() - start, 2), "seconds")
