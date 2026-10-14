from pathlib import Path
import re
import time

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MODEL_CACHE_DIR = BASE_DIR / "models" / "huggingface"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"

RETRIEVAL_TOP_K = 12
FINAL_TOP_K = 5

# This is a relevance gate, not a probability or confidence score.
RERANK_THRESHOLD = 0.35
OLLAMA_TIMEOUT = 120
MAX_CONTEXT_CHARS = 12000


# ============================================================
# TEXT / CHUNKING
# ============================================================

def _tokenize(text):
    """Consistent lightweight tokenization for lexical retrieval."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?|₹|%", text.lower())


def _is_heading(line):
    line = line.strip()
    if not line:
        return False
    if re.match(r"^#{1,6}\s+", line):
        return True
    if re.match(r"^\*\*[^*]{2,100}\*\*$", line):
        inner = line[2:-2].strip()
        return "?" not in inner and not inner.lower().startswith(("q:", "a:"))
    if (
        len(line.split()) <= 7
        and not any(mark in line for mark in ("?", "!", ".", ";"))
        and not re.match(r"^(?:q|a)\s*:", line, flags=re.I)
        and not re.match(r"^[-*]\s+", line)
    ):
        return True
    return False


def chunk_text(text, max_words=120):
    """
    Split text into heading-aware chunks.
    Keep paragraphs and FAQ answers together where possible.
    """
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]

    sections = []
    current_heading = ""
    current_paragraphs = []

    def flush_section():
        nonlocal current_paragraphs
        if current_heading or current_paragraphs:
            sections.append((current_heading, current_paragraphs[:]))
        current_paragraphs = []

    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        first_line = lines[0].strip()

        if _is_heading(first_line):
            flush_section()
            current_heading = first_line
            remainder = "\n".join(lines[1:]).strip()
            if remainder:
                current_paragraphs.append(remainder)
        else:
            current_paragraphs.append(paragraph)

    flush_section()
    chunks = []

    for heading, section_paragraphs in sections:
        prefix = heading.strip()
        current_parts = []
        current_count = len(_tokenize(prefix)) if prefix else 0

        def emit():
            nonlocal current_parts, current_count
            if not current_parts:
                return
            body = "\n".join(current_parts).strip()
            chunk = f"{prefix}\n{body}".strip() if prefix else body
            if chunk:
                chunks.append(chunk)
            current_parts = []
            current_count = len(_tokenize(prefix)) if prefix else 0

        for paragraph in section_paragraphs:
            pieces = [paragraph]
            if len(_tokenize(paragraph)) > max_words:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
                pieces = []
                group = []
                group_size = 0

                for sentence in sentences:
                    sentence_size = len(_tokenize(sentence))
                    if group and group_size + sentence_size > max_words:
                        pieces.append(" ".join(group))
                        group, group_size = [], 0

                    if sentence_size > max_words:
                        if group:
                            pieces.append(" ".join(group))
                            group, group_size = [], 0
                        words = sentence.split()
                        pieces.extend(
                            " ".join(words[start:start + max_words])
                            for start in range(0, len(words), max_words)
                        )
                    else:
                        group.append(sentence)
                        group_size += sentence_size

                if group:
                    pieces.append(" ".join(group))

            for piece in pieces:
                piece_size = len(_tokenize(piece))
                if current_parts and current_count + piece_size > max_words:
                    emit()
                current_parts.append(piece.strip())
                current_count += piece_size
        emit()

    return [chunk for chunk in chunks if chunk.strip()]


def split_question(query):
    """
    Split obvious multi-question input into independently answerable parts.
    Avoid splitting ordinary lists such as 'identity and address proof'.
    """
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text:
        return []

    # Question marks are the clearest boundary.
    question_parts = [p.strip(" ,.;!") for p in re.split(r"\?\s*", text) if p.strip(" ,.;!?")]
    if len(question_parts) > 1:
        return question_parts

    # Split conjunctions only when a new question-like clause starts.
    pattern = re.compile(
        r"\s+(?:and|also)\s+(?=(?:what|where|when|who|which|why|how|"
        r"is|are|can|could|do|does|did|will|would|tell me|explain)\b)",
        flags=re.IGNORECASE,
    )
    parts = [p.strip(" ,.;!?") for p in pattern.split(text) if p.strip(" ,.;!?")]
    return parts or [text]


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_documents():
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    documents = []
    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": file_path.name, "text": text})

    if not documents:
        raise FileNotFoundError(
            f"No non-empty .txt knowledge files found in {KNOWLEDGE_DIR}"
        )
    return documents


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    def __init__(self):
        print("Loading RAG system...")
        self.chunks = []

        for document in load_documents():
            for chunk in chunk_text(document["text"]):
                self.chunks.append({
                    "source": document["source"],
                    "text": chunk,
                })

        if not self.chunks:
            raise ValueError("Knowledge files did not produce searchable chunks.")

        print(f"Total chunks: {len(self.chunks)}")
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(MODEL_CACHE_DIR),
        )

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        self.bm25 = BM25Okapi([_tokenize(text) for text in texts])

        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print("RAG system ready.")

    def retrieve_candidates(self, query, top_k=RETRIEVAL_TOP_K):
        count = len(self.chunks)
        if count == 0:
            return []

        k = min(top_k, count)
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        vector_scores, vector_indices = self.faiss_index.search(query_embedding, k)
        vector_scores = vector_scores[0]
        vector_indices = vector_indices[0]

        bm25_scores = np.asarray(
            self.bm25.get_scores(_tokenize(query)),
            dtype="float32",
        )
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
        max_bm25 = float(bm25_scores.max()) if bm25_scores.size else 0.0
        normalized_bm25 = (
            bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)
        )

        vector_score_by_index = {
            int(index): float(score)
            for score, index in zip(vector_scores, vector_indices)
            if int(index) >= 0
        }
        candidate_indices = set(vector_score_by_index)
        candidate_indices.update(int(index) for index in bm25_top_indices)

        candidates = []
        for index in candidate_indices:
            vector_score = vector_score_by_index.get(index, 0.0)
            bm25_score = float(normalized_bm25[index])
            candidates.append({
                "index": index,
                "source": self.chunks[index]["source"],
                "text": self.chunks[index]["text"],
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "hybrid_score": 0.6 * vector_score + 0.4 * bm25_score,
            })

        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        print(
            f"RETRIEVAL DEBUG | vector={len(vector_score_by_index)} "
            f"| BM25={len(bm25_top_indices)} | unique={len(candidates)}"
        )
        return candidates

    def rerank(self, query, candidates, final_top_k=FINAL_TOP_K,
               threshold=RERANK_THRESHOLD):
        if not candidates:
            return []

        pairs = [[query, item["text"]] for item in candidates]
        raw_scores = np.asarray(
            self.reranker.predict(pairs), dtype="float32"
        ).reshape(-1)

        # Sigmoid is only used to make the reranker score easier to inspect.
        safe_scores = np.clip(raw_scores, -60, 60)
        scores = 1.0 / (1.0 + np.exp(-safe_scores))

        results = []
        for candidate, score, raw_score in zip(candidates, scores, raw_scores):
            item = candidate.copy()
            item["rerank_score"] = float(score)
            item["rerank_raw_score"] = float(raw_score)
            results.append(item)

        results.sort(key=lambda item: item["rerank_score"], reverse=True)
        for item in results:
            print(
                f"RERANK DEBUG | score={item['rerank_score']:.4f} "
                f"| raw={item['rerank_raw_score']:.4f} "
                f"| text={item['text'][:100]!r}"
            )

        return [
            item for item in results
            if item["rerank_score"] >= threshold
        ][:final_top_k]

    def search(self, query, final_top_k=FINAL_TOP_K,
               threshold=RERANK_THRESHOLD):
        """Search each clear question part and retain evidence for each part."""
        parts = split_question(query)
        if not parts:
            return []

        combined = []
        seen = set()

        for part in parts:
            print(f"RAG SEARCH PART: {part}")
            candidates = self.retrieve_candidates(part)
            matches = self.rerank(
                part, candidates,
                final_top_k=final_top_k,
                threshold=threshold,
            )
            for result in matches:
                key = (result["source"], result["text"])
                if key in seen:
                    continue
                seen.add(key)
                result = result.copy()
                result["matched_question_part"] = part
                combined.append(result)

        # Preserve each matched part for the answer generator and debugging.
        return combined


# ============================================================
# GROUNDED ANSWER GENERATION
# ============================================================

def _context_to_text(context):
    """Accept the string format used by main.py or retrieved result dictionaries."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, (list, tuple)):
        blocks = []
        for item in context:
            if isinstance(item, dict):
                source = item.get("source", "unknown source")
                text = item.get("text", "")
                matched_part = item.get("matched_question_part")
                if text:
                    prefix = f"Source: {source}"
                    if matched_part:
                        prefix += f" | Retrieved for: {matched_part}"
                    blocks.append(f"{prefix}\n{text}")
            else:
                blocks.append(str(item))
        return "\n\n".join(blocks).strip()
    return str(context).strip()


def _meaningful_tokens(text):
    stop_words = {
        "what", "which", "how", "does", "did", "are", "is", "the", "a", "an",
        "for", "of", "to", "i", "can", "you", "me", "my", "and", "on", "in",
        "from", "please", "tell", "explain", "about", "would", "could", "should",
        "do", "it", "be", "this", "that", "with", "have", "has", "had",
    }
    return {
        token for token in _tokenize(text)
        if token not in stop_words and len(token) > 1
    }


def _context_is_relevant(question, context):
    """
    Conservative, general-purpose evidence check.
    This does not encode any loan facts or domain-specific field names.
    """
    question_tokens = _meaningful_tokens(question)
    context_tokens = _meaningful_tokens(context)
    if not question_tokens or not context_tokens:
        return False

    overlap = question_tokens.intersection(context_tokens)
    # One strong shared term is often enough for short questions; longer
    # questions need a little more overlap to reduce unrelated-context leakage.
    needed = 1 if len(question_tokens) <= 4 else 2
    return len(overlap) >= needed


def ask_ollama(query, context):
    """
    Generate answers from retrieved knowledge only.
    No policy values, field-specific answers, or domain names are hard-coded.
    """
    started = time.perf_counter()
    parts = split_question(query)
    context_text = _context_to_text(context)

    if not parts:
        return "Please ask a question."

    if not context_text:
        return (
            "I couldn't find supporting information for that in the available "
            "knowledge base. Please check the source documents or ask about a "
            "detail covered there."
        )

    answers = []
    for part in parts:
        # Avoid passing unrelated retrieved chunks to the small model.
        if not _context_is_relevant(part, context_text):
            answers.append(
                f"I couldn't find supporting information for '{part}' in the "
                "available knowledge base."
            )
            continue

        system_prompt = """
You are a careful customer-support assistant answering from a supplied knowledge base.

Rules:
- Use only information explicitly supported by the supplied context.
- Do not use general knowledge, assumptions, guesses, or facts from memory.
- Answer only the current question part.
- Check every clause in the question and answer every part that the context supports.
- Do not confuse a starting value with a maximum, a minimum with a maximum, or a possible outcome with a guarantee.
- If a requested detail is absent, say it is not specified in the available context.
- If a calculation is requested, calculate only when the context supplies every required input. Otherwise state which inputs are missing.
- Do not invent numbers, dates, locations, requirements, or policy details.
- Give a complete answer in concise, natural spoken English. Never stop mid-sentence.
- Return only the answer, with no explanation of these instructions.
"""

        user_prompt = f"""AVAILABLE KNOWLEDGE:
{context_text[:MAX_CONTEXT_CHARS]}

QUESTION PART:
{part}

Answer this question part using only the available knowledge above.
If the context does not support a requested fact, explicitly say that it is not available.
"""

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_predict": 220,
            },
        }

        try:
            response = requests.post(
                OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
            if answer:
                answers.append(answer)
            else:
                answers.append(
                    f"I couldn't generate a supported answer for '{part}'."
                )
        except requests.RequestException as exc:
            print(f"Ollama request failed for question part {part!r}: {exc}")
            answers.append(
                "I'm having trouble generating the answer right now. Please try again."
            )

    final_answer = " ".join(answer.strip() for answer in answers if answer.strip())
    print(f"ANSWER GENERATION TOTAL: {time.perf_counter() - started:.2f}s")
    return final_answer or (
        "I couldn't find supporting information in the available knowledge base."
    )
