from pathlib import Path
import re

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

KNOWLEDGE_DIR = Path("knowledge")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:0.5b"

# Retrieve broadly first; let the cross-encoder rank the union.
RETRIEVAL_TOP_K = 12
FINAL_TOP_K = 5

# This is a sigmoid-transformed cross-encoder score, NOT a calibrated
# probability. A lower gate avoids rejecting borderline relevant chunks.
RERANK_THRESHOLD = 0.35


# ============================================================
# TEXT / CHUNKING HELPERS
# ============================================================

def _tokenize(text):
    """Tokenize words and numbers consistently for BM25."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|₹|%", text.lower())


def _is_heading(line):
    """Recognize actual section headings without treating every short line as one."""
    line = line.strip()
    if not line:
        return False

    if re.match(r"^#{1,6}\s+", line):
        return True
    if re.match(r"^\*\*[^*]{2,100}\*\*$", line):
        inner = line[2:-2].strip()
        if not inner.lower().startswith(("q:", "a:")) and "?" not in inner:
            return True

    numbered_heading = re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+).{2,100}$", line)
    if numbered_heading and not line.rstrip().endswith((".", "?", "!")):
        return True

    # Plain-text headings such as "Personal Loan" or "Loan Repayment".
    # Do not treat questions, answers, or sentence-like lines as headings.
    if (
        len(line.split()) <= 7
        and not re.match(r"^(?:q|a)\s*:", line, flags=re.I)
        and not any(mark in line for mark in ("?", "!", ".", ";"))
        and not re.match(r"^[\-*]\s+", line)
    ):
        return True

    return False


def chunk_text(text, max_words=120):
    """
    Build heading-aware chunks while keeping paragraphs and FAQ answers together.
    Headings are repeated in each chunk from their section for retrieval context.
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
        nonlocal current_heading, current_paragraphs
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
            # If a heading wasn't detected, preserve the paragraph as content.
            current_paragraphs.append(paragraph)

    flush_section()

    chunks = []

    for heading, section_paragraphs in sections:
        prefix = heading.strip()
        content_parts = []
        current_count = len(_tokenize(prefix)) if prefix else 0

        def emit():
            nonlocal content_parts, current_count
            if not content_parts:
                return
            body = "\n".join(content_parts).strip()
            chunk = f"{prefix}\n{body}".strip() if prefix else body
            if chunk:
                chunks.append(chunk)
            content_parts = []
            current_count = len(_tokenize(prefix)) if prefix else 0

        for paragraph in section_paragraphs:
            paragraph_words = _tokenize(paragraph)

            # Split only an unusually long paragraph, preferably at sentence boundaries.
            pieces = [paragraph]
            if len(paragraph_words) > max_words:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
                pieces = []
                sentence_group = []
                sentence_count = 0

                for sentence in sentences:
                    sentence_size = len(_tokenize(sentence))
                    if sentence_group and sentence_count + sentence_size > max_words:
                        pieces.append(" ".join(sentence_group))
                        sentence_group = []
                        sentence_count = 0
                    if sentence_size > max_words:
                        if sentence_group:
                            pieces.append(" ".join(sentence_group))
                            sentence_group = []
                            sentence_count = 0
                        words = sentence.split()
                        for start in range(0, len(words), max_words):
                            pieces.append(" ".join(words[start:start + max_words]))
                    else:
                        sentence_group.append(sentence)
                        sentence_count += sentence_size

                if sentence_group:
                    pieces.append(" ".join(sentence_group))

            for piece in pieces:
                piece_count = len(_tokenize(piece))
                if content_parts and current_count + piece_count > max_words:
                    emit()
                content_parts.append(piece.strip())
                current_count += piece_count

        emit()

    return [chunk for chunk in chunks if chunk.strip()]


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    documents = []
    print("Loading documents...")

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR.resolve()}"
        )

    for file_path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": file_path.name, "text": text})

    if not documents:
        raise FileNotFoundError(
            f"No non-empty .txt knowledge files found in {KNOWLEDGE_DIR.resolve()}"
        )

    return documents


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    def __init__(self):
        print("Loading RAG system...")

        documents = load_documents()
        self.chunks = []

        for document in documents:
            for chunk in chunk_text(document["text"]):
                self.chunks.append({
                    "source": document["source"],
                    "text": chunk,
                })

        if not self.chunks:
            raise ValueError("The knowledge files did not produce any searchable chunks.")

        print(f"Total chunks: {len(self.chunks)}")

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        print("FAISS index created.")

        self.bm25 = BM25Okapi([_tokenize(text) for text in texts])
        print("BM25 index created.")

        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print("Reranker loaded.")
        print("RAG system ready.")

    def retrieve_candidates(self, query, top_k=RETRIEVAL_TOP_K):
        """Return the union of vector and lexical candidates; don't truncate the union."""
        count = len(self.chunks)
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

        bm25_scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype="float32")
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]

        max_bm25 = float(bm25_scores.max()) if bm25_scores.size else 0.0
        normalized_bm25 = bm25_scores / max_bm25 if max_bm25 > 0 else np.zeros_like(bm25_scores)

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
            hybrid_score = 0.6 * vector_score + 0.4 * bm25_score

            candidates.append({
                "index": index,
                "source": self.chunks[index]["source"],
                "text": self.chunks[index]["text"],
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score,
            })

        candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
        print(
            f"RETRIEVAL DEBUG | vector candidates={len(vector_score_by_index)} "
            f"| BM25 candidates={len(bm25_top_indices)} "
            f"| unique candidates={len(candidates)}"
        )
        return candidates

    def rerank(self, query, candidates, final_top_k=FINAL_TOP_K, threshold=RERANK_THRESHOLD):
        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        raw_scores = np.asarray(self.reranker.predict(pairs), dtype="float32").reshape(-1)

        # CrossEncoder logits are not calibrated probabilities. Sigmoid is retained
        # only as a monotonic score transform for compatibility with existing logs.
        safe_scores = np.clip(raw_scores, -60, 60)
        scores = 1.0 / (1.0 + np.exp(-safe_scores))

        results = []
        for candidate, score, raw_score in zip(candidates, scores, raw_scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            result["rerank_raw_score"] = float(raw_score)
            results.append(result)

        results.sort(key=lambda item: item["rerank_score"], reverse=True)

        for result in results:
            print(
                f"RERANK DEBUG | score={result['rerank_score']:.4f} "
                f"| raw={result['rerank_raw_score']:.4f} "
                f"| threshold={threshold:.2f} "
                f"| text={result['text'][:120]!r}"
            )

        relevant = [item for item in results if item["rerank_score"] >= threshold]

        # If no chunk passes the gate, refuse rather than send weak context to the LLM.
        return relevant[:final_top_k]

    def search(self, query, final_top_k=FINAL_TOP_K, threshold=RERANK_THRESHOLD):
        candidates = self.retrieve_candidates(query)
        return self.rerank(
            query,
            candidates,
            final_top_k=final_top_k,
            threshold=threshold,
        )


# ============================================================
# OLLAMA ANSWER GENERATION
# ============================================================

def ask_ollama(query, context):
    system_prompt = """
You are a careful bank customer-support assistant.

Use ONLY the supplied knowledge context to answer the user's question.

Rules:
1. Answer every distinct part of a multi-part question.
2. Use only facts explicitly supported by the context.
3. Never guess, infer a policy, or fill gaps with general knowledge.
4. Distinguish minimum, maximum, starting, and guaranteed values.
5. If the context says something is not specified, clearly say it is not specified.
6. If the context answers only part of the question, answer that part and clearly state which remaining detail is not available.
7. Do not say an application is approved or guaranteed unless the context explicitly says so.
8. Keep the answer concise and conversational.
9. If the context contains no answer to any part of the question, reply exactly:
I don't have that information.
"""

    user_prompt = f"""KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{query}

Answer from the context only. Check each part of the question before responding."""

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
            "num_predict": 120,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("message", {}).get("content", "").strip()
        return answer or "I don't have that information."
    except Exception as exc:
        print(f"Ollama request failed: {exc}")
        return "I'm unable to retrieve an answer right now."
