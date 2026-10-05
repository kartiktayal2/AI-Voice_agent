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

RETRIEVAL_TOP_K = 5
FINAL_TOP_K = 3

RERANK_THRESHOLD = 0.50


# ============================================================
# GENERIC LOAN TYPE EXTRACTION
# ============================================================

def extract_loan_type(text):
    """
    Extract the loan type from the first line of a chunk
    or from the user's question.

    Examples:

    Personal Loan
    The personal loan amount ranges...

    -> personal

    What is the interest rate for a personal loan?
    -> personal
    """

    text = text.lower().strip()

    # --------------------------------------------------------
    # If this is a chunk, use its first line.
    # --------------------------------------------------------

    first_line = text.splitlines()[0].strip()

    # Example:
    # "Personal Loan" -> "personal"
    heading_match = re.fullmatch(
        r"([a-z]+(?:\s+[a-z]+)?)\s+loan",
        first_line
    )

    if heading_match:
        return heading_match.group(1).strip()

    # --------------------------------------------------------
    # Otherwise search the query.
    # --------------------------------------------------------

    matches = re.findall(
        r"\b([a-z]+(?:\s+[a-z]+)?)\s+loan\b",
        text
    )

    if not matches:
        return None

    # Remove articles such as "a" / "the"
    topic = matches[-1].strip()

    topic = re.sub(
        r"^(a|an|the)\s+",
        "",
        topic
    )

    return topic


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    documents = []

    print("Loading documents...")

    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


# ============================================================
# HYBRID CHUNKING
# ============================================================

def chunk_text(text, max_words=120):
    """
    Hybrid chunking:

    1. Uses headings/sections to preserve meaning.
    2. Splits large sections by word count.
    """

    sections = []

    raw_sections = text.split("\n\n")

    current_section = ""

    for block in raw_sections:

        block = block.strip()

        if not block:
            continue

        # Treat short blocks as possible headings.
        if len(block.split()) <= 5:

            if current_section:
                sections.append(
                    current_section.strip()
                )

            current_section = block

        else:

            if current_section:
                current_section += "\n" + block
            else:
                current_section = block

    if current_section:
        sections.append(
            current_section.strip()
        )

    # --------------------------------------------------------
    # Split large sections
    # --------------------------------------------------------

    final_chunks = []

    for section in sections:

        words = section.split()

        if len(words) <= max_words:

            final_chunks.append(section)

        else:

            for i in range(
                0,
                len(words),
                max_words
            ):

                chunk = " ".join(
                    words[i:i + max_words]
                )

                final_chunks.append(chunk)

    return final_chunks


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(self):

        print("Loading RAG system...")

        # ----------------------------------------------------
        # Load documents
        # ----------------------------------------------------

        documents = load_documents()

        self.chunks = []

        for document in documents:

            chunks = chunk_text(
                document["text"]
            )

            for chunk in chunks:

                self.chunks.append({
                    "source": document["source"],
                    "text": chunk
                })

        print(
            f"Total chunks: {len(self.chunks)}"
        )

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            "float32"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        self.faiss_index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.faiss_index.add(
            embeddings
        )

        print("FAISS index created.")

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        tokenized_documents = [
            text.lower().split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print("BM25 index created.")

        # ----------------------------------------------------
        # Load reranker
        # ----------------------------------------------------

        print("Loading reranker model...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("Reranker loaded.")

        print("RAG system ready.")

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def retrieve_candidates(
        self,
        query,
        top_k=RETRIEVAL_TOP_K
    ):

        # ----------------------------------------------------
        # Vector search
        # ----------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        vector_scores, vector_indices = (
            self.faiss_index.search(
                query_embedding,
                min(
                    top_k,
                    len(self.chunks)
                )
            )
        )

        vector_scores = vector_scores[0]
        vector_indices = vector_indices[0]

        # ----------------------------------------------------
        # BM25 search
        # ----------------------------------------------------

        tokenized_query = query.lower().split()

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_top_indices = np.argsort(
            bm25_scores
        )[::-1][
            :min(
                top_k,
                len(self.chunks)
            )
        ]

        # ----------------------------------------------------
        # Normalize BM25 scores
        # ----------------------------------------------------

        max_bm25 = np.max(
            bm25_scores
        )

        if max_bm25 > 0:

            normalized_bm25 = (
                bm25_scores / max_bm25
            )

        else:

            normalized_bm25 = np.zeros_like(
                bm25_scores
            )

        # ----------------------------------------------------
        # Combine FAISS + BM25 candidates
        # ----------------------------------------------------

        candidate_indices = set(
            vector_indices.tolist()
        )

        candidate_indices.update(
            bm25_top_indices.tolist()
        )

        candidates = []

        for index in candidate_indices:

            vector_score = 0.0

            if index in vector_indices:

                position = np.where(
                    vector_indices == index
                )[0][0]

                vector_score = float(
                    vector_scores[position]
                )

            bm25_score = float(
                normalized_bm25[index]
            )

            hybrid_score = (
                0.6 * vector_score
                +
                0.4 * bm25_score
            )

            candidates.append({
                "index": int(index),
                "source": self.chunks[index]["source"],
                "text": self.chunks[index]["text"],
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score
            })

        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return candidates[:top_k]

    # ========================================================
    # TOPIC CONSISTENCY CHECK
    # ========================================================

    def is_topic_consistent(
        self,
        query,
        result
    ):
        """
        Prevent information from one loan type
        being used for another loan type.

        This is dynamic.

        It does NOT contain hardcoded rules such as:
            if "home loan"
            if "car loan"

        Instead it extracts the loan type from:
            - the user query
            - the retrieved chunk
        """

        query_loan_type = extract_loan_type(
            query
        )

        chunk_loan_type = extract_loan_type(
            result["text"]
        )

        print(
            f"DEBUG | Query topic: "
            f"{query_loan_type} | "
            f"Chunk topic: "
            f"{chunk_loan_type}"
        )

        # If the query does not explicitly
        # mention a loan type, don't reject it.
        if query_loan_type is None:
            return True

        # If the chunk does not contain
        # a detectable loan type, don't reject it.
        if chunk_loan_type is None:
            return True

        # Only allow matching loan types.
        return (
            query_loan_type
            == chunk_loan_type
        )

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query,
        candidates,
        final_top_k=FINAL_TOP_K,
        threshold=RERANK_THRESHOLD
    ):

        if not candidates:
            return []

        # ----------------------------------------------------
        # Create query-document pairs
        # ----------------------------------------------------

        pairs = [
            [
                query,
                candidate["text"]
            ]
            for candidate in candidates
        ]

        # ----------------------------------------------------
        # CrossEncoder scores
        # ----------------------------------------------------

        raw_scores = self.reranker.predict(
            pairs
        )

        raw_scores = np.asarray(
            raw_scores
        )

        # Convert logits to probability-like
        # values for easier thresholding.
        rerank_scores = (
            1 /
            (
                1 +
                np.exp(-raw_scores)
            )
        )

        # ----------------------------------------------------
        # Attach reranker scores
        # ----------------------------------------------------

        results = []

        for candidate, score in zip(
            candidates,
            rerank_scores
        ):

            result = candidate.copy()

            result["rerank_score"] = float(
                score
            )

            results.append(result)

        # ----------------------------------------------------
        # Sort by reranker score
        # ----------------------------------------------------

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # ----------------------------------------------------
        # Relevance gate
        #
        # BOTH conditions must be true:
        #
        # 1. Reranker score is high enough.
        # 2. Loan type is consistent.
        # ----------------------------------------------------

        relevant_results = [
            result
            for result in results
            if (
                result["rerank_score"]
                >= threshold
                and
                self.is_topic_consistent(
                    query,
                    result
                )
            )
        ]

        return relevant_results[
            :final_top_k
        ]

    # ========================================================
    # COMPLETE SEARCH
    # ========================================================

    def search(
        self,
        query,
        final_top_k=FINAL_TOP_K,
        threshold=RERANK_THRESHOLD
    ):

        candidates = self.retrieve_candidates(
            query
        )

        results = self.rerank(
            query,
            candidates,
            final_top_k=final_top_k,
            threshold=threshold
        )

        return results


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(
    query,
    context
):

    system_prompt = """
You are a loan customer support assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not assume missing information.
4. Do not transfer information from one loan type,
   product, or category to another.
5. If the answer is not present in the context,
   say exactly:
   "I don't have that information."
6. Keep the answer short and conversational.
"""

    user_prompt = f"""
Context:

{context}


User question:

{query}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "options": {
            "num_predict": 60
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"].strip()

    except Exception as e:

        return (
            f"Ollama error: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    retriever = HybridRetriever()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = retriever.search(
            query
        )

        print("\nRetrieved Results:")

        if not results:

            print(
                "No sufficiently relevant "
                "information found."
            )

            print()

            continue

        # ----------------------------------------------------
        # Display retrieved chunks
        # ----------------------------------------------------

        for result in results:

            print(
                f"\nRerank Score: "
                f"{result['rerank_score']:.4f}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print(
                result["text"]
            )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        # ----------------------------------------------------
        # Ask Ollama
        # ----------------------------------------------------

        answer = ask_ollama(
            query,
            context
        )

        print(
            f"\nAssistant: {answer}"
        )

        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()