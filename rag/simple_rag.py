"""Lightweight RAG system (MiniRAG-inspired) for small language models"""

import os
import json


class SimpleRAG:
    """
    Simple RAG using TF-IDF-like embedding and cosine search.
    Does not rely on external embedding models — uses built-in character n-grams.
    """

    def __init__(self, storage_path="rag_store.json"):
        self.storage_path = storage_path
        self.chunks = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
            except (json.JSONDecodeError, IOError):
                self.chunks = []

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump({"chunks": self.chunks}, f, indent=2)
        except IOError:
            pass

    def _char_ngrams(self, text, n=3):
        """Extract character n-grams for simple matching"""
        text = text.lower()
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    def _score(self, query_ngrams, chunk_ngrams):
        """Jaccard similarity between n-gram sets"""
        if not query_ngrams or not chunk_ngrams:
            return 0
        intersection = query_ngrams & chunk_ngrams
        union = query_ngrams | chunk_ngrams
        return len(intersection) / len(union)

    def add_text(self, text, source=""):
        """Split text into chunks and add to index"""
        chunks = self._chunk(text)
        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:
                continue
            self.chunks.append({
                "text": chunk,
                "source": source,
                "index": i,
            })
        self._save()

    def add_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.add_text(text, source=filepath)

    def _chunk(self, text, max_chars=500, overlap=50):
        """Simple chunking by character count with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            start += max_chars - overlap
        return chunks

    def search(self, query, top_k=3):
        """Search for most relevant chunks"""
        query_ngrams = self._char_ngrams(query)
        scored = []
        for chunk in self.chunks:
            chunk_ngrams = self._char_ngrams(chunk["text"])
            score = self._score(query_ngrams, chunk_ngrams)
            if score > 0.02:
                scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        results = []
        seen = set()
        for score, chunk in scored[:top_k]:
            text = chunk["text"]
            if text not in seen:
                results.append({"text": text, "source": chunk.get("source", ""), "score": round(score, 3)})
                seen.add(text)
        return results

    def build_context(self, query, top_k=3):
        results = self.search(query, top_k)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results):
            src = f" [{r['source']}]" if r['source'] else ""
            parts.append(f"[{i+1}]{src} {r['text']}")
        return "\n\n" + "\n\n".join(parts)
