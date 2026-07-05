"""BM25-based RAG system — replaces the n-gram SimpleRAG with proper term-based retrieval"""

import os
import json
import re
import math
from collections import Counter


class BetterRAG:
    def __init__(self, storage_path="rag_store.json"):
        self.storage_path = storage_path
        self.documents = []
        self.avgdl = 0
        self.k1 = 1.5
        self.b = 0.75
        self._idf = {}
        self._load()

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.avgdl = data.get('avgdl', 0)
                    self._rebuild_idf()
            except Exception:
                self.documents = []

    def _save(self):
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({'documents': self.documents, 'avgdl': self.avgdl}, f, indent=2)
        except Exception:
            pass

    def _rebuild_idf(self):
        N = len(self.documents)
        if N == 0:
            self._idf = {}
            return
        df = Counter()
        for doc in self.documents:
            for term in set(doc['tokens']):
                df[term] += 1
        self._idf = {term: math.log(1 + (N - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def add_text(self, text, source=''):
        tokens = self._tokenize(text)
        if len(tokens) < 5:
            return
        self.documents.append({'text': text, 'source': source, 'tokens': tokens})
        N = len(self.documents)
        self.avgdl = sum(len(d['tokens']) for d in self.documents) / N
        self._rebuild_idf()
        self._save()

    def add_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            self.add_text(text, source=os.path.basename(filepath))
        except Exception:
            pass

    def search(self, query, top_k=5):
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []
        scores = []
        for doc in self.documents:
            dl = len(doc['tokens'])
            tf = Counter(doc['tokens'])
            score = 0
            for term in query_tokens:
                if term not in self._idf:
                    continue
                idf = self._idf[term]
                f = tf.get(term, 0)
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((score, doc))
        scores.sort(key=lambda x: -x[0])
        results = []
        seen = set()
        for score, doc in scores:
            if score <= 0:
                continue
            text = doc['text']
            if text not in seen:
                results.append({'text': text, 'source': doc.get('source', ''), 'score': round(score, 3)})
                seen.add(text)
                if len(results) >= top_k:
                    break
        return results

    def build_context(self, query, top_k=3):
        results = self.search(query, top_k)
        if not results:
            return ''
        parts = []
        for i, r in enumerate(results):
            src = f' [{r["source"]}]' if r['source'] else ''
            parts.append(f'[{i+1}]{src} {r["text"]}')
        return '\n\n' + '\n\n'.join(parts)

    def list_sources(self):
        return sorted({d.get('source', '') for d in self.documents if d.get('source')})

    def clear(self):
        self.documents = []
        self._idf = {}
        self.avgdl = 0
        self._save()
