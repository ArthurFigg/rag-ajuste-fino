"""Busca por palavra (BM25) escrita à mão.

BM25 pontua cada pedaço pelas palavras da pergunta que ele contém: palavra rara vale
mais que palavra comum, e pedaço curto não perde para pedaço longo só por tamanho.
A Anthropic mostrou que ele acha identificadores exatos que o vetor perde, e a fusão
das duas buscas foi melhor que cada uma sozinha. O `rank_bm25` está parado desde 2022.
"""

import math
import re
import unicodedata
from collections import Counter

import numpy as np

PADRAO_PALAVRA = re.compile(r"\w+", re.UNICODE)


def sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def palavras(texto: str) -> list[str]:
    """Minúsculas, sem acento ("capitã" e "capita" viram a mesma palavra)."""
    achadas = PADRAO_PALAVRA.findall(sem_acento(texto).lower())
    return [p for p in achadas if len(p) > 1]


class BM25:
    def __init__(
        self, documentos: list[list[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self._frequencias = [Counter(doc) for doc in documentos]
        self._tamanhos = np.array([len(doc) for doc in documentos], dtype=np.float32)
        total = max(len(documentos), 1)
        self._media = float(self._tamanhos.sum()) / total or 1.0
        aparece: Counter[str] = Counter()
        for freq in self._frequencias:
            aparece.update(freq.keys())
        self._idf = {
            p: math.log(1 + (total - n + 0.5) / (n + 0.5)) for p, n in aparece.items()
        }

    def pontuar(self, pergunta: list[str]) -> np.ndarray:
        """Nota de cada documento para a pergunta (já dividida em palavras)."""
        notas = np.zeros(len(self._frequencias), dtype=np.float32)
        for palavra in set(pergunta):
            idf = self._idf.get(palavra)
            if idf is None:
                continue
            for i, freq in enumerate(self._frequencias):
                f = freq.get(palavra, 0)
                if f:
                    norma = 1 - self.b + self.b * self._tamanhos[i] / self._media
                    notas[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norma)
        return notas
