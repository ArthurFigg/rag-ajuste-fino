"""Texto -> números (embeddings), com um modelo multilíngue que roda local.

O modelo vem pronto (treinado por outros); é a única peça que não se refaz à mão.
A pesquisa de 07/11/2026 mostrou mE5 e BGE-M3 bem acima do embedding da OpenAI em
busca multilíngue (MIRACL), e ambos são gratuitos. Começamos pelo menor (mE5-small).
"""

from typing import Protocol

import numpy as np

MODELO_PADRAO = "intfloat/multilingual-e5-small"

# mE5 pede os prefixos "query: " na pergunta e "passage: " no trecho (cartão do modelo).
PREFIXOS = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-m3": ("", ""),
}


class Codificador(Protocol):
    """O que o resto do código espera de qualquer modelo de embeddings."""

    nome: str

    def pedacos(self, textos: list[str]) -> np.ndarray: ...

    def pergunta(self, texto: str) -> np.ndarray: ...


class CodificadorLocal:
    """Modelo do `sentence-transformers`, baixado uma vez e usado offline."""

    def __init__(self, nome: str = MODELO_PADRAO) -> None:
        from sentence_transformers import SentenceTransformer

        self.nome = nome
        self._modelo = SentenceTransformer(nome)
        self._prefixo_pergunta, self._prefixo_pedaco = PREFIXOS.get(nome, ("", ""))

    def pedacos(self, textos: list[str]) -> np.ndarray:
        entrada = [self._prefixo_pedaco + t for t in textos]
        return self._codificar(entrada)

    def pergunta(self, texto: str) -> np.ndarray:
        return self._codificar([self._prefixo_pergunta + texto])[0]

    def _codificar(self, textos: list[str]) -> np.ndarray:
        vetores = self._modelo.encode(
            textos, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return np.asarray(vetores, dtype=np.float32)
