"""Codificador de texto que serve tanto para o modelo original quanto para o ajustado.

O `CodificadorLocal` do rag-rpg decide os prefixos pelo nome do modelo; um modelo
ajustado fica numa pasta com outro nome e perderia os prefixos. Aqui eles são fixos.
"""

import numpy as np

PREFIXO_PERGUNTA = "query: "
PREFIXO_PEDACO = "passage: "


class CodificadorE5:
    """Modelo da família multilingual-e5 (nome do Hugging Face ou pasta local)."""

    def __init__(self, nome: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.nome = nome
        self._modelo = SentenceTransformer(nome)

    def pedacos(self, textos: list[str]) -> np.ndarray:
        return self._codificar([PREFIXO_PEDACO + t for t in textos])

    def pergunta(self, texto: str) -> np.ndarray:
        return self._codificar([PREFIXO_PERGUNTA + texto])[0]

    def _codificar(self, textos: list[str]) -> np.ndarray:
        vetores = self._modelo.encode(
            textos, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return np.asarray(vetores, dtype=np.float32)
