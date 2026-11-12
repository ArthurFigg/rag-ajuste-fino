import zlib

import numpy as np
import pytest


class CodificadorFalso:
    """Vetores de contagem de palavras (por hash), só para testar sem baixar modelo."""

    nome = "falso"
    DIMENSAO = 64

    def _vetor(self, texto: str) -> np.ndarray:
        v = np.zeros(self.DIMENSAO, dtype=np.float32)
        for palavra in texto.lower().replace(":", " ").split():
            v[zlib.crc32(palavra.encode()) % self.DIMENSAO] += 1
        norma = np.linalg.norm(v)
        return v / norma if norma else v

    def pedacos(self, textos: list[str]) -> np.ndarray:
        return np.stack([self._vetor(t) for t in textos])

    def pergunta(self, texto: str) -> np.ndarray:
        return self._vetor(texto)


@pytest.fixture
def codificador_falso() -> CodificadorFalso:
    return CodificadorFalso()


@pytest.fixture
def pasta_notas_pequena(tmp_path):
    (tmp_path / "bestas").mkdir()
    (tmp_path / "bestas" / "dragao.md").write_text(
        "# Dragão\n\nO dragão vermelho vive na montanha de fogo e guarda ouro. "
        "Ele cospe chamas contra quem se aproxima do tesouro antigo.\n" * 3,
        encoding="utf-8",
    )
    (tmp_path / "bestas" / "elfo.md").write_text(
        "# Elfo\n\nO elfo da floresta caça com arco e fala com as árvores. "
        "Ele guarda as clareiras e protege os viajantes perdidos.\n" * 3,
        encoding="utf-8",
    )
    return tmp_path
