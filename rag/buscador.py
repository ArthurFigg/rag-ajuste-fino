"""Buscar os pedaços mais úteis para uma pergunta."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from rag.cortador import Pedaco
from rag.embeddings import Codificador
from rag.fontes import link_obsidian
from rag.indice import Indice
from rag.mista import fundir_rrf
from rag.palavra import palavras
from rag.vetor import pontuar


class Modo(Enum):
    VETOR = "vetor"
    PALAVRA = "palavra"
    MISTA = "mista"


@dataclass(frozen=True)
class Resultado:
    pedaco: Pedaco
    nota: float  # nota do pedaço na busca (0 para vizinhos incluídos à força)

    @property
    def link(self) -> str:
        return link_obsidian(self.pedaco.nota)


class Buscador:
    def __init__(self, indice: Indice, codificador: Codificador) -> None:
        self.indice = indice
        self.codificador = codificador

    def notas(self, pergunta: str, modo: Modo = Modo.MISTA) -> np.ndarray:
        """Nota de cada pedaço do índice para a pergunta."""
        if modo is Modo.PALAVRA:
            return self.indice.bm25.pontuar(palavras(pergunta))
        por_vetor = pontuar(self.indice.vetores, self.codificador.pergunta(pergunta))
        if modo is Modo.VETOR:
            return por_vetor
        return fundir_rrf(self.indice.bm25.pontuar(palavras(pergunta)), por_vetor)

    def buscar(
        self,
        pergunta: str,
        k: int = 8,
        modo: Modo = Modo.MISTA,
        vizinhos: int = 0,
    ) -> list[Resultado]:
        """Os `k` melhores pedaços. Com `vizinhos` > 0, traz também os pedaços de
        antes e depois na mesma nota (para uma ficha não chegar partida)."""
        notas = self.notas(pergunta, modo)
        melhores = np.argsort(-notas, kind="stable")[:k]
        escolhidos = {int(i): float(notas[i]) for i in melhores if notas[i] > 0}
        if vizinhos > 0:
            escolhidos = self._com_vizinhos(escolhidos, vizinhos)
        ordenados = sorted(escolhidos, key=lambda i: (-escolhidos[i], i))
        return [Resultado(self.indice.pedacos[i], escolhidos[i]) for i in ordenados]

    def _com_vizinhos(self, escolhidos: dict[int, float], vizinhos: int):
        pedacos = self.indice.pedacos
        ampliados = dict(escolhidos)
        for i in escolhidos:
            for j in range(i - vizinhos, i + vizinhos + 1):
                mesma_nota = (
                    0 <= j < len(pedacos) and pedacos[j].nota == pedacos[i].nota
                )
                if mesma_nota:
                    ampliados.setdefault(j, 0.0)
        return ampliados
