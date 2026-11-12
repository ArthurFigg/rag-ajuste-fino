"""Busca mista: junta a nota da busca por palavra com a da busca por significado.

Padrão: RRF (fusão por posição), testado aqui em 08/11/2026 contra a soma das notas
normalizadas (o que o estudo MIRACL usou). Nas 46 perguntas antigas o RRF achou 89%
das notas certas com 12 pedaços (soma: 86%) e nas 10 sem palavra em comum 79% (soma:
57%). A soma deixa a busca por palavra, que nesses casos só acha ruído, puxar a nota
certa para baixo (a Zifel caiu de 6º no vetor para 27º). O RRF usa só a posição de cada
pedaço em cada busca, então não tem peso para ajustar. A soma (`fundir`) continua
aqui para comparação. Detalhe: etapa3/resultado-semantica-existe.md, seção 5.
"""

import numpy as np

CONSTANTE_RRF = 60  # valor padrão da literatura; não ajustado a este vault


def normalizar(notas: np.ndarray) -> np.ndarray:
    """Leva as notas para 0..1. Se todas forem iguais, devolve zeros."""
    if notas.size == 0:
        return notas
    menor, maior = float(notas.min()), float(notas.max())
    if maior - menor < 1e-9:
        return np.zeros_like(notas)
    return (notas - menor) / (maior - menor)


def fundir_rrf(*buscas: np.ndarray, constante: int = CONSTANTE_RRF) -> np.ndarray:
    """Soma 1/(constante + posição) de cada pedaço em cada busca.

    Pedaço com nota 0 numa busca (nenhuma palavra em comum) não ganha nada nela.
    """
    total = np.zeros_like(buscas[0], dtype=np.float64)
    for notas in buscas:
        ordem = np.argsort(-notas, kind="stable")
        posicao = np.empty_like(ordem)
        posicao[ordem] = np.arange(len(notas))
        total += np.where(notas > 0, 1.0 / (constante + posicao + 1), 0.0)
    return total


def fundir(
    por_palavra: np.ndarray, por_vetor: np.ndarray, peso_vetor: float = 0.5
) -> np.ndarray:
    """Soma ponderada das duas notas normalizadas (peso 0,5 = igual, sem ajuste)."""
    return (1 - peso_vetor) * normalizar(por_palavra) + peso_vetor * normalizar(
        por_vetor
    )
