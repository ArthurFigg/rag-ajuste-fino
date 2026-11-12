"""Busca por significado: compara o vetor da pergunta com o de cada pedaço."""

import numpy as np


def pontuar(matriz: np.ndarray, pergunta: np.ndarray) -> np.ndarray:
    """Similaridade de cosseno. Os vetores já vêm normalizados, então é um produto."""
    if matriz.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matriz @ pergunta
