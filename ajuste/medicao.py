"""Mede um modelo de embeddings contra a prova de perguntas reais.

A régua é a do rag-rpg: para cada pergunta, as notas certas aparecem entre os `k`
primeiros pedaços devolvidos? Assim o "antes" e o "depois" usam a mesma conta.
"""

from pathlib import Path

from avaliacao.medir_busca import carregar_perguntas, medir, resumir
from rag.buscador import Buscador, Modo
from rag.indice import montar_indice

KS = (5, 8, 12, 20)


def carregar_prova() -> list[dict]:
    """As perguntas da prova real (grupos final, relacao e confusao)."""
    return carregar_perguntas()


def medir_modelo(
    codificador, pasta_notas: Path, perguntas: list[dict], ks: tuple[int, ...] = KS
) -> dict:
    """Monta o índice com o modelo e devolve o resumo e as linhas por pergunta.

    Mede a busca só por vetor (a parte que o ajuste muda) e a mista (o que o RAG
    usa de verdade, que junta vetor com palavra).
    """
    indice = montar_indice(pasta_notas, codificador)
    buscador = Buscador(indice, codificador)
    linhas: dict[str, list[dict]] = {}
    for modo in (Modo.VETOR, Modo.MISTA):
        for k in ks:
            linhas[f"{modo.value}/{k}"] = [
                {**medir(p, buscador, modo, k), "id": p["id"], "grupo": p["grupo"]}
                for p in perguntas
            ]
    resumo = {chave: resumir(itens) for chave, itens in linhas.items()}
    return {
        "modelo": codificador.nome,
        "pedacos": len(indice.pedacos),
        "perguntas": len(perguntas),
        "resumo": resumo,
        "linhas": linhas,
    }
