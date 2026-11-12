"""Régua da busca: dos pedaços devolvidos, quantos são os certos? (sem usar IA)

Trecho do `avaliacao/medir_busca.py` do rag-rpg, sem a parte de linha de comando.
Para cada pergunta da prova vê se as notas do gabarito aparecem nos `k` primeiros
pedaços e, quando há trecho de gabarito, se o texto dele veio inteiro.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from rag.buscador import Buscador, Modo
from rag.palavra import sem_acento

AVAL = Path(__file__).parent


def _limpo(texto: str) -> str:
    return re.sub(r"\s+", " ", sem_acento(texto.lower())).strip()


def carregar_perguntas() -> list[dict]:
    """Junta as três provas num formato só: id, grupo, pergunta, notas, trechos."""
    itens = []
    for item in json.loads((AVAL / "prova-final.json").read_text("utf-8")):
        itens.append(
            {
                "id": item["id"],
                "grupo": f"final/{item['tipo']}",
                "pergunta": item["pergunta"],
                "notas": item["notas"],
                "trechos": item["trechos"],
            }
        )
    for item in json.loads((AVAL / "prova-relacao.json").read_text("utf-8")):
        gabarito = item["gabarito"]
        trechos = (
            [t for lista in gabarito.values() for t in lista]
            if isinstance(gabarito, dict)
            else []
        )
        itens.append(
            {
                "id": item["id"],
                "grupo": f"relacao/{item['tipo']}",
                "pergunta": item["pergunta"],
                "notas": list(gabarito),
                "trechos": trechos,
            }
        )
    for item in json.loads((AVAL / "prova-confusao.json").read_text("utf-8")):
        itens.append(
            {
                "id": item["id"],
                "grupo": "confusao",
                "pergunta": item["pergunta"],
                "notas": item["notas"],
                "trechos": item["trechos"],
            }
        )
    return itens


def _trecho_veio(trecho: str, textos_da_nota: list[str]) -> bool:
    """O trecho veio se cada linha dele aparece no texto dos pedaços devolvidos."""
    todo = _limpo(" ".join(textos_da_nota))
    linhas = [_limpo(linha) for linha in trecho.splitlines() if linha.strip()]
    return all(linha in todo for linha in linhas)


def medir(item: dict, buscador: Buscador, modo: Modo, k: int) -> dict:
    resultados = buscador.buscar(item["pergunta"], k=k, modo=modo)
    por_nota = defaultdict(list)
    for r in resultados:
        por_nota[r.pedaco.nota].append(r.pedaco.texto)
    notas = item["notas"]
    achadas = [n for n in notas if n in por_nota]
    todos_textos = [t for textos in por_nota.values() for t in textos]
    trechos = item["trechos"]
    trechos_ok = [t for t in trechos if _trecho_veio(t, todos_textos)]
    return {
        "notas_total": len(notas),
        "notas_achadas": len(achadas),
        "faltaram": [n for n in notas if n not in por_nota],
        "trechos_total": len(trechos),
        "trechos_ok": len(trechos_ok),
    }


def resumir(linhas: list[dict]) -> dict:
    com_nota = [x for x in linhas if x["notas_total"]]
    com_trecho = [x for x in linhas if x["trechos_total"]]
    return {
        "perguntas": len(com_nota),
        "todas_as_notas": sum(x["notas_achadas"] == x["notas_total"] for x in com_nota),
        "recall_notas": round(
            sum(x["notas_achadas"] for x in com_nota)
            / max(1, sum(x["notas_total"] for x in com_nota)),
            3,
        ),
        "recall_trechos": round(
            sum(x["trechos_ok"] for x in com_trecho)
            / max(1, sum(x["trechos_total"] for x in com_trecho)),
            3,
        ),
    }
