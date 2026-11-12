"""Gera as perguntas de treino com o Gemini (agy), em lotes, e monta os pares.

Passos (cada um é um subcomando):
  preparar  corta as notas em pedaços e divide em lotes (dados/lotes/lote-N/lote.json)
  validar   confere se a saída de um lote cobre os pedaços (usado por rodar_lotes.sh)
  juntar    lê a saída dos lotes, filtra e grava os pares de treino e de validação
Entre `preparar` e `juntar` roda `sh scripts/rodar_lotes.sh`.
"""

import argparse
import json
import sys
from pathlib import Path

from ajuste.ambiente import RAIZ_PROJETO, pasta_notas
from ajuste.dados import (
    dividir_em_lotes,
    exportar_pedacos,
    ler_saida_do_gemini,
    montar_pares,
    separar_treino_validacao,
)

DADOS = RAIZ_PROJETO / "dados"
LOTES = 6
COBERTURA_MINIMA = 0.9


def _salvar(caminho: Path, conteudo) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1), "utf-8")


def preparar() -> None:
    pedacos = exportar_pedacos(pasta_notas())
    _salvar(DADOS / "pedacos.json", pedacos)
    for numero, lote in enumerate(dividir_em_lotes(pedacos, LOTES), start=1):
        itens = [{"id": p["id"], "texto": p["texto"]} for p in lote]
        _salvar(DADOS / "lotes" / f"lote-{numero}" / "lote.json", itens)
    print(f"{len(pedacos)} pedaços em {LOTES} lotes")


def _cobertura(pasta_lote: Path, saida: Path) -> tuple[float, dict[str, list[str]]]:
    esperados = {i["id"] for i in json.loads((pasta_lote / "lote.json").read_text("utf-8"))}
    perguntas = ler_saida_do_gemini(saida.read_text("utf-8"))
    return len(esperados & perguntas.keys()) / len(esperados), perguntas


def validar(pasta_lote: Path, saida: Path) -> int:
    try:
        cobertura, _ = _cobertura(pasta_lote, saida)
    except (ValueError, KeyError, json.JSONDecodeError) as erro:
        print(f"saída inválida: {erro}")
        return 1
    print(f"cobertura {cobertura:.0%}")
    return 0 if cobertura >= COBERTURA_MINIMA else 1


def juntar() -> None:
    pedacos = json.loads((DADOS / "pedacos.json").read_text("utf-8"))
    por_id: dict[str, list[str]] = {}
    for lote in sorted((DADOS / "lotes").glob("lote-*")):
        saida = lote / "saida-final.txt"
        if saida.exists():
            por_id.update(ler_saida_do_gemini(saida.read_text("utf-8")))
    pares = montar_pares(pedacos, por_id)
    treino, validacao = separar_treino_validacao(pares)
    _salvar(DADOS / "pares-treino.json", treino)
    _salvar(DADOS / "pares-validacao.json", validacao)
    cobertos = len({p["id"] for p in pares})
    print(
        f"{len(pares)} pares ({len(treino)} treino, {len(validacao)} validação) "
        f"de {cobertos}/{len(pedacos)} pedaços"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("preparar")
    sub.add_parser("juntar")
    v = sub.add_parser("validar")
    v.add_argument("lote", type=Path)
    v.add_argument("saida", type=Path)
    args = parser.parse_args()
    if args.comando == "preparar":
        preparar()
    elif args.comando == "juntar":
        juntar()
    else:
        return validar(args.lote, args.saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
