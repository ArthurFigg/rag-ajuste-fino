"""Dados de treino: perguntas sintéticas (feitas pelo Gemini) para cada pedaço.

Fluxo: exportar os pedaços -> dividir em lotes -> o Gemini escreve perguntas ->
validar e filtrar -> montar pares (pergunta, pedaço) -> separar treino e validação.
Nenhuma pergunta da prova real entra aqui: ela fica guardada para medir no fim.
"""

import json
import random
import re
import unicodedata
from pathlib import Path

from rag.cortador import cortar_nota
from rag.indice import ler_notas

PERGUNTAS_POR_PEDACO = 4
MAXIMO_POR_PEDACO = 5
TAMANHO_MINIMO = 20
TAMANHO_MAXIMO = 300
# Pergunta que aponta para o documento só faz sentido com o documento na mão.
REFERE_AO_TEXTO = re.compile(
    r"\b(esta nota|essa nota|o texto|o trecho|a passagem|o documento|a secao|"
    r"o paragrafo|acima|abaixo|segundo o)\b"
)


def _sem_acento(texto: str) -> str:
    base = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in base if unicodedata.category(c) != "Mn")


def exportar_pedacos(pasta_notas: Path) -> list[dict]:
    """Corta as notas como o RAG corta e dá um id estável a cada pedaço."""
    pedacos = []
    for caminho, texto in ler_notas(pasta_notas).items():
        for p in cortar_nota(caminho, texto):
            pedacos.append(
                {
                    "id": f"p{len(pedacos):04d}",
                    "nota": p.nota,
                    "texto": p.texto_indexado,
                }
            )
    return pedacos


def dividir_em_lotes(itens: list[dict], quantidade: int) -> list[list[dict]]:
    """Reparte a lista em `quantidade` lotes de tamanho parecido, na mesma ordem."""
    tamanho = -(-len(itens) // quantidade)
    return [itens[i : i + tamanho] for i in range(0, len(itens), tamanho)]


def ler_saida_do_gemini(saida: str) -> dict[str, list[str]]:
    """Pega a lista JSON da saída (ignora texto solto antes e depois) -> id: perguntas."""
    inicio, fim = saida.find("["), saida.rfind("]")
    if inicio < 0 or fim < inicio:
        raise ValueError("a saída não tem uma lista JSON")
    # strict=False: o Gemini às vezes deixa uma quebra de linha solta dentro da frase
    itens = json.loads(saida[inicio : fim + 1], strict=False)
    return {
        str(i["id"]): [str(p) for p in i["perguntas"]]
        for i in itens
        if isinstance(i.get("perguntas"), list)
    }


def pergunta_valida(pergunta: str) -> bool:
    limpa = pergunta.strip()
    if not TAMANHO_MINIMO <= len(limpa) <= TAMANHO_MAXIMO:
        return False
    if not limpa.endswith("?"):
        return False
    return REFERE_AO_TEXTO.search(_sem_acento(limpa)) is None


def filtrar_perguntas(perguntas: list[str]) -> list[str]:
    """Fica com as válidas, sem repetir (ignora caixa e acento), até o máximo."""
    vistas: set[str] = set()
    escolhidas: list[str] = []
    for bruta in perguntas:
        pergunta = " ".join(bruta.split())
        chave = _sem_acento(pergunta)
        if chave in vistas or not pergunta_valida(pergunta):
            continue
        vistas.add(chave)
        escolhidas.append(pergunta)
    return escolhidas[:MAXIMO_POR_PEDACO]


def montar_pares(
    pedacos: list[dict], perguntas_por_id: dict[str, list[str]]
) -> list[dict]:
    """Um par {pergunta, pedaco, id} para cada pergunta aceita."""
    pares = []
    for pedaco in pedacos:
        for pergunta in filtrar_perguntas(perguntas_por_id.get(pedaco["id"], [])):
            pares.append(
                {"id": pedaco["id"], "pergunta": pergunta, "pedaco": pedaco["texto"]}
            )
    return pares


def separar_treino_validacao(
    pares: list[dict], fracao_validacao: float = 0.1, semente: int = 42
) -> tuple[list[dict], list[dict]]:
    """Separa por pergunta (o pedaço aparece nos dois lados; as perguntas não)."""
    sorteador = random.Random(semente)
    embaralhados = pares[:]
    sorteador.shuffle(embaralhados)
    corte = max(1, round(len(embaralhados) * fracao_validacao))
    return embaralhados[corte:], embaralhados[:corte]
