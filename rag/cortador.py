"""Corte das notas do vault em pedaços para a busca.

Decisão (pesquisa de 07/11/2026): corte simples fica perto do melhor e um estudo
mostrou que sobreposição não ajuda. Aqui o corte é por seção da nota (título do
Markdown), porque uma ficha não pode ser partida ao meio, e cada pedaço leva o nome
da nota e da pasta no começo (o "contexto" que a Anthropic gera com IA, de graça).
"""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

PADRAO_TITULO = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)
PADRAO_IMAGEM = re.compile(r"^!\[\[[^\]]*\]\]\s*$", re.MULTILINE)
LIMITE_MAXIMO = 1500
LIMITE_MINIMO = 200
TEXTO_NOTA_SEM_TEXTO = (
    "(nota sem texto: o arquivo existe, mas ainda não tem conteúdo escrito)"
)


@dataclass(frozen=True)
class Pedaco:
    nota: str  # caminho relativo da nota, com "/"
    secao: str  # título da seção (vazio no começo da nota)
    texto: str  # conteúdo do pedaço, sem o cabeçalho de contexto
    ordem: int  # posição do pedaço dentro da nota

    @property
    def titulo_nota(self) -> str:
        return PurePosixPath(self.nota).stem

    @property
    def cabecalho(self) -> str:
        pasta = PurePosixPath(self.nota).parent.as_posix()
        partes = [f"Nota: {self.titulo_nota}"]
        if pasta != ".":
            partes.append(f"Pasta: {pasta}")
        if self.secao:
            partes.append(f"Seção: {self.secao}")
        return "[" + " | ".join(partes) + "]"

    @property
    def texto_indexado(self) -> str:
        return f"{self.cabecalho}\n{self.texto}"


def _limpar_titulo(linha: str) -> str:
    """Tira os # e o negrito (**) do título para o cabeçalho ficar legível."""
    return linha.lstrip("#").replace("*", "").strip()


def _dividir_por_secao(texto: str) -> list[tuple[str, str]]:
    """Devolve (título, bloco com o título) para cada seção, na ordem."""
    inicios = [m.start() for m in PADRAO_TITULO.finditer(texto)]
    if not inicios or inicios[0] > 0:
        inicios.insert(0, 0)
    fins = inicios[1:] + [len(texto)]
    secoes = []
    for inicio, fim in zip(inicios, fins):
        bloco = texto[inicio:fim].strip()
        if not bloco:
            continue
        primeira = bloco.splitlines()[0]
        titulo = _limpar_titulo(primeira) if primeira.startswith("#") else ""
        secoes.append((titulo, bloco))
    return secoes


def _juntar_curtas(secoes: list[tuple[str, str]], minimo: int):
    """Une cada seção curta à seguinte, para não gerar pedaço sem assunto."""
    juntas: list[tuple[str, str]] = []
    pendente: tuple[str, str] | None = None
    for titulo, bloco in secoes:
        if pendente is not None:
            titulo = pendente[0] or titulo
            bloco = pendente[1] + "\n\n" + bloco
            pendente = None
        if len(bloco) < minimo:
            pendente = (titulo, bloco)
        else:
            juntas.append((titulo, bloco))
    if pendente is not None:
        if juntas:
            anterior = juntas.pop()
            juntas.append((anterior[0], anterior[1] + "\n\n" + pendente[1]))
        else:
            juntas.append(pendente)
    return juntas


def _quebrar_longo(bloco: str, maximo: int) -> list[str]:
    """Quebra um bloco grande em grupos de parágrafos de até `maximo` caracteres."""
    if len(bloco) <= maximo:
        return [bloco]
    partes: list[str] = []
    atual = ""
    for paragrafo in re.split(r"\n\s*\n", bloco):
        for trecho in _fatiar(paragrafo, maximo):
            if atual and len(atual) + len(trecho) + 2 > maximo:
                partes.append(atual)
                atual = trecho
            else:
                atual = f"{atual}\n\n{trecho}" if atual else trecho
    if atual:
        partes.append(atual)
    return partes


def _fatiar(paragrafo: str, maximo: int) -> list[str]:
    """Parágrafo maior que o limite é cortado por linhas, e por fim por tamanho."""
    if len(paragrafo) <= maximo:
        return [paragrafo]
    fatias: list[str] = []
    atual = ""
    for linha in paragrafo.splitlines():
        while len(linha) > maximo:
            if atual:
                fatias.append(atual)
                atual = ""
            fatias.append(linha[:maximo])
            linha = linha[maximo:]
        if atual and len(atual) + len(linha) + 1 > maximo:
            fatias.append(atual)
            atual = linha
        else:
            atual = f"{atual}\n{linha}" if atual else linha
    if atual:
        fatias.append(atual)
    return fatias


def cortar_nota(
    caminho: str,
    texto: str,
    maximo: int = LIMITE_MAXIMO,
    minimo: int = LIMITE_MINIMO,
) -> list[Pedaco]:
    """Corta uma nota em pedaços por seção.

    Nota sem texto (vazia ou só com imagem) vira um pedaço-aviso: sem ele a busca não
    sabe que o arquivo existe e responde "não consta" para algo que o dono já criou.
    """
    limpo = PADRAO_IMAGEM.sub("", texto).strip()
    if not limpo:
        return [Pedaco(caminho, "", TEXTO_NOTA_SEM_TEXTO, 0)]
    secoes = _juntar_curtas(_dividir_por_secao(limpo), minimo)
    pedacos: list[Pedaco] = []
    for titulo, bloco in secoes:
        for parte in _quebrar_longo(bloco, maximo):
            pedacos.append(Pedaco(caminho, titulo, parte, len(pedacos)))
    return pedacos
