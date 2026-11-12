"""Montar, salvar e carregar o índice (pedaços + vetores + BM25 + assinatura das notas)."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rag.cortador import Pedaco, cortar_nota
from rag.embeddings import Codificador
from rag.palavra import BM25, palavras

ARQUIVO_PEDACOS = "pedacos.json"
ARQUIVO_VETORES = "vetores.npy"
ARQUIVO_MODELO = "modelo.txt"
ARQUIVO_ASSINATURAS = "assinaturas.json"
ARQUIVO_VAULT = "vault.txt"


@dataclass(frozen=True)
class Assinatura:
    """Impressão digital de uma nota: serve para saber se ela mudou desde o índice.

    O carimbo (data de modificação + tamanho) evita reler a nota quando nada mudou;
    o hash do texto evita refazer a nota quando só o carimbo mudou (a sincronização do
    OneDrive às vezes "toca" um arquivo sem mudar nada).
    """

    hash: str
    mtime_ns: int
    tamanho: int


@dataclass
class Indice:
    pedacos: list[Pedaco]
    vetores: np.ndarray
    bm25: BM25
    modelo: str
    assinaturas: dict[str, Assinatura] = field(default_factory=dict)
    vault: str = ""  # pasta de onde as notas foram lidas (para atualizar sozinho)


def hash_do_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def assinar(arquivo: Path, texto: str) -> Assinatura:
    estatistica = arquivo.stat()
    return Assinatura(
        hash_do_texto(texto), estatistica.st_mtime_ns, estatistica.st_size
    )


def listar_notas(pasta_vault: Path) -> dict[str, Path]:
    """Caminho relativo (com "/") -> arquivo, de todas as notas .md, em ordem fixa."""
    achadas = {
        arquivo.relative_to(pasta_vault).as_posix(): arquivo
        for arquivo in pasta_vault.rglob("*.md")
    }
    return dict(sorted(achadas.items()))


def ler_notas(pasta_vault: Path) -> dict[str, str]:
    """Caminho relativo (com "/") -> texto, de todas as notas .md da pasta."""
    return {
        caminho: arquivo.read_text("utf-8")
        for caminho, arquivo in listar_notas(pasta_vault).items()
    }


def montar_bm25(pedacos: list[Pedaco]) -> BM25:
    return BM25([palavras(p.texto_indexado) for p in pedacos])


def montar_indice(pasta_vault: Path, codificador: Codificador) -> Indice:
    pedacos: list[Pedaco] = []
    assinaturas: dict[str, Assinatura] = {}
    for caminho, arquivo in listar_notas(pasta_vault).items():
        texto = arquivo.read_text("utf-8")
        assinaturas[caminho] = assinar(arquivo, texto)
        pedacos.extend(cortar_nota(caminho, texto))
    vetores = codificador.pedacos([p.texto_indexado for p in pedacos])
    return Indice(
        pedacos,
        vetores,
        montar_bm25(pedacos),
        codificador.nome,
        assinaturas,
        str(pasta_vault.resolve()),
    )


def salvar(indice: Indice, pasta: Path) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    dados = [
        {"nota": p.nota, "secao": p.secao, "texto": p.texto, "ordem": p.ordem}
        for p in indice.pedacos
    ]
    (pasta / ARQUIVO_PEDACOS).write_text(
        json.dumps(dados, ensure_ascii=False), encoding="utf-8"
    )
    np.save(pasta / ARQUIVO_VETORES, indice.vetores)
    (pasta / ARQUIVO_MODELO).write_text(indice.modelo, encoding="utf-8")
    assinaturas = {nota: asdict(a) for nota, a in indice.assinaturas.items()}
    (pasta / ARQUIVO_ASSINATURAS).write_text(
        json.dumps(assinaturas, ensure_ascii=False), encoding="utf-8"
    )
    (pasta / ARQUIVO_VAULT).write_text(indice.vault, encoding="utf-8")


def _ler_opcional(arquivo: Path) -> str | None:
    return arquivo.read_text("utf-8") if arquivo.exists() else None


def carregar(pasta: Path) -> Indice:
    dados = json.loads((pasta / ARQUIVO_PEDACOS).read_text("utf-8"))
    pedacos = [Pedaco(**d) for d in dados]
    vetores = np.load(pasta / ARQUIVO_VETORES)
    modelo = (pasta / ARQUIVO_MODELO).read_text("utf-8")
    # Índice feito antes da atualização automática não tem assinaturas nem vault:
    # abre normal, e a primeira atualização refaz as notas (ver atualizacao.py).
    bruto = _ler_opcional(pasta / ARQUIVO_ASSINATURAS)
    assinaturas = {
        nota: Assinatura(**a) for nota, a in json.loads(bruto or "{}").items()
    }
    vault = _ler_opcional(pasta / ARQUIVO_VAULT) or ""
    return Indice(pedacos, vetores, montar_bm25(pedacos), modelo, assinaturas, vault)
