"""Link que abre a nota direto no Obsidian.

Formato oficial do Obsidian: obsidian://open?vault=<nome>&file=<caminho sem .md>.
Ainda NÃO foi testado no Obsidian do dono (o nome do cofre é a pasta "Obsidian Vault").
"""

from urllib.parse import quote

COFRE_PADRAO = "Obsidian Vault"
PASTA_DO_RPG = "RPG - Novo Antigo Continente"


def link_obsidian(
    nota: str, cofre: str = COFRE_PADRAO, pasta_base: str = PASTA_DO_RPG
) -> str:
    """`nota` é o caminho relativo dentro da pasta do RPG, com "/" e com ".md"."""
    sem_extensao = nota.removesuffix(".md")
    caminho = f"{pasta_base}/{sem_extensao}" if pasta_base else sem_extensao
    return (
        f"obsidian://open?vault={quote(cofre, safe='')}&file={quote(caminho, safe='')}"
    )
