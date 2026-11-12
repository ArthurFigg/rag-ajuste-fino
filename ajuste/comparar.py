"""Compara dois resultados da prova (antes x depois) com margem de erro.

Com ~46 perguntas, uma diferença pequena pode ser sorte. Por isso, além da taxa de
acerto, o relatório traz o intervalo de Wilson de cada lado e o teste de McNemar
exato, que olha só as perguntas em que os dois modelos discordam.

Uso: uv run python -m ajuste.comparar resultados/base.json resultados/medio.json
"""

import argparse
import json
import math
from pathlib import Path

CHAVES = ("vetor/5", "vetor/8", "vetor/12", "vetor/20", "mista/5", "mista/12")
Z_95 = 1.96


def intervalo_wilson(acertos: int, total: int) -> tuple[float, float]:
    """Intervalo de 95% para uma taxa, melhor que o simples quando n é pequeno."""
    if total == 0:
        return (0.0, 0.0)
    p = acertos / total
    z2 = Z_95**2
    centro = (p + z2 / (2 * total)) / (1 + z2 / total)
    meia = (
        Z_95
        * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2))
        / (1 + z2 / total)
    )
    return (max(0.0, centro - meia), min(1.0, centro + meia))


def p_mcnemar_exato(so_a: int, so_b: int) -> float:
    """Chance de ver uma divisão tão desigual se os dois modelos fossem iguais."""
    discordancias = so_a + so_b
    if discordancias == 0:
        return 1.0
    menor = min(so_a, so_b)
    cauda = sum(math.comb(discordancias, i) for i in range(menor + 1))
    return min(1.0, 2 * cauda / 2**discordancias)


def acertou_tudo(linha: dict) -> bool:
    return linha["notas_total"] > 0 and linha["notas_achadas"] == linha["notas_total"]


def comparar_chave(linhas_base: list[dict], linhas_novo: list[dict]) -> dict:
    """Compara pergunta a pergunta (as duas listas vêm na mesma ordem)."""
    pares = [
        (acertou_tudo(a), acertou_tudo(b))
        for a, b in zip(linhas_base, linhas_novo, strict=True)
        if a["notas_total"] > 0
    ]
    total = len(pares)
    acertos_base = sum(a for a, _ in pares)
    acertos_novo = sum(b for _, b in pares)
    so_base = sum(a and not b for a, b in pares)
    so_novo = sum(b and not a for a, b in pares)
    return {
        "perguntas": total,
        "acertos_base": acertos_base,
        "acertos_novo": acertos_novo,
        "wilson_base": intervalo_wilson(acertos_base, total),
        "wilson_novo": intervalo_wilson(acertos_novo, total),
        "so_base": so_base,
        "so_novo": so_novo,
        "p": p_mcnemar_exato(so_base, so_novo),
    }


def veredito(comparacao: dict, nivel: float = 0.05) -> str:
    if comparacao["p"] >= nivel:
        return "empate (diferença pode ser sorte)"
    return "melhorou" if comparacao["so_novo"] > comparacao["so_base"] else "piorou"


def nome_curto(modelo: str) -> str:
    """`.../modelos/medio/modelo` vira `medio`; nome do Hugging Face fica como está."""
    caminho = Path(modelo)
    return caminho.parent.name if caminho.name == "modelo" else modelo


def tabela_markdown(base: dict, novo: dict) -> str:
    linhas = [
        f"| busca/k | {nome_curto(base['modelo'])} | {nome_curto(novo['modelo'])} | "
        "só o antigo acerta | só o novo acerta | p (McNemar) | veredito |",
        "|---|---|---|---|---|---|---|",
    ]
    for chave in CHAVES:
        c = comparar_chave(base["linhas"][chave], novo["linhas"][chave])
        n = c["perguntas"]
        linhas.append(
            f"| {chave} | {c['acertos_base']}/{n} | {c['acertos_novo']}/{n} | "
            f"{c['so_base']} | {c['so_novo']} | {c['p']:.3f} | {veredito(c)} |"
        )
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("novo", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base.read_text("utf-8"))
    novo = json.loads(args.novo.read_text("utf-8"))
    print(tabela_markdown(base, novo))


if __name__ == "__main__":
    main()
