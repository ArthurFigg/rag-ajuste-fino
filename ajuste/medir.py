"""Mede um modelo (nome do Hugging Face ou pasta) e salva o resultado em JSON.

Uso: uv run python -m ajuste.medir --modelo intfloat/multilingual-e5-small \
        --salvar resultados/base.json
"""

import argparse
import json
from pathlib import Path

from ajuste.ambiente import pasta_notas
from ajuste.codificador import CodificadorE5
from ajuste.medicao import carregar_prova, medir_modelo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--salvar", required=True)
    args = parser.parse_args()

    resultado = medir_modelo(
        CodificadorE5(args.modelo), pasta_notas(), carregar_prova()
    )
    Path(args.salvar).write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"modelo={resultado['modelo']} pedacos={resultado['pedacos']}")
    for chave, r in resultado["resumo"].items():
        print(
            f"{chave:10} perguntas={r['perguntas']} "
            f"todas_notas={r['todas_as_notas']} recall_notas={r['recall_notas']}"
        )


if __name__ == "__main__":
    main()
