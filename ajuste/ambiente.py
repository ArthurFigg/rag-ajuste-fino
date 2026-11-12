"""Onde estão as notas do RPG, sem caminho fixo no código.

Por padrão são as notas copiadas para `notas/` neste repositório; para medir sobre
outra pasta, defina a variável `NOTAS_PASTA`.
"""

import os
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent


def pasta_notas() -> Path:
    return Path(os.environ.get("NOTAS_PASTA", RAIZ_PROJETO / "notas"))
