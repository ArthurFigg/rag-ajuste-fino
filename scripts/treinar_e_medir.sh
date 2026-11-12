#!/bin/sh
# uso: sh scripts/treinar_e_medir.sh [config ...]   (padrão: medio forte suave)
# Treina cada configuração e mede o modelo ajustado na prova real.
# Modelos e checkpoints ficam FORA do OneDrive (pesam mais de 1 GB). Se o treino for
# interrompido, rodar de novo retoma do último checkpoint.
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
MODELOS="${RAG_AJUSTE_MODELOS:-$HOME/AppData/Local/rag-ajuste-fino/modelos}"
cd "$RAIZ" || exit 2
CONFIGS="${*:-medio forte suave}"
for C in $CONFIGS; do
  [ -f "resultados/$C.json" ] && { echo "== $C: já medido, pulando"; continue; }
  echo "== $C: treinando"
  [ -f "$MODELOS/$C/modelo/config.json" ] || uv run python -m ajuste.treinar --config "$C" --saida "$MODELOS/$C" || exit 1
  echo "== $C: medindo"
  uv run python -m ajuste.medir --modelo "$MODELOS/$C/modelo" --salvar "resultados/$C.json"
done
