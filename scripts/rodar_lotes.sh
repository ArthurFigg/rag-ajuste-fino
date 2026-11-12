#!/bin/sh
# uso: sh scripts/rodar_lotes.sh
# Manda cada lote (dados/lotes/lote-N) ao Gemini pelo agy, em paralelo.
# Se a saída não passar na validação (ou a cota acabar), tenta o modelo seguinte.
AGY="$HOME/AppData/Local/agy/bin/agy.exe"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

rodar_um() {
  cd "$1" || exit 2
  for M in gemini-4-argon-high gemini-3.8-flash-high; do
    "$AGY" -p "$(cat "$RAIZ/dados/pedido.txt")" --model "$M" --mode plan \
      --print-timeout 900s > "saida-$M.txt" 2> "erro-$M.txt"
    if (cd "$RAIZ" && uv run python -m ajuste.gerar validar "$1" "$1/saida-$M.txt") \
      > "validacao-$M.txt" 2>&1; then
      cp "saida-$M.txt" saida-final.txt
      echo "$M" > modelo.txt
      exit 0
    fi
  done
  exit 1
}

for L in "$RAIZ"/dados/lotes/lote-*; do
  [ -d "$L" ] || continue
  [ -f "$L/saida-final.txt" ] && continue
  rodar_um "$L" > "$L/log.txt" 2>&1 &
done
wait
for L in "$RAIZ"/dados/lotes/lote-*; do
  echo "$(basename "$L"): $(cat "$L/modelo.txt" 2>/dev/null || echo FALHOU)"
done
