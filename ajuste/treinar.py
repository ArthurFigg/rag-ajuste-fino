"""Ajuste fino do multilingual-e5-small com pares (pergunta, pedaço).

A ideia: o modelo aprende que a pergunta e o pedaço que a responde devem ficar
perto no espaço dos vetores. Os outros pedaços do mesmo lote servem de exemplo
negativo (MultipleNegativesRankingLoss), então o lote não pode ter o mesmo pedaço
duas vezes, senão o "negativo" seria na verdade o certo (NO_DUPLICATES).

Uso: uv run python -m ajuste.treinar --config medio --saida modelos/medio
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from ajuste.ambiente import RAIZ_PROJETO
from ajuste.codificador import PREFIXO_PEDACO, PREFIXO_PERGUNTA

MODELO_BASE = "intfloat/multilingual-e5-small"


@dataclass(frozen=True)
class Config:
    taxa_aprendizado: float
    epocas: int
    tamanho_lote: int


# Lote de 16 (e não 32): o treino de 11/11/2026 estourou a memória do PC com 32.
CONFIGS = {
    "suave": Config(taxa_aprendizado=1e-5, epocas=1, tamanho_lote=16),
    "medio": Config(taxa_aprendizado=2e-5, epocas=2, tamanho_lote=16),
    "forte": Config(taxa_aprendizado=5e-5, epocas=3, tamanho_lote=16),
}
PASSOS_ENTRE_CHECKPOINTS = 20


def tem_checkpoint(pasta: Path) -> bool:
    """Já existe um checkpoint salvo nesta pasta (dá para retomar o treino)?"""
    return pasta.is_dir() and any(pasta.glob("checkpoint-*"))


def montar_colunas(pares: list[dict]) -> dict[str, list[str]]:
    """Colunas no formato do treinador, com os prefixos que o e5 espera."""
    return {
        "anchor": [PREFIXO_PERGUNTA + p["pergunta"] for p in pares],
        "positive": [PREFIXO_PEDACO + p["pedaco"] for p in pares],
    }


def treinar(config: Config, pares: list[dict], saida: Path) -> dict:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    modelo = SentenceTransformer(MODELO_BASE)
    dataset = Dataset.from_dict(montar_colunas(pares))
    argumentos = SentenceTransformerTrainingArguments(
        output_dir=str(saida / "checkpoints"),
        num_train_epochs=config.epocas,
        per_device_train_batch_size=config.tamanho_lote,
        learning_rate=config.taxa_aprendizado,
        warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=10,
        save_strategy="steps",
        save_steps=PASSOS_ENTRE_CHECKPOINTS,
        save_total_limit=1,
        report_to="none",
        seed=42,
    )
    treinador = SentenceTransformerTrainer(
        model=modelo,
        args=argumentos,
        train_dataset=dataset,
        loss=losses.MultipleNegativesRankingLoss(modelo),
    )
    retomar = tem_checkpoint(saida / "checkpoints")
    resultado = treinador.train(resume_from_checkpoint=retomar or None)
    modelo.save(str(saida / "modelo"))
    historico = [h for h in treinador.state.log_history if "loss" in h]
    return {"perda_final": resultado.training_loss, "historico": historico}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    pares = json.loads((RAIZ_PROJETO / "dados" / "pares-treino.json").read_text("utf-8"))
    registro = treinar(CONFIGS[args.config], pares, args.saida)
    registro["config"] = {"nome": args.config, **vars(CONFIGS[args.config])}
    registro["pares_treino"] = len(pares)
    (args.saida / "registro.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{args.config}: perda final {registro['perda_final']:.4f}")


if __name__ == "__main__":
    main()
