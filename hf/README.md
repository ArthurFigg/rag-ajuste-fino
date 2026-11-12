---
language:
- pt
license: mit
library_name: sentence-transformers
base_model: intfloat/multilingual-e5-small
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- embeddings
- rag
- portuguese
- fine-tuned
---

# e5-small ajustado para um RAG de RPG (resultado: empate com o original)

Ajuste fino do [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) com 1.260 pares sintéticos de pergunta e trecho, tirados de um conjunto de notas de um mundo de RPG de mesa.

**Resultado honesto: este modelo não é melhor que o original de forma comprovada.** Nas 46 perguntas da prova, a diferença para o modelo base ficou dentro do que pode ser sorte (teste de McNemar exato, p ≥ 0,125 nas seis comparações feitas). Publico para mostrar o processo e a medição, não para substituir o modelo base.

## Uso

O modelo espera os mesmos prefixos do e5: `query: ` na pergunta e `passage: ` no trecho.

```python
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("USUARIO/e5-small-rpg-ajuste")
pergunta = modelo.encode("query: Quem lidera a Guarda Real?", normalize_embeddings=True)
trechos = modelo.encode(["passage: A Guarda Real é comandada por ..."], normalize_embeddings=True)
```

## Treino

- Base: `intfloat/multilingual-e5-small`.
- Dados: 1.260 pares (pergunta, trecho) de 362 trechos de 72 notas. As perguntas foram escritas pelo Gemini; nenhuma pergunta da prova entrou no treino.
- Configuração `médio`: `MultipleNegativesRankingLoss`, taxa de aprendizado 2e-5, 2 épocas, lote 16, CPU. Perda final 0,496.
- Também treinei `suave` (1e-5, 1 época) e `forte` (5e-5, 3 épocas); nenhuma foi melhor que o original de forma comprovada.

## Resultado (acertos em 46 perguntas)

| busca / k | original | este modelo |
|---|---|---|
| vetor / 5 | 30 | 32 |
| vetor / 12 | 37 | 35 |
| vetor / 20 | 39 | 43 |
| mista / 5 | 35 | 36 |

## Limites

- Ajustado em um único conjunto de notas de RPG; não é um modelo geral.
- Prova pequena (46 perguntas): só detectaria diferenças grandes.
- Uma rodada de treino por configuração.

Código, dados e medições: https://github.com/ArthurFigg/rag-ajuste-fino
