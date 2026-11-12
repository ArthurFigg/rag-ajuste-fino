# rag-ajuste-fino

Ajuste fino do modelo de busca (embeddings) de um RAG sobre as notas de um mundo de RPG de mesa, **medido contra uma prova de perguntas reais**.

## Resultado, em uma frase

Ajustei o `intfloat/multilingual-e5-small` com 1.260 pares sintéticos de pergunta e trecho, em três intensidades. **Nenhuma das três versões foi melhor que o modelo original de forma comprovada** nas 46 perguntas da prova: todas deram empate estatístico. O repositório mostra como cheguei a isso e o que um resultado assim permite (e não permite) concluir.

## O que foi medido

Para cada pergunta da prova, a busca devolve os `k` melhores trechos. A pergunta conta como acerto se **todas** as notas do gabarito aparecem entre eles. A prova tem 46 perguntas com gabarito de nota.

Acertos em 46 perguntas (busca só por vetor, que é a parte que o ajuste muda, e busca mista, que junta vetor com palavras):

| busca / k | original | suave | médio | forte |
|---|---|---|---|---|
| vetor / 5 | 30 | 30 | 32 | 32 |
| vetor / 8 | 32 | 35 | 33 | 35 |
| vetor / 12 | 37 | 36 | 35 | 37 |
| vetor / 20 | 39 | 40 | 43 | 40 |
| mista / 5 | 35 | 34 | 36 | 33 |
| mista / 12 | 41 | 40 | 42 | 41 |

Nas seis comparações feitas, o teste de McNemar exato deu p ≥ 0,125 (a mais próxima de "melhorou" foi `médio` em vetor / 20: 43 contra 39, p = 0,125). As tabelas completas, com as perguntas em que cada modelo acerta sozinho, estão em [`resultados/comparacao.md`](resultados/comparacao.md).

### Como ler isso

- Com 46 perguntas, o teste só enxerga diferenças grandes. Ver 2 a 4 perguntas a mais não distingue ajuste de sorte.
- Isso **não prova** que o ajuste não ajuda. Prova que, com estes dados e esta prova, não consegui mostrar que ajuda.
- Os modelos ajustados também não pioraram de forma comprovada.

## Como foi feito

1. **Ponto de partida** (`ajuste/medicao.py`): mede o modelo original na prova, com a mesma régua usada no RAG.
2. **Dados** (`ajuste/dados.py`, `ajuste/gerar.py`): as 72 notas viram 362 trechos. O Gemini (`gemini-3.8-flash-high`, pelo Antigravity CLI) escreveu 4 perguntas por trecho; filtrei as que apontam para "o texto", repetidas e fora do tamanho. Sobraram 1.400 pares: 1.260 de treino e 140 reservados para validação (não usados nesta decisão). **Nenhuma pergunta da prova entrou no treino.**
3. **Treino** (`ajuste/treinar.py`): `MultipleNegativesRankingLoss` do `sentence-transformers`, lote 16, em CPU.

   | configuração | taxa de aprendizado | épocas | perda final |
   |---|---|---|---|
   | suave | 1e-5 | 1 | 0,919 |
   | médio | 2e-5 | 2 | 0,496 |
   | forte | 5e-5 | 3 | 0,310 |

4. **Comparação** (`ajuste/comparar.py`): pergunta a pergunta, com intervalo de Wilson e teste de McNemar exato.

## Limites

- **Só serve a este mundo de RPG.** O modelo foi ajustado em trechos desse conjunto de notas; não é um modelo geral.
- **Prova pequena** (46 perguntas), do mesmo conjunto de notas.
- **Perguntas de treino sintéticas**, escritas por um modelo de linguagem; podem ter estilo diferente das perguntas de uma pessoa.
- **Uma rodada de treino por configuração**, sem repetir com outras sementes.
- Lote 16 (e não 32) porque o treino com 32 estourou a memória da máquina usada (CPU, 16 GB).

## Como reproduzir

Requer [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -v                                   # 48 testes
uv run python -m ajuste.medir --modelo intfloat/multilingual-e5-small --salvar resultados/base.json
uv run python -m ajuste.gerar preparar             # corta as notas em lotes
sh scripts/rodar_lotes.sh                          # pede as perguntas ao Gemini (agy)
uv run python -m ajuste.gerar juntar               # monta os pares
sh scripts/treinar_e_medir.sh medio                # treina e mede (retoma de checkpoint)
uv run python -m ajuste.comparar resultados/base.json resultados/medio.json
```

Os pares já gerados estão em `dados/`, então dá para pular o Gemini e ir direto ao treino.

## Origem dos dados

As notas em `notas/` são de um mundo de RPG de mesa pessoal (ficção). Estão aqui para qualquer pessoa poder reproduzir a medição. Os módulos em `rag/` e `avaliacao/` vêm de um RAG anterior do mesmo autor, em que a prova foi montada.

## Como foi construído

Código escrito com apoio do Claude Code, seguindo o fluxo de [claude-dev-flow](https://github.com/ArthurFigg/claude-dev-flow): testes primeiro, conferência por outro ator, resultado medido antes de afirmar. As perguntas de treino foram geradas pelo Gemini.

## Licença

MIT. O modelo base `intfloat/multilingual-e5-small` também é MIT.
