from ajuste.medicao import medir_modelo

PERGUNTAS = [
    {
        "id": 1,
        "grupo": "final/lista",
        "pergunta": "onde vive o dragão vermelho que guarda ouro?",
        "notas": ["bestas/dragao.md"],
        "trechos": [],
    },
    {
        "id": 2,
        "grupo": "final/lista",
        "pergunta": "quem caça com arco e fala com as árvores da floresta?",
        "notas": ["bestas/elfo.md"],
        "trechos": [],
    },
]


def test_medir_modelo_acha_as_notas_certas_por_vetor(
    codificador_falso, pasta_notas_pequena
):
    resultado = medir_modelo(
        codificador_falso, pasta_notas_pequena, PERGUNTAS, ks=(1,)
    )
    assert resultado["resumo"]["vetor/1"]["recall_notas"] == 1.0


def test_medir_modelo_conta_as_perguntas(codificador_falso, pasta_notas_pequena):
    resultado = medir_modelo(
        codificador_falso, pasta_notas_pequena, PERGUNTAS, ks=(1,)
    )
    assert resultado["perguntas"] == 2


def test_medir_modelo_guarda_uma_linha_por_pergunta(
    codificador_falso, pasta_notas_pequena
):
    resultado = medir_modelo(
        codificador_falso, pasta_notas_pequena, PERGUNTAS, ks=(1,)
    )
    assert len(resultado["linhas"]["vetor/1"]) == 2


def test_medir_modelo_mede_tambem_a_busca_mista(
    codificador_falso, pasta_notas_pequena
):
    resultado = medir_modelo(
        codificador_falso, pasta_notas_pequena, PERGUNTAS, ks=(1,)
    )
    assert "mista/1" in resultado["resumo"]


def test_medir_modelo_nota_fora_do_alcance_nao_conta(
    codificador_falso, pasta_notas_pequena
):
    perguntas = [{**PERGUNTAS[0], "notas": ["bestas/elfo.md"]}]
    resultado = medir_modelo(codificador_falso, pasta_notas_pequena, perguntas, ks=(1,))
    assert resultado["resumo"]["vetor/1"]["recall_notas"] == 0.0
