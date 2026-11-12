import pytest

from ajuste.dados import (
    dividir_em_lotes,
    exportar_pedacos,
    filtrar_perguntas,
    ler_saida_do_gemini,
    montar_pares,
    pergunta_valida,
    separar_treino_validacao,
)


def test_exportar_pedacos_da_um_id_unico_a_cada_pedaco(pasta_notas_pequena):
    pedacos = exportar_pedacos(pasta_notas_pequena)
    assert len({p["id"] for p in pedacos}) == len(pedacos)


def test_exportar_pedacos_guarda_a_nota_de_origem(pasta_notas_pequena):
    pedacos = exportar_pedacos(pasta_notas_pequena)
    assert {p["nota"] for p in pedacos} == {"bestas/dragao.md", "bestas/elfo.md"}


def test_dividir_em_lotes_nao_perde_nenhum_item():
    itens = [{"id": i} for i in range(10)]
    lotes = dividir_em_lotes(itens, 3)
    assert sum(len(lote) for lote in lotes) == 10


def test_dividir_em_lotes_mantem_a_ordem():
    itens = [{"id": i} for i in range(7)]
    lotes = dividir_em_lotes(itens, 2)
    assert [x["id"] for lote in lotes for x in lote] == list(range(7))


def test_ler_saida_ignora_texto_solto_em_volta():
    saida = 'Aqui está:\n[{"id": "p0001", "perguntas": ["Quem é o rei?"]}]\nFim.'
    assert ler_saida_do_gemini(saida) == {"p0001": ["Quem é o rei?"]}


def test_ler_saida_aceita_quebra_de_linha_solta_dentro_da_frase():
    saida = '[{"id": "p0001", "perguntas": ["Quem é\no rei do norte?"]}]'
    assert ler_saida_do_gemini(saida) == {"p0001": ["Quem é\no rei do norte?"]}


def test_ler_saida_sem_lista_da_erro():
    with pytest.raises(ValueError):
        ler_saida_do_gemini("não consegui")


def test_pergunta_valida_aceita_pergunta_normal():
    assert pergunta_valida("Onde vive o dragão vermelho da montanha?")


def test_pergunta_valida_recusa_a_que_aponta_para_o_texto():
    assert not pergunta_valida("O que o texto diz sobre o dragão vermelho?")


def test_pergunta_valida_recusa_a_que_nao_termina_em_interrogacao():
    assert not pergunta_valida("Fale sobre o dragão vermelho da montanha")


def test_pergunta_valida_recusa_a_curta_demais():
    assert not pergunta_valida("Quem é ele?")


def test_filtrar_perguntas_tira_repetida_ignorando_acento_e_caixa():
    entrada = ["Onde vive o dragão vermelho?", "ONDE VIVE O DRAGAO VERMELHO?"]
    assert filtrar_perguntas(entrada) == ["Onde vive o dragão vermelho?"]


def test_filtrar_perguntas_troca_quebra_de_linha_por_espaco():
    assert filtrar_perguntas(["Onde vive\no dragão vermelho?"]) == [
        "Onde vive o dragão vermelho?"
    ]


def test_filtrar_perguntas_respeita_o_maximo_por_pedaco():
    entrada = [f"Qual é a cor do dragão número {n} da montanha?" for n in range(9)]
    assert len(filtrar_perguntas(entrada)) == 5


def test_montar_pares_liga_pergunta_ao_texto_do_pedaco():
    pedacos = [{"id": "p0001", "nota": "a.md", "texto": "[Nota: a]\nO dragão vive."}]
    pares = montar_pares(pedacos, {"p0001": ["Onde vive o dragão da montanha?"]})
    assert pares[0]["pedaco"] == "[Nota: a]\nO dragão vive."


def test_montar_pares_ignora_pedaco_sem_pergunta():
    pedacos = [{"id": "p0001", "nota": "a.md", "texto": "texto"}]
    assert montar_pares(pedacos, {}) == []


def test_separar_treino_validacao_nao_repete_pergunta():
    pares = [{"id": "p", "pergunta": f"q{i}", "pedaco": "t"} for i in range(50)]
    treino, validacao = separar_treino_validacao(pares, 0.2)
    assert not {p["pergunta"] for p in treino} & {p["pergunta"] for p in validacao}


def test_separar_treino_validacao_e_repetivel_com_a_mesma_semente():
    pares = [{"id": "p", "pergunta": f"q{i}", "pedaco": "t"} for i in range(50)]
    assert separar_treino_validacao(pares, 0.2) == separar_treino_validacao(pares, 0.2)
