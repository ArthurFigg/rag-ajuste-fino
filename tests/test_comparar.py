import pytest

from ajuste.comparar import (
    acertou_tudo,
    comparar_chave,
    intervalo_wilson,
    nome_curto,
    p_mcnemar_exato,
    veredito,
)


def test_nome_curto_usa_a_pasta_da_configuracao():
    assert nome_curto("C:/x/modelos/medio/modelo") == "medio"


def test_nome_curto_mantem_nome_do_hugging_face():
    assert nome_curto("intfloat/multilingual-e5-small") == "intfloat/multilingual-e5-small"


def linha(achadas: int, total: int = 1) -> dict:
    return {"notas_total": total, "notas_achadas": achadas}


def test_p_mcnemar_sem_discordancia_e_um():
    assert p_mcnemar_exato(0, 0) == 1.0


def test_p_mcnemar_cinco_a_zero_vale_um_trinta_e_dois_dobrado():
    assert p_mcnemar_exato(0, 5) == pytest.approx(0.0625)


def test_p_mcnemar_e_simetrico():
    assert p_mcnemar_exato(2, 7) == p_mcnemar_exato(7, 2)


def test_p_mcnemar_empate_exato_nao_passa_de_um():
    assert p_mcnemar_exato(4, 4) == 1.0


def test_wilson_contem_a_taxa_observada():
    baixo, alto = intervalo_wilson(30, 46)
    assert baixo < 30 / 46 < alto


def test_wilson_com_todos_os_acertos_chega_a_um():
    assert intervalo_wilson(10, 10)[1] == 1.0


def test_wilson_sem_total_devolve_zero():
    assert intervalo_wilson(0, 0) == (0.0, 0.0)


def test_acertou_tudo_exige_todas_as_notas():
    assert not acertou_tudo(linha(1, total=2))


def test_acertou_tudo_ignora_pergunta_sem_nota():
    assert not acertou_tudo(linha(0, total=0))


def test_comparar_chave_conta_quem_so_o_novo_acerta():
    base = [linha(0), linha(1), linha(1)]
    novo = [linha(1), linha(1), linha(0)]
    assert comparar_chave(base, novo)["so_novo"] == 1


def test_comparar_chave_conta_quem_so_o_antigo_acerta():
    base = [linha(0), linha(1), linha(1)]
    novo = [linha(1), linha(1), linha(0)]
    assert comparar_chave(base, novo)["so_base"] == 1


def test_comparar_chave_nao_conta_pergunta_sem_nota():
    base = [linha(0, total=0), linha(1)]
    novo = [linha(0, total=0), linha(1)]
    assert comparar_chave(base, novo)["perguntas"] == 1


def test_veredito_diferenca_pequena_e_empate():
    comparacao = {"p": 0.4, "so_novo": 3, "so_base": 1}
    assert veredito(comparacao).startswith("empate")


def test_veredito_diferenca_grande_a_favor_do_novo_e_melhorou():
    comparacao = {"p": 0.001, "so_novo": 12, "so_base": 1}
    assert veredito(comparacao) == "melhorou"


def test_veredito_diferenca_grande_contra_o_novo_e_piorou():
    comparacao = {"p": 0.001, "so_novo": 1, "so_base": 12}
    assert veredito(comparacao) == "piorou"
