from ajuste.treinar import CONFIGS, montar_colunas, tem_checkpoint

PARES = [
    {"id": "p1", "pergunta": "Onde vive o dragão?", "pedaco": "[Nota: dragão]\nVive."},
    {"id": "p2", "pergunta": "Quem caça com arco?", "pedaco": "[Nota: elfo]\nCaça."},
]


def test_montar_colunas_poe_o_prefixo_de_pergunta_no_ancora():
    assert montar_colunas(PARES)["anchor"][0] == "query: Onde vive o dragão?"


def test_montar_colunas_poe_o_prefixo_de_pedaco_no_positivo():
    assert montar_colunas(PARES)["positive"][1] == "passage: [Nota: elfo]\nCaça."


def test_montar_colunas_mantem_o_mesmo_numero_de_linhas_nas_duas():
    colunas = montar_colunas(PARES)
    assert len(colunas["anchor"]) == len(colunas["positive"])


def test_configs_sao_tres_e_crescem_em_intensidade():
    taxas = [CONFIGS[n].taxa_aprendizado for n in ("suave", "medio", "forte")]
    assert taxas == sorted(taxas)


def test_tem_checkpoint_falso_para_pasta_que_nao_existe(tmp_path):
    assert not tem_checkpoint(tmp_path / "nada")


def test_tem_checkpoint_falso_para_pasta_vazia(tmp_path):
    assert not tem_checkpoint(tmp_path)


def test_tem_checkpoint_verdadeiro_quando_ha_pasta_checkpoint(tmp_path):
    (tmp_path / "checkpoint-20").mkdir()
    assert tem_checkpoint(tmp_path)


def test_lote_dos_treinos_e_16_para_poupar_memoria():
    assert {c.tamanho_lote for c in CONFIGS.values()} == {16}
