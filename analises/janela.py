"""Features de janela temporal (contagens no passado), calculadas sobre o CSV completo dentro do ZIP.

DoS-Heavy e DoS-Light têm fluxos idênticos (mesmo atacante, vítima, porta e 1 pacote); a diferença é só de
volume. Uma amostra de 150 fluxos por estrato não mede taxa, então as contagens saem do arquivo inteiro.

Para cada fluxo, com t = bidirectional_first_seen_ms e W em 1, 10 e 60 s, conta no intervalo (t − W, t]
(o próprio fluxo incluído; só passado, sem rótulo, como um IDS faria):
  jan{W}s_src_fluxos   fluxos com o mesmo src_ip
  jan{W}s_par_fluxos   fluxos com o mesmo (src_ip, dst_ip)
  jan{W}s_dst_fluxos   fluxos com o mesmo dst_ip
  jan{W}s_src_pacotes  soma de bidirectional_packets com o mesmo src_ip

Saída: só as linhas que estão na amostra (por linha_csv, a mesma numeração de amostrar.py).

Uso: python -I janela.py ZIP MEMBRO AMOSTRA.csv SAIDA.csv
"""
import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd

T, PK = 'bidirectional_first_seen_ms', 'bidirectional_packets'
JANELAS_S = [1, 10, 60]
CHAVES = {'src': ['src_ip'], 'par': ['src_ip', 'dst_ip'], 'dst': ['dst_ip']}


def contar(t, chave, pesos, w_ms):
    """Para cada linha, soma de pesos das linhas com a mesma chave e tempo em (t − w, t]."""
    t = t - t.min()
    ordem = np.lexsort((t, chave))
    v = chave[ordem].astype(np.int64) * (int(t.max()) + w_ms + 1) + t[ordem]
    acum = np.concatenate([[0], np.cumsum(pesos[ordem])])
    fim = np.searchsorted(v, v, side='right')        # inclui empates no mesmo ms
    ini = np.searchsorted(v, v - w_ms, side='right')
    out = np.empty(len(t), dtype=np.int64)
    out[ordem] = acum[fim] - acum[ini]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('zip'); p.add_argument('membro'); p.add_argument('amostra'); p.add_argument('saida')
    a = p.parse_args()
    linhas = pd.read_csv(a.amostra, usecols=['linha_csv'])['linha_csv'].to_numpy()

    partes = []
    with ZipFile(a.zip) as z, z.open(a.membro) as src:
        for chunk in pd.read_csv(src, chunksize=1_000_000, usecols=['src_ip', 'dst_ip', T, PK],
                                 dtype={'src_ip': 'category', 'dst_ip': 'category', T: 'int64', PK: 'int64'}):
            partes.append(chunk)
    df = pd.concat(partes, ignore_index=True)   # as categorias são unificadas como texto aqui
    n = len(df)
    t = df[T].to_numpy()
    pk = df[PK].to_numpy()
    um = np.ones(n, dtype=np.int64)
    cod = {nome: df.groupby(cols, sort=False, observed=True).ngroup().to_numpy() for nome, cols in CHAVES.items()}

    alvo = linhas - 1  # linha_csv começa em 1
    out = {'linha_csv': linhas}
    for w in JANELAS_S:
        for nome in CHAVES:
            out[f'jan{w}s_{nome}_fluxos'] = contar(t, cod[nome], um, w * 1000)[alvo]
        out[f'jan{w}s_src_pacotes'] = contar(t, cod['src'], pk, w * 1000)[alvo]
    pd.DataFrame(out).to_csv(a.saida, index=False)
    print(f'{a.saida}: {n:,} fluxos no CSV, {len(linhas):,} linhas da amostra')


if __name__ == '__main__':
    main()
