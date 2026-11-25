"""Sorteia amostras de um CSV do dataset XRCE-DDS/Zenoh (lido em blocos de dentro do ZIP).

Dois modos:
  blocos  Regra do notebook "XAIID e TRAP IoT": 12 blocos de 3 h a partir de --ancora (ms), exclui fluxos
          a menos de 4 min das fronteiras e sorteia até --cap fluxos por (bloco, classe). Com a semente
          20261117 e chunks de 150 mil linhas, reproduz o sorteio do notebook.
  classe  Para o NetSecBed (D1 e D2 com cronogramas diferentes): sorteia até --cap fluxos por classe,
          sem blocos.

Faz várias sementes numa só leitura: cada semente tem o seu gerador e a sua coluna de prioridade, e
a saída marca em colunas s_<semente> quais linhas entram na amostra daquela semente.

Uso:
  python -I amostrar.py ZIP MEMBRO CENARIO SAIDA.csv --modo blocos --ancora 1793576700000
  python -I amostrar.py ZIP MEMBRO CENARIO SAIDA.csv --modo classe --cap 1000
"""
import argparse
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

START, END, TARGET = 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'Attack_Type'
FEATURE = re.compile(
    r'^(bidirectional|src2dst|dst2src)_(duration_ms|packets|bytes|'
    r'(min|mean|stddev|max)_(ps|piat_ms)|'
    r'(syn|cwr|ece|urg|ack|psh|rst|fin)_packets)$'
)
EXTRA = {'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'Protocol', 'Scenario'}
BLOCK_MS = 3 * 60 * 60 * 1000
PURGE_MS = 4 * 60 * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('zip'); p.add_argument('membro'); p.add_argument('cenario'); p.add_argument('saida')
    p.add_argument('--modo', choices=['blocos', 'classe'], required=True)
    p.add_argument('--ancora', type=int)
    p.add_argument('--cap', type=int, default=150)
    p.add_argument('--sementes', default='20261117,20261118,20261119,20261120,20261121')
    a = p.parse_args()
    seeds = [int(s) for s in a.sementes.split(',')]
    if a.modo == 'blocos' and a.ancora is None:
        sys.exit('--ancora é obrigatória no modo blocos')

    rngs = {s: np.random.default_rng(s) for s in seeds}
    keep, audit, offset = [], [], 0
    group = ['bloco', TARGET] if a.modo == 'blocos' else [TARGET]
    with ZipFile(a.zip) as z, z.open(a.membro) as src:
        chunks = pd.read_csv(src, chunksize=150_000,
                             usecols=lambda c: c in {START, END, TARGET} | EXTRA or bool(FEATURE.fullmatch(c)),
                             dtype={START: 'int64', END: 'int64'}, low_memory=False)
        for frame in chunks:
            frame['linha_csv'] = np.arange(offset + 1, offset + len(frame) + 1)
            offset += len(frame)
            # Gera as prioridades de todas as sementes antes de qualquer filtro (igual ao notebook).
            for s in seeds:
                frame[f'p_{s}'] = rngs[s].random(len(frame))
            if a.modo == 'blocos':
                start = frame[START].to_numpy() - a.ancora
                end = frame[END].to_numpy() - a.ancora
                frame['bloco'] = start // BLOCK_MS + 1
                valid = ((start >= 0) & (end >= start) & (end < 12 * BLOCK_MS)
                         & (start // BLOCK_MS == end // BLOCK_MS)
                         & (start % BLOCK_MS >= PURGE_MS) & (end % BLOCK_MS < BLOCK_MS - PURGE_MS))
            else:
                frame['bloco'] = 0
                valid = np.ones(len(frame), dtype=bool)
            audit.append(frame.assign(incluido=valid).groupby(['bloco', TARGET, 'incluido']).size())
            frame = frame.loc[valid]
            # Guarda as linhas que estão no top-cap de alguma semente dentro deste chunk.
            mask = np.zeros(len(frame), dtype=bool)
            for s in seeds:
                top = frame.sort_values(f'p_{s}').groupby(group, sort=False).head(a.cap).index
                mask |= frame.index.isin(top)
            keep.append(frame.loc[mask])

    pool = pd.concat(keep, ignore_index=True)
    for s in seeds:
        chosen = pool.sort_values(f'p_{s}').groupby(group, sort=False).head(a.cap).index
        pool[f's_{s}'] = pool.index.isin(chosen)
    pool = pool[pool[[f's_{s}' for s in seeds]].any(axis=1)].drop(columns=[f'p_{s}' for s in seeds])
    pool['cenario'] = a.cenario
    pool.sort_values(['bloco', 'linha_csv']).to_csv(a.saida, index=False)
    counts = pd.concat(audit).groupby(level=[0, 1, 2]).sum()
    counts.rename('fluxos').reset_index().to_csv(a.saida.replace('.csv', '_cobertura.csv'), index=False)
    print(f'{a.saida}: {offset:,} linhas lidas; amostra da 1ª semente:')
    print(pool[pool[f's_{seeds[0]}']].groupby(TARGET).size().to_string())


if __name__ == '__main__':
    main()
