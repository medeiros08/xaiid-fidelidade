"""Compara o top 10 da informação mútua entre cenários, por classe, com várias sementes.

Para cada semente:
  1. pega a amostra daquela semente em cada cenário (colunas s_<semente> geradas por amostrar.py);
     - modo v2: só os blocos de treino do notebook (1, 3, 5, 6, 7, 8, 10);
     - modo nsb (NetSecBed): sorteia 80% de cada classe com a semente, para a amostra variar até nas
       classes pequenas (que entram inteiras no sorteio);
  2. calcula a MI um-contra-resto de cada classe (mediana do próprio cenário para os vazios);
  3. compara o top 10 (só features com MI > 0) entre cada par de cenários com Jaccard e Spearman, em três
     conjuntos de features: todas (57), sem tamanho (sem *_bytes e *_ps) e sem tamanho nem flags TCP.

Também calcula o Jaccard esperado por acaso, a partir de quantas features têm MI > 0.

Uso:
  python -I analisar.py SAIDA_DIR --modo v2 --protocolo xrce amostra_D1.csv amostra_D2.csv amostra_D3.csv
"""
import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

TARGET = 'Attack_Type'
TREINO = [1, 3, 5, 6, 7, 8, 10]
FEATURE = re.compile(
    r'^(bidirectional|src2dst|dst2src)_(duration_ms|packets|bytes|'
    r'(min|mean|stddev|max)_(ps|piat_ms)|'
    r'(syn|cwr|ece|urg|ack|psh|rst|fin)_packets)$'
)
FLAG = re.compile(r'_(syn|cwr|ece|urg|ack|psh|rst|fin)_packets$')


def conjuntos(features, payload=None):
    tamanho = [f for f in features if f.endswith('_bytes') or f.endswith('_ps')]
    flags = [f for f in features if FLAG.search(f)]
    sets = {
        'todas': features,
        'sem_tamanho': [f for f in features if f not in tamanho],
        'sem_tamanho_flags': [f for f in features if f not in tamanho and f not in flags],
    }
    if payload:
        sets['payload'] = sets['sem_tamanho'] + payload
        sets['payload_sem_flags'] = sets['sem_tamanho_flags'] + payload
    return sets


# Cabeçalho até a camada 4 (Ethernet 14 + IPv4 20 + L4) e piso comum por protocolo. O piso é o que o padding do
# quadro mínimo de 60 B (só no físico) esconde: 60 − 42 = 18 B em UDP/ICMP e 60 − 54 = 6 B em TCP.
CAB = {17: 42, 1: 42, 6: 54}
PISO = {17: 18, 1: 18, 6: 6}
CAB_OUTROS, PISO_OUTROS = 34, 26
TS_TCP = 12  # opção de timestamp TCP (Linux usa; o lwIP do ESP32, não)


def adicionar_payload(df):
    """Cria 15 features de payload (pl_*) a partir das de tamanho, descontando o cabeçalho por sentido.

    TCP: o cabeçalho é 54 B, ou 66 B se o menor quadro do sentido tiver 66 B ou mais (timestamp TCP; o menor
    quadro costuma ser um ACK puro). A mesma regra vale em todos os cenários. O desvio-padrão não muda com o
    desconto e é mantido. Retorna o DataFrame e a lista de colunas novas.
    """
    proto = df['protocol']
    cab0 = proto.map(CAB).fillna(CAB_OUTROS)
    piso = proto.map(PISO).fillna(PISO_OUTROS)
    novas, n, pl = {}, {}, {}
    for d in ('src2dst', 'dst2src'):
        n[d] = df[f'{d}_packets']
        cab = cab0 + np.where(proto.eq(6) & (df[f'{d}_min_ps'] >= cab0 + TS_TCP), TS_TCP, 0)
        vazio = n[d].eq(0)
        for est in ('min', 'mean', 'max'):
            v = np.maximum(df[f'{d}_{est}_ps'] - cab, piso)
            pl[d, est] = v.where(~vazio, 0)
            novas[f'pl_{d}_{est}_ps'] = pl[d, est]
        novas[f'pl_{d}_stddev_ps'] = df[f'{d}_stddev_ps']
        novas[f'pl_{d}_bytes'] = np.maximum(df[f'{d}_bytes'] - n[d] * cab, n[d] * piso).where(~vazio, 0)
    tem = {d: n[d] > 0 for d in n}
    inf = np.inf
    novas['pl_bidirectional_min_ps'] = np.minimum(pl['src2dst', 'min'].where(tem['src2dst'], inf),
                                                  pl['dst2src', 'min'].where(tem['dst2src'], inf)).replace(inf, 0)
    novas['pl_bidirectional_max_ps'] = np.maximum(pl['src2dst', 'max'], pl['dst2src', 'max'])
    tot = (n['src2dst'] + n['dst2src']).replace(0, np.nan)
    novas['pl_bidirectional_mean_ps'] = ((pl['src2dst', 'mean'] * n['src2dst'] + pl['dst2src', 'mean'] * n['dst2src'])
                                         / tot).fillna(0)
    novas['pl_bidirectional_stddev_ps'] = df['bidirectional_stddev_ps']
    novas['pl_bidirectional_bytes'] = novas['pl_src2dst_bytes'] + novas['pl_dst2src_bytes']
    novas = pd.DataFrame(novas, index=df.index)
    return pd.concat([df, novas], axis=1), list(novas.columns)


def filtrar_nsb(df, protocolo):
    """Regras do LEIA-ME do NetSecBed."""
    df = df[df[TARGET] != 'XRCE-DiscoveryPoison']  # superfícies diferentes em D1 e D2
    if protocolo == 'xrce':
        benign = df[TARGET].eq('Benign')
        par = (df['src_ip'].astype(str).str.endswith(('.103', '.105'))
               | df['dst_ip'].astype(str).str.endswith(('.103', '.105')))
        porta = df['src_port'].eq(8888) | df['dst_port'].eq(8888)
        df = df[~benign | (par & porta & df['protocol'].eq(17))]
    return df


def preparar(df, modo, protocolo, seed):
    df = df[df[f's_{seed}']].copy()
    if modo == 'v2':
        return df[df['bloco'].isin(TREINO)]
    df = filtrar_nsb(df, protocolo)
    # Depois dos filtros: 80% de cada classe, no máximo 1000 fluxos (o sorteio de amostrar.py usa teto maior
    # para o filtro do benigno não ficar sem fluxos).
    rng = np.random.default_rng(seed)
    idx = [g.index[rng.permutation(len(g))[:min(1000, max(1, int(round(0.8 * len(g)))))]]
           for _, g in df.groupby(TARGET)]
    return df.loc[np.concatenate(idx)]


def mi_por_classe(df, features, classes, seed, contra_benign=False):
    x = df[features].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0)
    out = {}
    for c in classes:
        if contra_benign:
            # Só a classe contra o Benign: tira o efeito da mistura de classes, que muda entre cenários.
            if c == 'Benign':
                continue
            rows = df[TARGET].isin([c, 'Benign']).to_numpy()
            xs, y = x[rows], df.loc[rows, TARGET].eq(c).astype(int)
        else:
            xs, y = x, df[TARGET].eq(c).astype(int)
        active = xs.columns[xs.nunique() > 1]
        s = pd.Series(0.0, index=features)
        if y.nunique() == 2 and len(active):
            s.loc[active] = mutual_info_classif(xs[active], y, random_state=seed)
        out[c] = (s, int(y.sum()))
    return out


def jaccard(a, b, k=10):
    ta, tb = (set(s[s > 0].nlargest(k).index) for s in (a, b))
    u = ta | tb
    return len(ta & tb) / len(u) if u else np.nan


def jaccard_acaso(n, k=10):
    if n == 0:
        return np.nan
    k = min(k, n)
    comum = k * k / n
    return comum / (2 * k - comum)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('saida'); p.add_argument('amostras', nargs='+')
    p.add_argument('--modo', choices=['v2', 'nsb'], required=True)
    p.add_argument('--protocolo', choices=['xrce', 'zenoh'], required=True)
    p.add_argument('--sementes', default='20261117,20261118,20261119,20261120,20261121')
    p.add_argument('--contra-benign', action='store_true', help='MI de cada ataque só contra o Benign')
    p.add_argument('--payload', action='store_true',
                   help='acrescenta as features de payload (adicionar_payload) e os conjuntos payload e payload_sem_flags')
    a = p.parse_args()
    seeds = [int(s) for s in a.sementes.split(',')]
    out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.modo}_{a.protocolo}' + ('_vs_benign' if a.contra_benign else '') + ('_payload' if a.payload else '')

    dados, plcols = {}, []
    for path in a.amostras:
        df = pd.read_csv(path, low_memory=False)
        if a.payload:
            df, plcols = adicionar_payload(df)
        dados[df['cenario'].iloc[0]] = df
    cenarios = sorted(dados)
    features = [c for c in dados[cenarios[0]].columns if FEATURE.fullmatch(c)]
    sets = conjuntos(features, plcols)

    def mi_todas(df, classes, seed):
        # A MI das features originais é calculada como antes (o ruído do estimador depende das colunas);
        # a das de payload, à parte, e as duas são juntadas.
        base = mi_por_classe(df, features, classes, seed, a.contra_benign)
        if not plcols:
            return base
        extra = mi_por_classe(df, plcols, classes, seed, a.contra_benign)
        return {c: (pd.concat([base[c][0], extra[c][0]]), base[c][1]) for c in base}

    rows, tops, suporte = [], [], []
    for seed in seeds:
        prep = {d: preparar(dados[d], a.modo, a.protocolo, seed) for d in cenarios}
        classes = sorted(set.intersection(*(set(prep[d][TARGET]) for d in cenarios)))
        mi = {d: mi_todas(prep[d], classes, seed) for d in cenarios}
        if a.contra_benign:
            classes = [c for c in classes if c != 'Benign']
        for d in cenarios:
            for c in classes:
                suporte.append({'semente': seed, 'cenario': d, 'classe': c, 'n_pos': mi[d][c][1]})
        for nome, cols in sets.items():
            for c in classes:
                for d1, d2 in combinations(cenarios, 2):
                    s1, s2 = mi[d1][c][0].reindex(cols), mi[d2][c][0].reindex(cols)
                    rho = spearmanr(s1, s2).statistic if s1.nunique() > 1 and s2.nunique() > 1 else np.nan
                    n = max(int((s1 > 0).sum()), int((s2 > 0).sum()))
                    rows.append({'semente': seed, 'conjunto': nome, 'classe': c, 'par': f'{d1} × {d2}',
                                 'jaccard_top10': jaccard(s1, s2), 'spearman': rho,
                                 'n_mi_positiva': n, 'jaccard_acaso': jaccard_acaso(n)})
                    if seed == seeds[0] and d1 == cenarios[0] and d2 == cenarios[1]:
                        t1 = set(s1[s1 > 0].nlargest(10).index); t2 = set(s2[s2 > 0].nlargest(10).index)
                        tops.append({'conjunto': nome, 'classe': c, 'par': f'{d1} × {d2}',
                                     f'so_{d1}': ', '.join(sorted(t1 - t2)), f'so_{d2}': ', '.join(sorted(t2 - t1))})

    res = pd.DataFrame(rows)
    res.to_csv(out / f'{tag}_jaccard_por_semente.csv', index=False)
    pd.DataFrame(tops).to_csv(out / f'{tag}_top10_diferencas.csv', index=False)
    sup = pd.DataFrame(suporte)
    sup.to_csv(out / f'{tag}_suporte.csv', index=False)
    resumo = (res.groupby(['conjunto', 'classe', 'par'])
              .agg(jaccard_media=('jaccard_top10', 'mean'), jaccard_min=('jaccard_top10', 'min'),
                   jaccard_max=('jaccard_top10', 'max'), spearman_media=('spearman', 'mean'),
                   acaso_media=('jaccard_acaso', 'mean'))
              .round(2).reset_index())
    resumo.to_csv(out / f'{tag}_resumo.csv', index=False)

    pd.set_option('display.width', 250)
    print(f'=== {tag}: suporte (positivos na amostra de análise, média das sementes) ===')
    print(sup.groupby(['classe', 'cenario'])['n_pos'].mean().unstack().round(0).to_string())
    for nome in sets:
        if a.protocolo == 'xrce' and nome in ('sem_tamanho_flags', 'payload_sem_flags'):
            continue
        print(f'\n=== {tag} · {nome}: Jaccard top 10 (média [mín–máx] em {len(seeds)} sementes) · acaso ===')
        r = resumo[resumo['conjunto'] == nome].copy()
        r['valor'] = r.apply(lambda x: f"{x.jaccard_media:.2f} [{x.jaccard_min:.2f}–{x.jaccard_max:.2f}]", axis=1)
        tab = r.pivot(index='classe', columns='par', values='valor')
        tab['acaso'] = r.groupby('classe')['acaso_media'].mean().round(2)
        print(tab.to_string())


if __name__ == '__main__':
    main()
