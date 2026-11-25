"""Treino × teste cruzado (D1 ↔ D2) com e sem as features de tamanho, com várias sementes.

Mesmo desenho da análise de treino × teste cruzado do notebook do grupo: Random Forest multiclasse (100 árvores, profundidade 12),
peso de classe balanceado no treino, mediana do treino para os vazios, treino nos blocos 1, 3, 5, 6, 7, 8 e 10
e teste nos blocos 4, 9 e 11.

Modos:
  v2   amostras v2 de amostrar.py --modo blocos (blocos de 3 h já marcados na coluna 'bloco').
  nsb  amostras NetSecBed de amostrar.py --modo classe. Cada cenário é dividido em 12 blocos iguais a partir do
       início do Benign, que cobre a captura toda (12 h no D1 → blocos de 1 h; 4 h no D2 → blocos de 20 min);
       cada bloco tem uma rodada de cada ataque. O bloco vem do início do fluxo, sem exclusão nas fronteiras:
       os fluxos duram no máximo 2 min e as rodadas do EntityFlood começam na virada do bloco. Aplica as
       regras do LEIA-ME (filtrar_nsb) e sorteia até 150 fluxos por (bloco, classe) com a semente, como a v2.

Uso: python -I classificar.py SAIDA_DIR --protocolo xrce [--modo nsb] amostra_D1.csv amostra_D2.csv
"""
import argparse
import sys
from pathlib import Path

# Com python -I a pasta do script não entra no sys.path; analisar.py fica ao lado deste arquivo.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from analisar import FEATURE, TARGET, TREINO, conjuntos, filtrar_nsb

TESTE = [4, 9, 11]
START, END = 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms'
CAP = 150


def blocos_nsb(df):
    """12 blocos iguais a partir do Benign do próprio cenário (calculado sobre o arquivo inteiro)."""
    b = df[df[TARGET] == 'Benign']
    ancora = b[START].min()
    passo = (b[END].max() - ancora) / 12
    bloco = np.floor((df[START] - ancora) / passo).astype(int) + 1
    return df.assign(bloco=bloco.where((bloco >= 1) & (bloco <= 12), 0)), passo


def amostra_nsb(df, protocolo, seed):
    s = filtrar_nsb(df[df[f's_{seed}']], protocolo)
    s = s[s['bloco'].isin(TREINO + TESTE)]
    rng = np.random.default_rng(seed)
    s = s.assign(_p=rng.random(len(s))).sort_values('_p')
    return s.groupby(['bloco', TARGET], sort=False).head(CAP).drop(columns='_p')


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # o console do Windows (cp1252) não imprime '→'
    p = argparse.ArgumentParser()
    p.add_argument('saida'); p.add_argument('amostras', nargs=2)
    p.add_argument('--protocolo', choices=['xrce', 'zenoh'], required=True)
    p.add_argument('--modo', choices=['v2', 'nsb'], default='v2')
    p.add_argument('--sementes', default='20261117,20261118,20261119,20261120,20261121')
    p.add_argument('--janela', action='store_true',
                   help='junta as features de janela de janela.py (janela_<amostra>.csv, na mesma pasta) e '
                        'avalia cada conjunto com e sem elas')
    a = p.parse_args()
    seeds = [int(s) for s in a.sementes.split(',')]
    dados = {}
    for path in a.amostras:
        df = pd.read_csv(path, low_memory=False)
        if a.janela:
            jan = pd.read_csv(Path(path).with_name('janela_' + Path(path).name))
            df = df.merge(jan, on='linha_csv', how='left', validate='one_to_one')
            assert df[jan.columns].notna().all().all(), 'linha da amostra sem feature de janela'
        if a.modo == 'nsb':
            df, passo = blocos_nsb(df)
            print(f"{df['cenario'].iloc[0]}: blocos de {passo / 60000:.1f} min")
        dados[df['cenario'].iloc[0]] = df
    features = [c for c in dados['D1'].columns if FEATURE.fullmatch(c)]
    sets = conjuntos(features)
    if a.protocolo == 'xrce':
        sets.pop('sem_tamanho_flags')
    if a.janela:
        jcols = [c for c in dados['D1'].columns if c.startswith('jan')]
        sets.update({f'{nome}+janela': cols + jcols for nome, cols in list(sets.items())})

    rows, per_class, suporte = [], [], []
    for seed in seeds:
        part = {}
        for d, df in dados.items():
            s = amostra_nsb(df, a.protocolo, seed) if a.modo == 'nsb' else df[df[f's_{seed}']]
            part[d, 'treino'] = s[s['bloco'].isin(TREINO)]
            part[d, 'teste'] = s[s['bloco'].isin(TESTE)]
            for etapa in ('treino', 'teste'):
                suporte += [{'semente': seed, 'cenario': d, 'etapa': etapa, 'classe': c, 'n': n}
                            for c, n in part[d, etapa][TARGET].value_counts().items()]
        for nome, cols in sets.items():
            for origem in ('D1', 'D2'):
                tr = part[origem, 'treino']
                xt = tr[cols].replace([np.inf, -np.inf], np.nan)
                med = xt.median().fillna(0)
                classes = sorted(tr[TARGET].unique())
                w = tr[TARGET].map(len(tr) / (len(classes) * tr[TARGET].value_counts()))
                model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
                model.fit(xt.fillna(med), tr[TARGET], sample_weight=w)
                for destino in ('D1', 'D2'):
                    te = part[destino, 'teste']
                    te = te[te[TARGET].isin(classes)]
                    pred = model.predict(te[cols].replace([np.inf, -np.inf], np.nan).fillna(med))
                    rows.append({'semente': seed, 'conjunto': nome, 'treino': origem, 'teste': destino,
                                 'f1_macro': f1_score(te[TARGET], pred, average='macro')})
                    rec = recall_score(te[TARGET], pred, labels=classes, average=None, zero_division=0)
                    per_class += [{'semente': seed, 'conjunto': nome, 'treino': origem, 'teste': destino,
                                   'classe': c, 'recall': r} for c, r in zip(classes, rec)]

    out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.modo}_{a.protocolo}' + ('_janela' if a.janela else '')
    res = pd.DataFrame(rows); res.to_csv(out / f'{tag}_cruzado_f1.csv', index=False)
    pc = pd.DataFrame(per_class); pc.to_csv(out / f'{tag}_cruzado_recall.csv', index=False)
    pd.set_option('display.width', 250)
    if a.modo == 'nsb':
        sup = pd.DataFrame(suporte); sup.to_csv(out / f'{tag}_cruzado_suporte.csv', index=False)
        print(f'=== {tag}: suporte (média das sementes) ===')
        print(sup.groupby(['classe', 'cenario', 'etapa'])['n'].mean().unstack([1, 2]).round(0).to_string())
    print(f'\n=== {tag}: F1 macro (média [mín–máx] em {len(seeds)} sementes) ===')
    g = res.groupby(['conjunto', 'treino', 'teste'])['f1_macro'].agg(['mean', 'min', 'max']).round(3)
    print(g.to_string())
    for origem, destino in (('D1', 'D2'), ('D2', 'D1')):
        print(f'\n=== {tag}: recall por classe, treino {origem} → teste {destino} (média das sementes) ===')
        r = (pc[(pc.treino == origem) & (pc.teste == destino)]
             .groupby(['classe', 'conjunto'])['recall'].mean().unstack().round(2))
        r[f'{origem}→{origem} todas'] = (pc[(pc.treino == origem) & (pc.teste == origem) & (pc.conjunto == 'todas')]
                                        .groupby('classe')['recall'].mean().round(2))
        print(r.to_string())


if __name__ == '__main__':
    main()
