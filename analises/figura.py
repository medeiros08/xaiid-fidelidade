"""Gera a figura-resumo D1 × D2 (Jaccard top 10 da MI, média e faixa em 5 sementes) dos quatro conjuntos."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R = Path(sys.argv[1])
ORANGE, BLUE = '#eb6834', '#2a78d6'
INK, INK2, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
PAINEIS = [
    ('v2_xrce', 'sem_tamanho', 'XRCE-DDS · 8 classes (v2)', 'sem tamanho'),
    ('v2_zenoh', 'sem_tamanho', 'Zenoh · 8 classes (v2)', 'sem tamanho'),
    ('nsb_xrce', 'sem_tamanho', 'XRCE-DDS · NetSecBed', 'sem tamanho'),
    ('nsb_zenoh', 'sem_tamanho_flags', 'Zenoh · NetSecBed', 'sem tamanho e sem flags TCP'),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), dpi=170)
fig.patch.set_facecolor(BG)
for ax, (tag, conj, titulo, rotulo) in zip(axes.flat, PAINEIS):
    df = pd.read_csv(R / f'{tag}_resumo.csv')
    df = df[df['par'] == 'D1 × D2']
    classes = sorted(df['classe'].unique(), key=lambda c: (c != 'Benign', c))
    y = list(range(len(classes)))[::-1]
    ax.set_facecolor(BG)
    for nome, cor, dy in (('todas', ORANGE, 0.14), (conj, BLUE, -0.14)):
        d = df[df['conjunto'] == nome].set_index('classe').reindex(classes)
        yy = [v + dy for v in y]
        ax.hlines(yy, d['jaccard_min'], d['jaccard_max'], color=cor, lw=2, alpha=0.45, zorder=2)
        ax.scatter(d['jaccard_media'], yy, s=46, color=cor, edgecolor=BG, linewidth=1.5, zorder=3)
        acaso = d['acaso_media'].mean()
        ax.axvline(acaso, ymax=0.97, color=cor, lw=1, ls=(0, (3, 3)), alpha=0.8, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([c.replace('XRCE-', '').replace('Zenoh-', '') for c in classes], fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.04); ax.set_ylim(-0.7, len(classes) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(['0', '0,25', '0,50', '0,75', '1'], fontsize=8.5, color=INK2)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(AXIS); ax.tick_params(length=0)
    ax.set_title(f'{titulo}\n(azul: {rotulo})', fontsize=10.5, color=INK, loc='left')
fig.suptitle('Jaccard do top 10 da informação mútua, D1 físico × D2 emulado\n'
             'ponto = média de 5 sementes · traço = mín–máx · tracejado = esperado por acaso',
             x=0.02, ha='left', fontsize=12.5, color=INK, fontweight='bold')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=ORANGE, label='Todas as 57 features'),
                    Line2D([], [], marker='o', ls='', color=BLUE, label='Sem as features de tamanho (e, no Zenoh NetSecBed, sem flags TCP)')],
           loc='upper left', bbox_to_anchor=(0.015, 0.905), ncol=2, frameon=False, fontsize=9)
fig.text(0.02, 0.01, 'v2: piloto com a regra do notebook (até 150 fluxos por bloco e classe, blocos de treino). '
         'NetSecBed: até 1000 fluxos por classe (80% por semente); benigno do XRCE só .103/.105 em UDP 8888; DiscoveryPoison fora.',
         fontsize=7.5, color=INK2)
fig.subplots_adjust(top=0.8, bottom=0.07, left=0.11, right=0.98, hspace=0.38, wspace=0.32)
fig.savefig(R.parent / 'jaccard_D1xD2_quatro_conjuntos.png', facecolor=BG)
