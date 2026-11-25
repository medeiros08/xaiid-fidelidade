# Célula para colar no notebook "XAIID e TRAP IoT" (Kaggle), depois das células v2 (precisa da v2 · 1).
# Usa o que a seção 4 e a v2 · 1 já calcularam:
#   features, importance, mi_half, n_pos, jaccard_top, spearman_ok, mean_ok,
#   CLASSES, SCENARIOS, OUT, save_plot, np, pd, plt, sns.

# %% Arthur v3 · Sensibilidade: comparação D1 × D2 sem as features de tamanho
# Motivo: o LEIA-ME do dataset confirma que o quadro curto tem 60 B no D1 e 42/54 B no D2/D3, porque a
# bridge do Docker não completa o quadro Ethernet. Toda feature de tamanho (*_bytes, *_min/mean/max/stddev_ps)
# carrega esse artefato. Pergunta: as diferenças D1 × D2 da seção 4 continuam sem essas features?
#   - Se o Jaccard D1 × D2 subir muito sem elas, a diferença era sobretudo o artefato do Docker.
#   - Se continuar baixo, há diferença real de comportamento (tempo, contagens, flags).
# Não é preciso recalcular a informação mútua: o mutual_info_classif estima cada feature separadamente,
# então basta tirar as features de tamanho dos rankings já calculados e refazer o top 10.
from itertools import combinations

TAMANHO = [f for f in features if f.endswith('_bytes') or f.endswith('_ps')]
SEM_TAMANHO = [f for f in features if f not in TAMANHO]
print(f'{len(features)} features: {len(TAMANHO)} de tamanho removidas, {len(SEM_TAMANHO)} mantidas '
      '(duração, contagem de pacotes, intervalos e flags).')


def mi_ranking(classe, cenario, cols):
    f = importance[importance['metodo'].eq('informacao_mutua') & importance['classe'].eq(classe)
                   & importance['cenario'].eq(cenario)].set_index('feature')['score']
    return f.reindex(cols)


# 1. Treino inteiro (mesma comparação da seção 4), com e sem as features de tamanho
full_rows = []
for classe in CLASSES:
    for a, b in combinations(SCENARIOS, 2):
        for nome, cols in (('todas', features), ('sem_tamanho', SEM_TAMANHO)):
            sa, sb = mi_ranking(classe, a, cols), mi_ranking(classe, b, cols)
            full_rows.append({'classe': classe, 'par': f'{a} × {b}', 'conjunto': nome,
                              'jaccard_top10': jaccard_top(sa, sb), 'spearman': spearman_ok(sa, sb)})
full = pd.DataFrame(full_rows)
full.to_csv(OUT / 'arthur_v3_sem_tamanho_rankings.csv', index=False)

print('\nJaccard top 10 da MI, treino inteiro (todas → sem tamanho):')
pares = [f'{a} × {b}' for a, b in combinations(SCENARIOS, 2)]
tab = (full.pivot_table(index='classe', columns=['par', 'conjunto'], values='jaccard_top10')
       .reindex(index=CLASSES, columns=pd.MultiIndex.from_product([pares, ['todas', 'sem_tamanho']])))
print(tab.round(2).to_string())

# 2. Piso de ruído (metade × metade, igual à v2 · 1), só com as features sem tamanho
def compare_halves_subset(a, b, classe, cols):
    pairs = [(0, 1)] if a == b else [(0, 1), (1, 0)]
    jac = [jaccard_top(mi_half[a, classe, i].reindex(cols), mi_half[b, classe, j].reindex(cols))
           for i, j in pairs]
    return mean_ok(jac)


half_rows = []
for classe in CLASSES:
    row = {'classe': classe}
    for d in SCENARIOS:
        row[f'dentro {d}'] = compare_halves_subset(d, d, classe, SEM_TAMANHO)
    for a, b in combinations(SCENARIOS, 2):
        row[f'{a} × {b}'] = compare_halves_subset(a, b, classe, SEM_TAMANHO)
        row[f'folga {a} × {b}'] = pd.Series([row[f'dentro {a}'], row[f'dentro {b}']]).min() - row[f'{a} × {b}']
    half_rows.append(row)
halves = pd.DataFrame(half_rows).set_index('classe')
halves.to_csv(OUT / 'arthur_v3_sem_tamanho_piso.csv')
print('\nPiso de ruído sem as features de tamanho (metade × metade; folga = menor dentro − entre):')
print(halves.round(2).to_string())

# 3. Quais features mudam entre D1 e D2 quando o tamanho sai da conta
print('\nTop 10 da MI sem tamanho, D1 × D2:')
for classe in CLASSES:
    tops = [set(mi_ranking(classe, d, SEM_TAMANHO).pipe(lambda s: s[s.gt(0)]).nlargest(10).index)
            for d in ('D1', 'D2')]
    print(f'{classe}: {len(tops[0] & tops[1])} de 10 em comum')
    print('  só no D1:', sorted(tops[0] - tops[1]))
    print('  só no D2:', sorted(tops[1] - tops[0]))

# 4. Gráfico: Jaccard D1 × D2 (treino inteiro) com todas × sem tamanho, e a folga sem tamanho
plot = full[full['par'].eq('D1 × D2')].pivot(index='classe', columns='conjunto', values='jaccard_top10')
plot = plot.reindex(CLASSES)[['todas', 'sem_tamanho']]
plot['folga (metades, sem tamanho)'] = halves['folga D1 × D2']
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(plot, vmin=-0.3, vmax=1, center=0, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
ax.axvline(2, color='black', lw=2)
ax.set_title('Arthur v3 · D1 × D2 sem as features de tamanho (MI, top 10)')
ax.set_xlabel('')
fig.tight_layout()
save_plot(fig, 'arthur_v3_sem_tamanho.png')


# %% Arthur v3 · b. Quantas features têm MI positiva (para saber o Jaccard esperado por acaso)
# Com poucas features disponíveis, dois top 10 sorteados ao acaso já coincidem bastante. O Jaccard esperado
# por acaso, sorteando k de n features duas vezes, é aproximadamente (k²/n) / (2k − k²/n).
def n_positivas(classe, cenario, cols):
    return int(mi_ranking(classe, cenario, cols).gt(0).sum())


def jaccard_acaso(n, k=10):
    if n == 0:
        return np.nan
    k = min(k, n)
    comum = k * k / n
    return comum / (2 * k - comum)


acaso_rows = []
for classe in CLASSES:
    for nome, cols in (('todas', features), ('sem_tamanho', SEM_TAMANHO)):
        n1, n2 = n_positivas(classe, 'D1', cols), n_positivas(classe, 'D2', cols)
        observado = full.query('classe == @classe and conjunto == @nome and par == "D1 × D2"')['jaccard_top10'].iloc[0]
        acaso_rows.append({'classe': classe, 'conjunto': nome, 'n_pos_D1': n1, 'n_pos_D2': n2,
                           'jaccard_acaso': jaccard_acaso(max(n1, n2)), 'jaccard_D1xD2': observado})
acaso = pd.DataFrame(acaso_rows)
acaso.to_csv(OUT / 'arthur_v3_jaccard_acaso.csv', index=False)
print('Features com MI positiva e Jaccard D1 × D2 esperado por acaso (todas × sem tamanho):')
print(acaso.pivot(index='classe', columns='conjunto').reindex(CLASSES).round(2).to_string())
