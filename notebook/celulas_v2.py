# Células para colar no notebook "XAIID e TRAP IoT" (Kaggle), logo depois da seção 4 · Arthur.
# Dependem do que as células 1, 5, 6 e 10 do notebook já criam:
#   data, features, part, read_x, fit_median, TARGET, CLASSES, SCENARIOS, SEED, OUT,
#   importance e similarity (rankings e comparação da seção 4), save_plot, np, pd, plt, sns,
#   RandomForestClassifier. Os demais imports estão no início de cada célula.
# Cada bloco "# %%" é uma célula.

# %% Arthur v2 · 0. Ver o resultado da seção 4 em texto
# Mostra o Spearman (o notebook só desenha o Jaccard) e quais features entram ou saem do top 10
# da informação mútua entre D1 e D2. É daqui que sai a interpretação da entrega.
print('Spearman entre cenários (1 = mesma ordem, 0 = sem relação):')
print(similarity.pivot_table(index='classe', columns=['metodo', 'par'], values='spearman')
      .reindex(CLASSES).round(2).to_string())


def top10(metodo, classe, cenario):
    f = importance[importance['metodo'].eq(metodo) & importance['classe'].eq(classe)
                   & importance['cenario'].eq(cenario)]
    return list(f[f['score'].gt(0)].nlargest(10, 'score')['feature'])


for classe in CLASSES:
    a, b = top10('informacao_mutua', classe, 'D1'), top10('informacao_mutua', classe, 'D2')
    print(f'\n{classe}: {len(set(a) & set(b))} de 10 em comum')
    print('  só no D1:', sorted(set(a) - set(b)))
    print('  só no D2:', sorted(set(b) - set(a)))


# %% Arthur v2 · 1. Piso de ruído da informação mútua: metade × metade, dentro e entre cenários
# Pergunta: a diferença D1 × D2 é maior do que a diferença entre duas metades do próprio cenário?
# Se o Jaccard "dentro" já for baixo (classes raras), o Jaccard baixo "entre" cenários não prova
# que o cenário muda as features: pode ser só amostra pequena.
#
# Para a comparação ser justa, "dentro" e "entre" usam exatamente o mesmo desenho:
#   dentro D1 = MI(D1, metade A) × MI(D1, metade B)
#   D1 × D2   = média de MI(D1, A) × MI(D2, B) e MI(D1, B) × MI(D2, A)
# Nos dois casos as amostras têm o mesmo tamanho e vêm de blocos diferentes; a única coisa que muda
# no "entre" é o cenário. (Comparar A × A entre cenários usaria os mesmos blocos de tempo e
# deixaria o "entre" artificialmente parecido.)
from itertools import combinations

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

METADES = ([1, 5, 7, 10], [3, 6, 8])  # os 7 blocos de treino divididos em duas metades


def mi_scores(frame, label, median):
    x = read_x(frame, median)
    y = frame[TARGET].eq(label).astype(int)
    scores = pd.Series(0.0, index=features)
    active = x.columns[x.nunique().gt(1)]
    if y.nunique() == 2 and len(active):
        scores.loc[active] = mutual_info_classif(x[active], y, random_state=SEED)
    return scores, int(y.sum())


def jaccard_top(a, b, k=10):
    # Mesmo critério da seção 4: só entra no top k quem tem score positivo.
    ta, tb = (set(s[s.gt(0)].nlargest(k).index) for s in (a, b))
    union = ta | tb
    return len(ta & tb) / len(union) if union else np.nan


def spearman_ok(a, b):
    return spearmanr(a, b).statistic if a.nunique() > 1 and b.nunique() > 1 else np.nan


# MI de cada (cenário, classe, metade). A mediana de imputação vem do treino inteiro do cenário,
# igual à seção 4, e é a mesma para as duas metades.
mi_half, n_pos = {}, {}
for d in SCENARIOS:
    train = part(d, 'treino')
    median = fit_median(train)
    for h, blocos in enumerate(METADES):
        half = train[train['bloco'].isin(blocos)]
        for label in CLASSES:
            mi_half[d, label, h], n_pos[d, label, h] = mi_scores(half, label, median)


def mean_ok(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def compare_halves(a, b, label):
    # a == b: dentro do cenário (A × B). a != b: entre cenários, metades cruzadas nos dois sentidos.
    pairs = [(0, 1)] if a == b else [(0, 1), (1, 0)]
    jac = [jaccard_top(mi_half[a, label, i], mi_half[b, label, j]) for i, j in pairs]
    rho = [spearman_ok(mi_half[a, label, i], mi_half[b, label, j]) for i, j in pairs]
    return mean_ok(jac), mean_ok(rho)


piso_rows = []
for label in CLASSES:
    for d in SCENARIOS:
        jac, rho = compare_halves(d, d, label)
        piso_rows.append({'classe': label, 'comparacao': f'dentro {d}', 'tipo': 'dentro',
                          'cenario_a': d, 'cenario_b': d,
                          'n_pos_a': n_pos[d, label, 0], 'n_pos_b': n_pos[d, label, 1],
                          'jaccard_top10': jac, 'spearman': rho})
    for a, b in combinations(SCENARIOS, 2):
        jac, rho = compare_halves(a, b, label)
        piso_rows.append({'classe': label, 'comparacao': f'{a} × {b}', 'tipo': 'entre',
                          'cenario_a': a, 'cenario_b': b,
                          'n_pos_a': n_pos[a, label, 0] + n_pos[a, label, 1],
                          'n_pos_b': n_pos[b, label, 0] + n_pos[b, label, 1],
                          'jaccard_top10': jac, 'spearman': rho})
piso = pd.DataFrame(piso_rows)
piso.to_csv(OUT / 'arthur_piso_ruido_mi.csv', index=False)

ordem = [f'dentro {d}' for d in SCENARIOS] + [f'{a} × {b}' for a, b in combinations(SCENARIOS, 2)]
comparativo = piso.pivot(index='classe', columns='comparacao', values='jaccard_top10') \
    .reindex(index=CLASSES, columns=ordem)

# folga = menor "dentro" dos dois cenários − "entre". Positiva e grande: a diferença vem do cenário.
# Perto de zero ou negativa: o "entre" não é pior que o ruído de amostra, então é inconclusivo.
for a, b in combinations(SCENARIOS, 2):
    comparativo[f'folga {a} × {b}'] = (comparativo[[f'dentro {a}', f'dentro {b}']].min(axis=1)
                                        - comparativo[f'{a} × {b}'])
comparativo.to_csv(OUT / 'arthur_mi_dentro_vs_entre.csv')
print('Jaccard top 10 da MI, sempre metade × metade (folga = menor dentro − entre):')
print(comparativo.round(2).to_string())

# Referência: o "entre" da seção 4 usa o treino inteiro, então tende a sair maior que o daqui.
print('\nSeção 4 (treino inteiro), só para referência:')
print(similarity.query("metodo == 'informacao_mutua'")
      .pivot(index='classe', columns='par', values='jaccard_top10').reindex(CLASSES).round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(comparativo[ordem], vmin=0, vmax=1, annot=True, fmt='.2f', cmap='YlGn', ax=ax)
ax.axvline(len(SCENARIOS), color='black', lw=2)
ax.set_xlabel('')
ax.set_title('Arthur · Jaccard top 10 da MI (metade × metade): dentro do cenário × entre cenários')
fig.tight_layout()
save_plot(fig, 'arthur_mi_dentro_vs_entre.png')


# %% Arthur v2 · 2. Permutação por grupos de features correlacionadas
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score

# Na seção 4 a permutação saiu quase toda zero. Motivo provável: há várias features carregando a
# mesma informação (ex.: contagens bidirecionais e por sentido). Embaralhar uma só não muda a AP,
# porque o modelo usa a vizinha. Aqui as features com |Spearman| >= LIMIAR_RHO são embaralhadas
# juntas, e o critério de "importante" é a queda média de AP passar de 2 desvios-padrão.
LIMIAR_RHO = 0.9
N_REPETICOES = 10


def feature_groups(threshold=LIMIAR_RHO):
    # Grupos calculados no treino dos três cenários juntos, para serem os mesmos em todos.
    train = data[data['split'].eq('treino')]
    x = read_x(train, fit_median(train))
    rho = x.corr(method='spearman').abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1.0)
    labels = fcluster(linkage(squareform(1 - rho, checks=False), method='average'),
                      t=1 - threshold, criterion='distance')
    groups = {}
    for g in np.unique(labels):
        cols = list(x.columns[labels == g])
        name = cols[0] if len(cols) == 1 else f'{cols[0]} (+{len(cols) - 1})'
        groups[name] = cols
    return groups


def grouped_permutation(model, x, y, groups):
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, model.predict_proba(x)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(N_REPETICOES):
            xp = x.copy()
            xp[cols] = xp[cols].to_numpy()[rng.permutation(len(xp))]
            drops.append(base - average_precision_score(y, model.predict_proba(xp)[:, 1]))
        rows.append({'grupo': name, 'n_features': len(cols),
                     'queda_AP_media': float(np.mean(drops)), 'queda_AP_dp': float(np.std(drops))})
    return base, pd.DataFrame(rows)


groups = feature_groups()
pd.DataFrame([{'grupo': g, 'feature': f} for g, cols in groups.items() for f in cols]) \
    .to_csv(OUT / 'arthur_grupos_features.csv', index=False)
print(f'{len(features)} features em {len(groups)} grupos com |rho| >= {LIMIAR_RHO}')

group_rows = []
for d in SCENARIOS:
    train, val = part(d, 'treino'), part(d, 'validacao')
    median = fit_median(train)
    xt, xv = read_x(train, median), read_x(val, median)
    for label in CLASSES:
        yt = train[TARGET].eq(label).astype(int)
        yv = val[TARGET].eq(label).astype(int)
        if yt.nunique() < 2 or yv.nunique() < 2:
            continue
        # Mesmo modelo da seção 4, para mudar só a forma de permutar.
        model = RandomForestClassifier(n_estimators=40, max_depth=8, class_weight='balanced_subsample',
                                       random_state=SEED, n_jobs=-1).fit(xt, yt)
        base, table = grouped_permutation(model, xv, yv, groups)
        group_rows.append(table.assign(cenario=d, classe=label, AP_base=base,
                                       n_pos_treino=int(yt.sum()), n_pos_validacao=int(yv.sum())))
grouped = pd.concat(group_rows, ignore_index=True)
grouped['significativo'] = grouped['queda_AP_media'].gt(2 * grouped['queda_AP_dp']) & grouped['queda_AP_media'].gt(0)
grouped.to_csv(OUT / 'arthur_permutacao_grupos.csv', index=False)

# AP de base: se já estiver perto de 1 com folga, nenhuma permutação isolada consegue derrubar.
print(grouped.drop_duplicates(['cenario', 'classe'])
      .pivot(index='classe', columns='cenario', values='AP_base').reindex(CLASSES).round(3).to_string())

group_sim = []
for label, frame in grouped.groupby('classe'):
    score = frame.pivot(index='grupo', columns='cenario', values='queda_AP_media')
    keep = frame.pivot(index='grupo', columns='cenario', values='significativo').fillna(False).astype(bool)
    for a, b in combinations(SCENARIOS, 2):
        if not {a, b}.issubset(score.columns):
            continue
        tops = [set(score[d][keep[d]].nlargest(10).index) for d in (a, b)]
        union = tops[0] | tops[1]
        group_sim.append({'classe': label, 'par': f'{a} × {b}',
                          'n_sig_a': len(tops[0]), 'n_sig_b': len(tops[1]),
                          'jaccard_top10': len(tops[0] & tops[1]) / len(union) if union else np.nan,
                          'spearman': spearman_ok(score[a], score[b])})
group_sim = pd.DataFrame(group_sim)
group_sim.to_csv(OUT / 'arthur_comparacao_grupos.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for ax, metric in zip(axes, ['jaccard_top10', 'spearman']):
    table = group_sim.pivot(index='classe', columns='par', values=metric).reindex(CLASSES)
    sns.heatmap(table, vmin=0 if metric == 'jaccard_top10' else -1, vmax=1,
                annot=True, fmt='.2f', cmap='YlGn', ax=ax)
    ax.set_title(metric.replace('_', ' '))
fig.suptitle('Arthur · Permutação por grupos (queda de AP > 2 dp)')
fig.tight_layout()
save_plot(fig, 'arthur_permutacao_grupos.png')
