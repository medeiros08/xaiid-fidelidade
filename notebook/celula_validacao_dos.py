# Célula para colar no notebook "XAIID e TRAP IoT" (Kaggle), depois das células v2 e antes da seção 5.
# Depende do que as células 1, 2, 3 e 4 do notebook criam:
#   features, part, read_x, fit_median, TARGET, PARTS, SCENARIOS, SEED, OUT, np, pd, RandomForestClassifier.

# %% Arthur · Validação DoS-Heavy × DoS-Light
# Pergunta: os fluxos rotulados como DoS-Heavy e DoS-Light são diferentes em algum cenário?
# Três testes, por cenário, usando todos os fluxos de DoS do piloto (treino, validação e teste juntos,
# porque aqui é auditoria do rótulo, não avaliação de modelo):
#   1. Vetores idênticos: quantos fluxos de uma classe têm exatamente o mesmo vetor de features de um
#      fluxo da outra classe.
#   2. Classificador só Heavy × Light: Random Forest com validação cruzada estratificada (5 folds).
#      Acurácia balanceada perto (ou abaixo) de 0,5 = as features não separam as duas classes.
#   3. Kolmogorov–Smirnov por feature: KS perto de 0 = mesma distribuição nas duas classes.
# Leitura:
#   - Se Heavy × Light for inseparável em todos os cenários, o problema provavelmente é de desenho das
#     features (a diferença está na taxa do ataque, que um fluxo isolado não mostra), não de rótulo.
#   - Se for separável em um cenário e não em outro, suspeitar do rótulo no cenário inseparável.
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_score

DOS = ['DoS-Heavy', 'DoS-Light']
RESUMO_FEATURES = [f for f in ['bidirectional_packets', 'bidirectional_bytes', 'bidirectional_duration_ms',
                                'bidirectional_mean_piat_ms', 'bidirectional_mean_ps'] if f in features]

summary_rows, ks_rows, median_rows = [], [], []
for d in SCENARIOS:
    every = pd.concat([part(d, s) for s in PARTS], ignore_index=True)
    dos = every[every[TARGET].isin(DOS)].reset_index(drop=True)
    heavy, light = dos[dos[TARGET].eq('DoS-Heavy')], dos[dos[TARGET].eq('DoS-Light')]

    # 1. Vetores idênticos entre as classes
    hash_heavy, hash_light = set(heavy['hash_features']), set(light['hash_features'])

    # 2. Classificador binário Heavy × Light
    x = read_x(dos, fit_median(dos))
    y = dos[TARGET].eq('DoS-Light').astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    bal_acc = cross_val_score(model, x, y, cv=cv, scoring='balanced_accuracy')

    # 3. KS por feature
    ks = {}
    for f in features:
        a, b = heavy[f].dropna(), light[f].dropna()
        ks[f] = ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
        ks_rows.append({'cenario': d, 'feature': f, 'ks': ks[f]})
    ks = pd.Series(ks)

    summary_rows.append({
        'cenario': d, 'n_heavy': len(heavy), 'n_light': len(light),
        'vetores_distintos_heavy': len(hash_heavy), 'vetores_distintos_light': len(hash_light),
        'vetores_em_comum': len(hash_heavy & hash_light),
        'light_com_vetor_igual_a_heavy': round(float(light['hash_features'].isin(hash_heavy).mean()), 3),
        'acuracia_balanceada_media': round(float(bal_acc.mean()), 3),
        'acuracia_balanceada_dp': round(float(bal_acc.std()), 3),
        'ks_maximo': round(float(ks.max()), 3), 'feature_ks_maximo': ks.idxmax(),
        'n_features_ks_maior_0_2': int(ks.gt(0.2).sum()),
    })
    for label, frame in (('DoS-Heavy', heavy), ('DoS-Light', light)):
        median_rows.append({'cenario': d, 'classe': label,
                            **frame[RESUMO_FEATURES].median().round(2).to_dict()})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'arthur_validacao_dos.csv', index=False)
pd.DataFrame(ks_rows).to_csv(OUT / 'arthur_validacao_dos_ks.csv', index=False)
medians = pd.DataFrame(median_rows)
medians.to_csv(OUT / 'arthur_validacao_dos_medianas.csv', index=False)

pd.set_option('display.width', 200)
print('Resumo por cenário (acurácia balanceada perto ou abaixo de 0,5 = não separa; 1,0 = separa perfeitamente).')
print('Abaixo de 0,5 acontece quando há vetores idênticos com rótulos opostos: o "gêmeo" no treino tem o rótulo trocado.')
print(summary.T.to_string())
print('\nMedianas de algumas features:')
print(medians.to_string(index=False))
print('\nTop 5 features por KS em cada cenário:')
ks_table = pd.DataFrame(ks_rows)
for d in SCENARIOS:
    top = ks_table[ks_table['cenario'].eq(d)].nlargest(5, 'ks')
    print(f'  {d}: ' + ', '.join(f"{r.feature} ({r.ks:.2f})" for r in top.itertuples()))
