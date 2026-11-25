# XAIID · Fidelidade entre testbed físico e emulação em IDS

Código das análises que comparam um testbed físico (D1) com emulações em Docker (D2, e D3 com 100 nós) de redes que usam os protocolos **XRCE-DDS** e **Zenoh**, no contexto de detecção de intrusão com aprendizado de máquina. Faz parte do projeto de pesquisa XAIID (estratégias inteligentes e explicáveis para detecção de intrusões em sistemas ciber-físicos).

A pergunta central é: as features que um IDS considera importantes no ambiente físico continuam importantes no emulado? E um modelo treinado num ambiente funciona no outro?

## O que tem aqui

```
analises/
  amostrar.py      sorteia amostras dos CSVs de fluxos (lidos de dentro dos ZIPs), com várias sementes
  analisar.py      informação mútua por classe, top 10, Jaccard e Spearman entre cenários
  classificar.py   Random Forest com treino num cenário e teste no outro (D1 ↔ D2)
  figura.py        figura-resumo do Jaccard D1 × D2
  janela.py        features de janela (fluxos por intervalo de tempo), em desenvolvimento
notebook/
  celulas_v2.py               células extras para o notebook do grupo (piso de ruído, permutação por grupos)
  celulas_v3_sem_tamanho.py   comparação sem as features de tamanho de pacote
  celula_validacao_dos.py     teste se DoS-Heavy e DoS-Light são separáveis
```

As células da pasta `notebook/` foram escritas para um notebook interno do grupo e dependem de variáveis definidas nele (`data`, `part`, `read_x`, `features` etc.). Os scripts da pasta `analises/` funcionam sozinhos.

## Dados

Os datasets **não estão neste repositório**. São fluxos bidirecionais extraídos com o nfstream 6.6.0 (57 features estatísticas de duração, contagem, tamanho, intervalo entre pacotes e flags TCP) e pertencem aos autores do dataset. Os scripts esperam os ZIPs originais, com a estrutura `<variante>/<cenário>/dataset-xrce.csv` e `dataset-zenoh.csv`, e conferem a integridade pelo SHA-256 publicado com eles.

## Como rodar

Python 3.11 ou mais recente:

```
python -m venv .venv
.venv\Scripts\activate        # no Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

1. Sortear as amostras (exemplo: XRCE-DDS, cenário D1, regra de blocos de 3 h):

   ```
   python -I analises/amostrar.py sem-ruido_D1_v2.zip sem-ruido/D1/dataset-xrce.csv D1 amostras/v2_xrce_D1.csv --modo blocos --ancora <início da captura em ms>
   ```

   Repita para D2 e D3. O modo `classe` sorteia por classe, sem blocos, para cenários com cronogramas diferentes.

2. Comparar os rankings de features entre os cenários (5 sementes):

   ```
   python -I analises/analisar.py resultados --modo v2 --protocolo xrce amostras/v2_xrce_D1.csv amostras/v2_xrce_D2.csv amostras/v2_xrce_D3.csv
   ```

   Com `--contra-benign`, cada ataque é comparado só com o tráfego benigno.

3. Treino e teste cruzados:

   ```
   python -I analises/classificar.py resultados --protocolo xrce amostras/v2_xrce_D1.csv amostras/v2_xrce_D2.csv
   ```

## Método, em resumo

- **Amostra.** Até 150 fluxos por bloco de 3 h e classe, excluindo 4 min junto às fronteiras dos blocos. Os blocos 1, 3, 5, 6, 7, 8 e 10 são de treino, e os blocos 4, 9 e 11, de teste.
- **Importância.** Informação mútua de cada feature com o alvo binário "é desta classe ou não", calculada no treino.
- **Comparação.** O índice de Jaccard entre os dez primeiros atributos de cada cenário (1 = mesmas features, 0 = nenhuma em comum) é comparado com o valor esperado por acaso. O Spearman compara a ordem completa. Todas as análises são repetidas com cinco sementes (20261117 a 20261121).
- **Conjuntos de features.** As análises rodam com todas as features, sem as de tamanho (`*_bytes`, `*_ps`), que carregam diferenças do ambiente de captura (por exemplo, o padding do quadro Ethernet), e sem tamanho nem flags TCP.

## Autor

Arthur de Freitas Medeiros, bolsista de iniciação científica no projeto XAIID (UNIPAMPA).

## Licença

O código está sob a licença MIT (veja `LICENSE`). Os datasets não fazem parte do repositório nem da licença.
