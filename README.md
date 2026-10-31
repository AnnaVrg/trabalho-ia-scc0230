# Projeto de Inteligência Artificial (SCC0230) — Abordagem A: Busca em Redes

Repositório dedicado ao desenvolvimento e experimentação de algoritmos de busca clássicos e informados aplicados a **Redes Aleatórias**, desenvolvido por um grupo de 4 estudantes da disciplina de Inteligência Artificial do ICMC/USP.

---

## 👥 Integrantes do Grupo
* Integrante 1
* Integrante 2
* Integrante 3
* Integrante 4

---

## 📂 Estrutura do Repositório

O projeto foi estruturado em módulos independentes e limpos (sem arquivos compactados, conforme exigido nas diretrizes da disciplina):

```text
.
├── rede_aleatoria.py           # Gerador da Rede Aleatória (vértices, arestas e heurística)
├── algoritmos_busca.py         # Implementação de BFS, DFS, Best-First, A*, HC Estrito, HC Não-Estrito e Dijkstra
├── main_experimentos.py        # Script unificado para execução dos Experimentos I, II e III
├── graficos/                   # Diretório gerado automaticamente com as imagens e gráficos
│   ├── exp1/                   # Visualizações da exploração do espaço de busca (n=200)
│   ├── exp2/                   # Gráficos de distância média e tempo médio (n=2500, p em {5%, 2.5%, 1%})
│   └── exp3/                   # Comparativos diretos entre A* e Dijkstra
└── .gitignore                  # Arquivos ignorados pelo Git
```

---

## 🚀 Como Executar os Experimentos

### 1. Pré-requisitos
Certifique-se de ter o Python 3 instalado juntamente com as bibliotecas `matplotlib` e `numpy` utilizadas para visualização e métricas:

```bash
pip install matplotlib numpy
```

### 2. Executar os Testes e Gerar Gráficos
Para rodar todos os experimentos (`Experimento I`, `Experimento II` e `Experimento III`) e salvar automaticamente todos os gráficos e imagens nas subpastas `graficos/exp1`, `graficos/exp2` e `graficos/exp3`, execute:

```bash
python main_experimentos.py
```

---

## 🧪 Descrição dos Experimentos Realizados

1. **Experimento I (Rede Reduzida e Visualização):**
   * Gera uma Rede Aleatória com $n = 200$ vértices e $p = 2.5\%$.
   * Seleciona um par de vértices `(origem, destino)` e executa todos os algoritmos de busca.
   * Salva imagens individuais detalhando o espaço de busca explorado e o caminho final percorrido para cada algoritmo.

2. **Experimento II (Escalabilidade e Variação de Parâmetros):**
   * Avalia redes de grande porte com $n = 2500$ vértices sob três configurações de probabilidade: $p \in \{5\%, 2.5\%, 1\%\}$.
   * Coleta métricas sobre **10 pares distintos de vértices**, reportando a **distância média percorrida**, o **tempo médio de execução** e o **desvio padrão** de cada algoritmo.

3. **Experimento III (A* vs. Dijkstra):**
   * Realiza um comparativo direto de eficiência entre o $A^*$ (com heurística de distância euclidiana) e o Algoritmo de Dijkstra.
   * Analisa o ganho expressivo de desempenho proporcionado pela poda de nós em redes de alta densidade.

---

## 🔍 Algoritmos Implementados
* **Busca em Largura (BFS)**
* **Busca em Profundidade (DFS)**
* **Busca Gulosa (Best-First)**
* **Algoritmo $A^*$**
* **Hill Climbing Estrito (sem backtracking)**
* **Hill Climbing Não-Estrito (sem backtracking)**
* **Algoritmo de Dijkstra**
