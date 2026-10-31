from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Dict, List, Optional, Set, Tuple
from rede_aleatoria import RedeAleatoria


@dataclass
class ResultadoBusca:
    algoritmo: str
    sucesso: bool
    caminho: List[int]
    distancia_total: float
    explorados: List[int]
    tempo_segundos: float


def _reconstruir_caminho(
    pais: Dict[int, Optional[int]],
    pesos_acumulados: Dict[int, float],
    destino: int,
) -> Tuple[List[int], float]:
    caminho = []
    atual: Optional[int] = destino
    while atual is not None:
        caminho.append(atual)
        atual = pais[atual]
    caminho.reverse()
    return caminho, pesos_acumulados[destino]


# 1. Busca em Largura (BFS)
def busca_em_largura(rede: RedeAleatoria, origem: int, destino: int) -> ResultadoBusca:
    inicio = time.perf_counter()
    fila = deque([origem])
    visitados: Set[int] = {origem}
    pais: Dict[int, Optional[int]] = {origem: None}
    distancias: Dict[int, float] = {origem: 0.0}
    explorados: List[int] = []

    while fila:
        atual = fila.popleft()
        explorados.append(atual)

        if atual == destino:
            caminho, dist = _reconstruir_caminho(pais, distancias, destino)
            return ResultadoBusca(
                "BFS",
                True,
                caminho,
                dist,
                explorados,
                time.perf_counter() - inicio,
            )

        for vizinho, peso in rede.adj[atual]:
            if vizinho not in visitados:
                visitados.add(vizinho)
                pais[vizinho] = atual
                distancias[vizinho] = distancias[atual] + peso
                fila.append(vizinho)

    return ResultadoBusca(
        "BFS", False, [], float("inf"), explorados, time.perf_counter() - inicio
    )


# 2. Busca em Profundidade (DFS)
def busca_em_profundidade(
    rede: RedeAleatoria, origem: int, destino: int
) -> ResultadoBusca:
    inicio = time.perf_counter()
    pilha = [origem]
    visitados: Set[int] = set()
    pais: Dict[int, Optional[int]] = {origem: None}
    distancias: Dict[int, float] = {origem: 0.0}
    explorados: List[int] = []

    while pilha:
        atual = pilha.pop()
        if atual in visitados:
            continue

        visitados.add(atual)
        explorados.append(atual)

        if atual == destino:
            caminho, dist = _reconstruir_caminho(pais, distancias, destino)
            return ResultadoBusca(
                "DFS",
                True,
                caminho,
                dist,
                explorados,
                time.perf_counter() - inicio,
            )

        for vizinho, peso in rede.adj[atual]:
            if vizinho not in visitados:
                pais[vizinho] = atual
                distancias[vizinho] = distancias[atual] + peso
                pilha.append(vizinho)

    return ResultadoBusca(
        "DFS", False, [], float("inf"), explorados, time.perf_counter() - inicio
    )


# 3. Algoritmo Best-First (Busca Gulosa)
def best_first(rede: RedeAleatoria, origem: int, destino: int) -> ResultadoBusca:
    inicio = time.perf_counter()
    pq = [(rede.heuristica(origem, destino), origem)]
    visitados: Set[int] = set()
    pais: Dict[int, Optional[int]] = {origem: None}
    distancias: Dict[int, float] = {origem: 0.0}
    explorados: List[int] = []

    while pq:
        _, atual = heapq.heappop(pq)
        if atual in visitados:
            continue

        visitados.add(atual)
        explorados.append(atual)

        if atual == destino:
            caminho, dist = _reconstruir_caminho(pais, distancias, destino)
            return ResultadoBusca(
                "Best-First",
                True,
                caminho,
                dist,
                explorados,
                time.perf_counter() - inicio,
            )

        for vizinho, peso in rede.adj[atual]:
            if vizinho not in visitados:
                if vizinho not in pais:
                    pais[vizinho] = atual
                    distancias[vizinho] = distancias[atual] + peso
                heapq.heappush(pq, (rede.heuristica(vizinho, destino), vizinho))

    return ResultadoBusca(
        "Best-First",
        False,
        [],
        float("inf"),
        explorados,
        time.perf_counter() - inicio,
    )


# 4. Algoritmo A*
def a_estrela(rede: RedeAleatoria, origem: int, destino: int) -> ResultadoBusca:
    inicio = time.perf_counter()
    pq = [(rede.heuristica(origem, destino), 0.0, origem)]
    visitados: Set[int] = set()
    pais: Dict[int, Optional[int]] = {origem: None}
    g_score: Dict[int, float] = {origem: 0.0}
    explorados: List[int] = []

    while pq:
        f_atual, g_atual, atual = heapq.heappop(pq)
        if atual in visitados:
            continue

        visitados.add(atual)
        explorados.append(atual)

        if atual == destino:
            caminho, dist = _reconstruir_caminho(pais, g_score, destino)
            return ResultadoBusca(
                "A*",
                True,
                caminho,
                dist,
                explorados,
                time.perf_counter() - inicio,
            )

        for vizinho, peso in rede.adj[atual]:
            if vizinho in visitados:
                continue
            tentativa_g = g_atual + peso
            if tentativa_g < g_score.get(vizinho, float("inf")):
                pais[vizinho] = atual
                g_score[vizinho] = tentativa_g
                f_score = tentativa_g + rede.heuristica(vizinho, destino)
                heapq.heappush(pq, (f_score, tentativa_g, vizinho))

    return ResultadoBusca(
        "A*", False, [], float("inf"), explorados, time.perf_counter() - inicio
    )


# 5. Algoritmo de Dijkstra
def dijkstra(rede: RedeAleatoria, origem: int, destino: int) -> ResultadoBusca:
    inicio = time.perf_counter()
    pq = [(0.0, origem)]
    visitados: Set[int] = set()
    pais: Dict[int, Optional[int]] = {origem: None}
    g_score: Dict[int, float] = {origem: 0.0}
    explorados: List[int] = []

    while pq:
        g_atual, atual = heapq.heappop(pq)
        if atual in visitados:
            continue

        visitados.add(atual)
        explorados.append(atual)

        if atual == destino:
            caminho, dist = _reconstruir_caminho(pais, g_score, destino)
            return ResultadoBusca(
                "Dijkstra",
                True,
                caminho,
                dist,
                explorados,
                time.perf_counter() - inicio,
            )

        for vizinho, peso in rede.adj[atual]:
            if vizinho in visitados:
                continue
            tentativa_g = g_atual + peso
            if tentativa_g < g_score.get(vizinho, float("inf")):
                pais[vizinho] = atual
                g_score[vizinho] = tentativa_g
                heapq.heappush(pq, (tentativa_g, vizinho))

    return ResultadoBusca(
        "Dijkstra",
        False,
        [],
        float("inf"),
        explorados,
        time.perf_counter() - inicio,
    )


# 6. Base do Hill Climbing (sem backtracking)
def _hill_climbing_base(
    rede: RedeAleatoria,
    origem: int,
    destino: int,
    estrito: bool,
    nome_alg: str,
) -> ResultadoBusca:
    inicio = time.perf_counter()
    atual = origem
    caminho = [atual]
    explorados = [atual]
    visitados: Set[int] = {atual}
    distancia_total = 0.0

    while atual != destino:
        h_atual = rede.heuristica(atual, destino)
        melhor_vizinho = None
        # Se estrito=True, exige h(vizinho) < h(atual). Se False, aceita o menor h(vizinho) disponível.
        melhor_h = h_atual if estrito else float("inf")
        peso_escolhido = 0.0

        for vizinho, peso in rede.adj[atual]:
            if vizinho not in visitados:
                h_viz = rede.heuristica(vizinho, destino)
                if h_viz < melhor_h:
                    melhor_h = h_viz
                    melhor_vizinho = vizinho
                    peso_escolhido = peso

        if melhor_vizinho is None:
            return ResultadoBusca(
                nome_alg,
                False,
                caminho,
                float("inf"),
                explorados,
                time.perf_counter() - inicio,
            )

        atual = melhor_vizinho
        visitados.add(atual)
        caminho.append(atual)
        explorados.append(atual)
        distancia_total += peso_escolhido

    return ResultadoBusca(
        nome_alg,
        True,
        caminho,
        distancia_total,
        explorados,
        time.perf_counter() - inicio,
    )


def hill_climbing_estrito(
    rede: RedeAleatoria, origem: int, destino: int
) -> ResultadoBusca:
    """Hill Climbing Estrito (Steepest Ascent):

    Só avança se h(vizinho) < h(atual). Para em mínimos locais.
    """
    return _hill_climbing_base(
        rede, origem, destino, estrito=True, nome_alg="HC Estrito"
    )


def hill_climbing_nao_estrito(
    rede: RedeAleatoria, origem: int, destino: int
) -> ResultadoBusca:
    """Hill Climbing Não-Estrito (Greedy Walk sem Backtracking):

    Avança sempre para o melhor vizinho não visitado, mesmo que h(vizinho) >=
    h(atual). Para apenas em becos sem saída (todos os vizinhos já visitados).
    """
    return _hill_climbing_base(
        rede, origem, destino, estrito=False, nome_alg="HC Não-Estrito"
    )
