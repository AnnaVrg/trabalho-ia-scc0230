import math
import random
from typing import Dict, List, Tuple

Coordenada = Tuple[float, float]
Adjacencia = Dict[int, List[Tuple[int, float]]]


class RedeAleatoria:
    def __init__(self, n: int, p: float, seed: int = None):
        """Inicializa e gera uma Rede Aleatória.

        :param n: Número de vértices (define também o limite [0, n] do plano xy)
        :param p: Probabilidade [0, 1] de existir uma aresta entre qualquer par
        de vértices
        :param seed: Semente para reprodutibilidade dos experimentos
        """
        if seed is not None:
            random.seed(seed)

        self.n = n
        self.p = p
        self.vertices: Dict[int, Coordenada] = {}
        self.adj: Adjacencia = {i: [] for i in range(n)}
        self.num_arestas = 0

        self._gera_rede_aleatoria()

    @staticmethod
    def distancia_geometrica(c1: Coordenada, c2: Coordenada) -> float:
        """Calcula a distância euclidiana entre duas coordenadas."""
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _gera_vertices(self) -> None:
        """Gera n vértices com coordenadas x e y aleatórias entre 0 e n."""
        for i in range(self.n):
            x = random.uniform(0, self.n)
            y = random.uniform(0, self.n)
            self.vertices[i] = (x, y)

    def _gera_arestas(self) -> None:
        """Para cada par de vértices, gera um valor aleatório entre 0 e 1.

        Se for menor do que p, conecta os vértices com peso igual à distância
        geométrica.
        """
        for i in range(self.n):
            c_i = self.vertices[i]
            for j in range(i + 1, self.n):
                if random.random() < self.p:
                    c_j = self.vertices[j]
                    peso = self.distancia_geometrica(c_i, c_j)
                    self.adj[i].append((j, peso))
                    self.adj[j].append((i, peso))
                    self.num_arestas += 1

    def _gera_rede_aleatoria(self) -> None:
        self._gera_vertices()
        self._gera_arestas()

    def heuristica(self, u: int, destino: int) -> float:
        """Distância geométrica direta até o destino (admissível e consistente)."""
        return self.distancia_geometrica(self.vertices[u], self.vertices[destino])
