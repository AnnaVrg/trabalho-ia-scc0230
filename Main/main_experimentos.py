from pathlib import Path
import random
import statistics
from algoritmos_busca import (
    a_estrela,
    best_first,
    busca_em_largura,
    busca_em_profundidade,
    dijkstra,
    hill_climbing_estrito,
    hill_climbing_nao_estrito,
)
import matplotlib.pyplot as plt
import numpy as np
from rede_aleatoria import RedeAleatoria

PASTA_BASE = Path("graficos")
PASTA_EXP1 = PASTA_BASE / "exp1"
PASTA_EXP2 = PASTA_BASE / "exp2"
PASTA_EXP3 = PASTA_BASE / "exp3"

for pasta in (PASTA_EXP1, PASTA_EXP2, PASTA_EXP3):
    pasta.mkdir(parents=True, exist_ok=True)

ALGORITMOS = [
    ("DFS", "dfs", busca_em_profundidade),
    ("BFS", "bfs", busca_em_largura),
    ("Best-First", "best_first", best_first),
    ("A*", "a_estrela", a_estrela),
    ("HC Estrito", "hc_estrito", hill_climbing_estrito),
    ("HC Não-Estrito", "hc_nao_estrito", hill_climbing_nao_estrito),
    ("Dijkstra", "dijkstra", dijkstra),
]


def escolher_par_conectado(rede: RedeAleatoria):
    """Sorteia um par (origem, destino) garantindo que exista caminho entre eles."""
    vertices = list(rede.vertices.keys())
    for _ in range(200):
        u, v = random.sample(vertices, 2)
        res = busca_em_largura(rede, u, v)
        if res.sucesso and len(res.caminho) >= 3:
            return u, v
    validos = [x for x in vertices if len(rede.adj[x]) > 0]
    return random.sample(validos, 2) if len(validos) >= 2 else (0, min(1, rede.n - 1))


def plotar_grafo_individual(
    rede: RedeAleatoria,
    origem: int,
    destino: int,
    res,
    nome_exibicao: str,
    nome_arquivo: str,
):
    """Gera e guarda um ficheiro PNG individual na subpasta exp1/."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Arestas do grafo ao fundo
    for u in range(rede.n):
        x1, y1 = rede.vertices[u]
        for v, _ in rede.adj[u]:
            if u < v:
                x2, y2 = rede.vertices[v]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="#e6e6e6",
                    linewidth=0.6,
                    zorder=1,
                )

    # 2. Vértices não visitados
    xs = [rede.vertices[i][0] for i in range(rede.n)]
    ys = [rede.vertices[i][1] for i in range(rede.n)]
    ax.scatter(xs, ys, color="#cccccc", s=18, zorder=2, label="Não visitados")

    # 3. Vértices explorados (Espaço de busca)
    exp_x = [rede.vertices[i][0] for i in res.explorados]
    exp_y = [rede.vertices[i][1] for i in res.explorados]
    ax.scatter(
        exp_x,
        exp_y,
        color="#f39c12",
        s=35,
        zorder=3,
        label=f"Explorados ({len(res.explorados)})",
    )

    # 4. Caminho solução final (ou tentativa até parar)
    if res.caminho:
        cam_x = [rede.vertices[i][0] for i in res.caminho]
        cam_y = [rede.vertices[i][1] for i in res.caminho]
        cor_cam = "#2980b9" if res.sucesso else "#e74c3c"
        estilo_cam = "-" if res.sucesso else "--"
        label_cam = (
            f"Caminho ({len(res.caminho)} nós)"
            if res.sucesso
            else f"Trajeto até falha ({len(res.caminho)} nós)"
        )
        ax.plot(
            cam_x,
            cam_y,
            color=cor_cam,
            linestyle=estilo_cam,
            linewidth=2.5,
            zorder=4,
            label=label_cam,
        )

    # 5. Origem e Destino
    ax.scatter(
        *rede.vertices[origem],
        color="#27ae60",
        s=100,
        zorder=5,
        label=f"Origem ({origem})",
    )
    ax.scatter(
        *rede.vertices[destino],
        color="#c0392b",
        s=100,
        zorder=5,
        label=f"Destino ({destino})",
    )

    dist_str = f"{res.distancia_total:.2f}" if res.sucesso else "Falhou (Inf)"
    ax.set_title(
        f"{nome_exibicao} — Rede Aleatória (n={rede.n}, p={rede.p * 100:.1f}%)\n"
        f"Distância Final: {dist_str} | Vértices Explorados:"
        f" {len(res.explorados)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, rede.n)
    ax.set_ylim(0, rede.n)

    plt.tight_layout()
    caminho_png = PASTA_EXP1 / f"exp1_rede_aleatoria_{nome_arquivo}.png"
    plt.savefig(caminho_png, dpi=200)
    plt.close(fig)
    print(f"   -> Imagem guardada em '{caminho_png}'\n")


def executar_experimento_1():
    print("=" * 78)
    print("EXPERIMENTO I: Rede Aleatória Reduzida (n=200, p=2.5%) e Visualizações")
    print("=" * 78)

    rede = RedeAleatoria(n=200, p=0.025, seed=42)
    origem, destino = escolher_par_conectado(rede)
    grau_esperado = (rede.n - 1) * rede.p
    grau_real = (2 * rede.num_arestas) / rede.n
    print(f"Rede gerada: n={rede.n}, p={rede.p * 100:.1f}%")
    print(
        f"Arestas: {rede.num_arestas} | Grau médio esperado:"
        f" {grau_esperado:.2f} | Grau médio real: {grau_real:.2f}"
    )
    print(f"Par selecionado: Origem={origem} -> Destino={destino}\n")

    for nome_exibicao, sufixo_arq, func in ALGORITMOS:
        res = func(rede, origem, destino)
        status = "Sucesso" if res.sucesso else "Falhou (Mínimo Local / Beco sem saída)"
        print(
            f"[{nome_exibicao}] Status: {status} | Distância Final:"
            f" {res.distancia_total:.2f} | Nós explorados:"
            f" {len(res.explorados)} | Tempo: {res.tempo_segundos * 1000:.3f} ms"
        )
        print(f"   Caminho ({len(res.caminho)} vértices): {res.caminho}")

        plotar_grafo_individual(rede, origem, destino, res, nome_exibicao, sufixo_arq)


def plotar_graficos_exp2_e_exp3(historico: dict, configuracoes_p: list):
    """Gera gráficos separados nas subpastas exp2/ e exp3/."""
    nomes_alg = [nome for nome, _, _ in ALGORITMOS]
    cores = [
        "#8e44ad",
        "#2980b9",
        "#16a085",
        "#27ae60",
        "#e67e22",
        "#d35400",
        "#c0392b",
    ]

    for p in configuracoes_p:
        p_str = str(p).replace(".", "_")
        dados_p = historico[p]

        # ---------------------------------------------------------------------
        # 1A. Experimento II: Distância Média (COM DFS - Escala Logarítmica)
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5.5))
        medias_d = [dados_p[alg]["media_d"] for alg in nomes_alg]

        barras = ax.bar(nomes_alg, medias_d, color=cores, alpha=0.85, edgecolor="black")
        ax.set_yscale("log")
        ax.set_title(
            f"Experimento II: Distância Média em Escala Log (n=2500, p={p * 100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Distância Geométrica Média (Escala Log)")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.5)

        for barra, m in zip(barras, medias_d):
            if m > 0:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    m * 1.15,
                    f"{m:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_ylim(bottom=100, top=max(medias_d) * 4)
        plt.tight_layout()
        arq_dist_log = PASTA_EXP2 / f"exp2_distancia_log_p_{p_str}.png"
        plt.savefig(arq_dist_log, dpi=200)
        plt.close(fig)
        print(f"-> Gráfico guardado em '{arq_dist_log}'")

        # ---------------------------------------------------------------------
        # 1B. Experimento II: Distância Média (SEM DFS - Escala Linear + DP)
        # ---------------------------------------------------------------------
        nomes_sem_dfs = [n for n in nomes_alg if n != "DFS"]
        cores_sem_dfs = cores[1:]
        medias_d_sem_dfs = [dados_p[alg]["media_d"] for alg in nomes_sem_dfs]
        dps_d_sem_dfs = [dados_p[alg]["dp_d"] for alg in nomes_sem_dfs]

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        barras = ax.bar(
            nomes_sem_dfs,
            medias_d_sem_dfs,
            yerr=dps_d_sem_dfs,
            capsize=5,
            color=cores_sem_dfs,
            alpha=0.85,
            edgecolor="black",
        )
        dfs_m = dados_p["DFS"]["media_d"]
        dfs_dp = dados_p["DFS"]["dp_d"]
        ax.set_title(
            "Experimento II: Distância Média — Sem DFS (n=2500,"
            f" p={p * 100:.1f}%)\n[Nota: DFS obteve {dfs_m:.1f} ± {dfs_dp:.1f}]",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_ylabel("Distância Geométrica Média (± DP)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for barra, m, dp in zip(barras, medias_d_sem_dfs, dps_d_sem_dfs):
            if m > 0:
                ax.text(
                    barra.get_x() + barra.get_width() / 2,
                    m + dp + (max(medias_d_sem_dfs) * 0.02),
                    f"{m:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_ylim(0, (max(medias_d_sem_dfs) + max(dps_d_sem_dfs, default=0)) * 1.18)
        plt.tight_layout()
        arq_dist_sem_dfs = PASTA_EXP2 / f"exp2_distancia_sem_dfs_p_{p_str}.png"
        plt.savefig(arq_dist_sem_dfs, dpi=200)
        plt.close(fig)
        print(f"-> Gráfico guardado em '{arq_dist_sem_dfs}'")

        # ---------------------------------------------------------------------
        # 1C. Experimento II: Tempo Médio ± Desvio Padrão
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5.5))
        medias_t = [dados_p[alg]["media_t"] for alg in nomes_alg]
        dps_t = [dados_p[alg]["dp_t"] for alg in nomes_alg]

        barras = ax.bar(
            nomes_alg,
            medias_t,
            yerr=dps_t,
            capsize=5,
            color=cores,
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_title(
            f"Experimento II: Tempo Médio de Execução (n=2500, p={p * 100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Tempo Médio em ms (± DP)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for barra, m, dp in zip(barras, medias_t, dps_t):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                m + dp + (max(medias_t) * 0.02),
                f"{m:.2f}ms",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_ylim(0, (max(medias_t) + max(dps_t, default=0)) * 1.18)
        plt.tight_layout()
        arq_tempo = PASTA_EXP2 / f"exp2_tempo_p_{p_str}.png"
        plt.savefig(arq_tempo, dpi=200)
        plt.close(fig)
        print(f"-> Gráfico guardado em '{arq_tempo}'")

    # -------------------------------------------------------------------------
    # 2. Gráficos Comparativos do Experimento III (A* vs Dijkstra) -> exp3/
    # -------------------------------------------------------------------------
    labels_p = [f"p = {p * 100:.1f}%" for p in configuracoes_p]
    x = np.arange(len(configuracoes_p))
    largura = 0.35

    # 2A. Tempo de Execução (A* vs Dijkstra)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    t_a = [historico[p]["A*"]["media_t"] for p in configuracoes_p]
    dp_t_a = [historico[p]["A*"]["dp_t"] for p in configuracoes_p]
    t_dijk = [historico[p]["Dijkstra"]["media_t"] for p in configuracoes_p]
    dp_t_dijk = [historico[p]["Dijkstra"]["dp_t"] for p in configuracoes_p]

    ax.bar(
        x - largura / 2,
        t_a,
        largura,
        yerr=dp_t_a,
        capsize=5,
        label="A*",
        color="#27ae60",
        edgecolor="black",
    )
    ax.bar(
        x + largura / 2,
        t_dijk,
        largura,
        yerr=dp_t_dijk,
        capsize=5,
        label="Dijkstra",
        color="#c0392b",
        edgecolor="black",
    )

    ax.set_title(
        "Experimento III: Tempo Médio de Execução — A* vs. Dijkstra (n=2500)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_p)
    ax.set_ylabel("Tempo Médio em ms (± DP)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    arq_exp3_t = PASTA_EXP3 / "exp3_a_estrela_vs_dijkstra_tempo.png"
    plt.savefig(arq_exp3_t, dpi=200)
    plt.close(fig)
    print(f"-> Gráfico guardado em '{arq_exp3_t}'")

    # 2B. Vértices Explorados (A* vs Dijkstra)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    exp_a = [historico[p]["A*"]["media_exp"] for p in configuracoes_p]
    exp_dijk = [historico[p]["Dijkstra"]["media_exp"] for p in configuracoes_p]

    ax.bar(
        x - largura / 2,
        exp_a,
        largura,
        label="A*",
        color="#27ae60",
        edgecolor="black",
    )
    ax.bar(
        x + largura / 2,
        exp_dijk,
        largura,
        label="Dijkstra",
        color="#c0392b",
        edgecolor="black",
    )

    ax.set_title(
        "Experimento III: Média de Vértices Explorados — A* vs. Dijkstra (n=2500)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_p)
    ax.set_ylabel("Número Médio de Vértices Expandidos")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    arq_exp3_n = PASTA_EXP3 / "exp3_a_estrela_vs_dijkstra_nos.png"
    plt.savefig(arq_exp3_n, dpi=200)
    plt.close(fig)
    print(f"-> Gráfico guardado em '{arq_exp3_n}'")


def executar_experimentos_2_e_3():
    print("=" * 78)
    print(
        "EXPERIMENTOS II e III: Rede Aleatória n=2500, p in {5%, 2.5%, 1%} (10 pares)"
    )
    print("=" * 78)

    configuracoes_p = [0.05, 0.025, 0.01]
    num_pares = 10
    historico = {}

    for p in configuracoes_p:
        historico[p] = {}
        print(f"\n---> A gerar Rede Aleatória (n=2500, p={p * 100:.1f}%)...")
        rede = RedeAleatoria(n=2500, p=p, seed=100)
        grau_esperado = (rede.n - 1) * p
        grau_real = (2 * rede.num_arestas) / rede.n
        print(
            f"     Arestas: {rede.num_arestas} | Grau médio esperado:"
            f" {grau_esperado:.2f} | Grau médio real: {grau_real:.2f}"
        )

        pares = [escolher_par_conectado(rede) for _ in range(num_pares)]

        print(
            f"{'Algoritmo':<15} | {'Sucessos':<8} | {'Dist. Média ± DP':<22} |"
            f" {'Tempo Médio ± DP (ms)':<24} | {'Nós Exp. (Média)'}"
        )
        print("-" * 92)

        for nome_exibicao, _, func in ALGORITMOS:
            distancias = []
            tempos_ms = []
            explorados_cont = []
            sucessos = 0

            for u, v in pares:
                res = func(rede, u, v)
                tempos_ms.append(res.tempo_segundos * 1000.0)
                explorados_cont.append(len(res.explorados))
                if res.sucesso:
                    sucessos += 1
                    distancias.append(res.distancia_total)

            if distancias:
                media_d = statistics.mean(distancias)
                dp_d = statistics.stdev(distancias) if len(distancias) > 1 else 0.0
                dist_str = f"{media_d:8.2f} ± {dp_d:6.2f}"
            else:
                media_d = 0.0
                dp_d = 0.0
                dist_str = "   N/A (0 sucessos)"

            media_t = statistics.mean(tempos_ms)
            dp_t = statistics.stdev(tempos_ms) if len(tempos_ms) > 1 else 0.0
            media_exp = statistics.mean(explorados_cont)

            historico[p][nome_exibicao] = {
                "sucessos": sucessos,
                "media_d": media_d,
                "dp_d": dp_d,
                "media_t": media_t,
                "dp_t": dp_t,
                "media_exp": media_exp,
            }

            print(
                f"{nome_exibicao:<15} | {sucessos:>2}/{num_pares:<5} |"
                f" {dist_str:<22} | {media_t:8.3f} ± {dp_t:6.3f} ms    |"
                f" {media_exp:8.1f}"
            )

    print("\n" + "=" * 78)
    print("Gera os gráficos nas pastas (graficos/exp2 e graficos/exp3)")
    print("=" * 78)
    plotar_graficos_exp2_e_exp3(historico, configuracoes_p)


if __name__ == "__main__":
    executar_experimento_1()
    executar_experimentos_2_e_3()
