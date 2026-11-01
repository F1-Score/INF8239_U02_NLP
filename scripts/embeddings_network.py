from __future__ import annotations

import argparse
import json
import re

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset


def tokenize(value: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(value).lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default="servicio")
    parser.add_argument("--network-demo", action="store_true")
    parser.add_argument("--window", type=int, default=5, help="Tamaño de la ventana de contexto (base: 5)")
    parser.add_argument("--min-count", type=int, default=1, help="Frecuencia mínima para entrar al vocabulario (base: 1)")
    args = parser.parse_args()
    df = load_dataset()
    sentences = [tokenize(value) for value in df[settings.text_column].dropna()]
    model = Word2Vec(sentences, vector_size=60, window=args.window, min_count=args.min_count, workers=1, seed=42, epochs=30)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    model.save(str(ROOT / "models/word2vec.model"))
    tokens = [token for sentence in sentences for token in sentence]
    coverage = sum(token in model.wv for token in tokens) / max(len(tokens), 1)
    print(f"Configuración: window={args.window}, min_count={args.min_count}")
    print("Vocabulario:", len(model.wv), "Cobertura:", round(coverage, 3))
    neighbors = []
    if args.word in model.wv:
        neighbors = model.wv.most_similar(args.word, topn=min(10, len(model.wv) - 1))
        print("Vecinos:", [(w, round(s, 3)) for w, s in neighbors])
    else:
        print(f"La palabra '{args.word}' no está en el vocabulario")
    run = {
        "word": args.word, "window": args.window, "min_count": args.min_count,
        "vocabulary": len(model.wv), "tokens": len(tokens), "coverage": round(coverage, 4),
        "neighbors": [{"word": w, "similarity": round(float(s), 4)} for w, s in neighbors],
    }
    run_file = ROOT / f"reports/word2vec_{args.word}_w{args.window}_mc{args.min_count}.json"
    run_file.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Resultado guardado en", run_file.relative_to(ROOT))
    if not args.network_demo:
        return
    graph = nx.karate_club_graph()
    metrics = pd.DataFrame({
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph),
        "pagerank": nx.pagerank(graph),
    })
    metrics.index.name = "node"
    metrics.to_csv(ROOT / "reports/centralities.csv")
    communities = list(nx.community.greedy_modularity_communities(graph))
    community = {node: index for index, group in enumerate(communities) for node in group}
    print("Nodos:", graph.number_of_nodes(), "Aristas:", graph.number_of_edges())
    print("Comunidades:", len(communities), "Modularidad:", round(nx.community.modularity(graph, communities), 3))
    palette = ["#A593D3", "#A8D8C2", "#9EC3E6", "#E6C8DD", "#D9D4C7"]
    borders = ["#6B58A6", "#4F9478", "#2E4F73", "#9C5F8A", "#7A7566"]
    position = nx.spring_layout(graph, seed=3, k=0.55, iterations=200)
    size = [220 + 1500 * metrics.loc[node, "degree"] for node in graph]
    fig, ax = plt.subplots(figsize=(8, 6.2), facecolor="white")
    nx.draw_networkx_edges(graph, position, ax=ax, edge_color="#C9C7BE", width=0.9)
    nx.draw_networkx_nodes(graph, position, ax=ax, node_size=size,
                           node_color=[palette[community[n] % 5] for n in graph],
                           edgecolors=[borders[community[n] % 5] for n in graph], linewidths=1.2)
    nx.draw_networkx_labels(graph, position, ax=ax, font_size=8, font_color="#0B0B0B")
    for index in range(len(communities)):
        ax.scatter([], [], s=90, color=palette[index % 5], edgecolors=borders[index % 5], label=f"Comunidad {index + 1} ({len(communities[index])} nodos)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title("Red del club de karate de Zachary · comunidades (tamaño = centralidad de grado)", loc="left", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    nx.write_graphml(graph, ROOT / "reports/social_network.graphml")


if __name__ == "__main__":
    main()
