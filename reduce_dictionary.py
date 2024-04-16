import os
import json
import random
from graph_utils import load_graph_dict, get_num_vertices
from typing import Dict, List, Union, Tuple


def add_recursive(full_graph: Dict[int, List], reduced_graph: Dict[int, List], index : int) -> None:
    for dest in full_graph[index]:
        if dest not in reduced_graph.keys():
            reduced_graph[dest] = full_graph[dest]
            add_recursive(full_graph, reduced_graph, dest)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SP prob calc')

    parser.add_argument('--load_dir', type=str, default="graph_dir/",
                        help='path to dir where the results and graph are stored')
    parser.add_argument('--N', type=int,
                        default=1000,
                        help='Number of experiments to run')
    parser.add_argument('--n_cores', type=int,
                        default=12,
                        help='Num cores to use'
                        )
    parser.add_argument('--seed', type=int,
                        default=2,
                        help='random seed to use'
                        )

    args = parser.parse_args()

    graph = load_graph_dict(os.path.join(args.load_dir, "graph.json"))
    num_vertices = get_num_vertices(os.path.join(args.load_dir, "encoding_dict.json"))
    reduced_keys = random.sample(graph.keys(), 1000)
    reduced_graph = {key:graph[key] for key in reduced_keys}

    for v in reduced_keys:
        add_recursive(graph, reduced_graph, v)

    with open("russian_reduced.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)

    num_vertices = get_num_vertices(os.path.join(args.load_dir, "encoding_dict.json"))
