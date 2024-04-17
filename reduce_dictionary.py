import os
import json
import random
from graph_utils import load_graph_dict, get_num_vertices

import collections



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SP prob calc')

    parser.add_argument('--load_dir', type=str, default="graph_dir/",
                        help='path to dir where the results and graph are stored')
    parser.add_argument('--N', type=int,
                        default=1000,
                        help='Max number of words in reduced dictionary')
    parser.add_argument('--seed', type=int,
                        default=2,
                        help='random seed to use'
                        )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Load graph from given directory
    graph = load_graph_dict(os.path.join(args.load_dir, "graph.json"))
    num_vertices = get_num_vertices(os.path.join(args.load_dir, "encoding_dict.json"))

    # Initialize sets

    graph_vertices = set(graph.keys())
    visited = set()

    # Depth-first search.  Done because we want to prioritize deeper words, i.e. closer to primes, rather than
    # More "higher" level words
    # Adapted from https://www.geeksforgeeks.org/iterative-depth-first-traversal/

    # While we haven't reached the total yet, perform a new depth-first search
    while len(visited) < args.N:
        # Choose a new random vertex to start depth-searching from
        root = random.choice(list(graph_vertices - visited))

        # Create a stack for DFS
        stack = collections.deque(root)

        while (len(stack)) and len(visited) < args.N:
            # Pop a vertex from stack
            v = stack[-1]
            stack.pop()

            # Stack may contain same vertex twice. So
            # we need to print the popped item only
            # if it is not visited.
            if v not in visited:
                visited.add(v)

            # Get all adjacent vertices of the popped vertex s
            # If a adjacent has not been visited, then push it
            # to the stack.
            for neighbor in graph[v]:
                if neighbor not in visited:
                    stack.append(neighbor)

    # Save resulting dictionary
    with open("graph_reduced.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
