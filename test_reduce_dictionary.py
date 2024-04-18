from reduce_dictionary import add_recursive


def test_add_recursive():
    full_graph = {0: [2, 3], 1: [], 2: [4], 3: [4, 5], 4: [], 5: [1], 6: [7], 7: []}
    reduced_keys = [0, 1]
    reduced_graph = {key:full_graph[key] for key in reduced_keys}
    add_recursive(full_graph, reduced_graph, 0)
    assert reduced_graph == {0: [2, 3], 1: [], 2: [4], 3: [4, 5], 4: [], 5: [1]}

    full_graph = {0: [2, 3], 1: [], 2: [4], 3: [4, 5], 4: [], 5: [1], 6: [7, 8], 7: [], 8: []}
    reduced_keys = [0, 1, 6]
    reduced_graph = {key:full_graph[key] for key in reduced_keys}
    for v in reduced_keys:
        add_recursive(full_graph, reduced_graph, v)
    print(reduced_graph, full_graph)
    assert reduced_graph == full_graph