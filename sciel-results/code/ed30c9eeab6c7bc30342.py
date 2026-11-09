def contains_q3_with_edge(u, v, graph_edges):
    other_vertices = [x for x in range(16) if x != u and x != v]
    for combo in itertools.combinations(other_vertices, 6):
        subset = (u, v) + combo
        # Check degrees
        degs = {x: 0 for x in subset}
        for a in subset:
            for b in subset:
                if a == b: continue
                if (a, b) in graph_edges or (b, a) in graph_edges:
                    degs[a] += 1
        if any(d != 3 for d in degs.values()):
            continue
        # Check bipartite
        # ... (same as before)
        return True
    return False
