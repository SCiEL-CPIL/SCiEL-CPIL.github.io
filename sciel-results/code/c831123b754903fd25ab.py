<<<<<<< SEARCH
def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
=======
def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _precompute_subsets(n, sizes):
    """Precompute masks for subsets of given sizes."""
    result = {}
    for size in sizes:
        subsets = []
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets.append((comb, mask))
        result[size] = subsets
    return result
>>>>>>> REPLACE

