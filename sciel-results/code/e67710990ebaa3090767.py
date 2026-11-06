"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_regular_subgraph(adj, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    k = len(subset)
    if k < 6:
        return False
    idx_map = {v: i for i, v in enumerate(subset)}
    degrees = [0] * k
    for i, u in enumerate(subset):
        for v in subset:
            if v != u and adj[u][v]:
                degrees[i] += 1
    return all(d == degrees[0] for d in degrees)

def _check_validity(adj, n):
    """Check if graph has no regular induced subgraph on 6+ vertices."""
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_subgraph(adj, subset):
                return False, subset
    return True, None

def _build_adj(edges, n):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _random_graph(n, p=0.5):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append([i, j])
    return edges

def _try_extend(adj, n, new_vertex, deadline):
    """Try to add a new vertex with various edge patterns."""
    best_edges = None
    best_score = -1
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        new_edges = []
        for i in range(n):
            if mask & (1 << i):
                new_edges.append([new_vertex, i])
        # Build new adjacency
        new_n = n + 1
        new_adj = [[False] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
        for i in range(n):
            if mask & (1 << i):
                new_adj[new_vertex][i] = True
                new_adj[i][new_vertex] = True
        # Quick check: only need to check subsets involving new_vertex
        valid = True
        for size in range(6, new_n + 1):
            for subset in combinations(range(new_n), size):
                if new_vertex not in subset:
                    continue
                if _is_regular_subgraph(new_adj, subset):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return new_edges, new_adj
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to construct a 20-vertex valid graph
    random.seed(42)
    
    # First, try known good structures - use a random search for n=20
    best_n = 5
    best_edges = []
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        n = 20
        p = 0.5 + random.uniform(-0.2, 0.2)
        edges = _random_graph(n, p)
        adj = _build_adj(edges, n)
        valid, _ = _check_validity(adj, n)
        if valid:
            best_n = n
            best_edges = edges
            break
    
    # If we found a 20-vertex graph, try to extend to 21
    if best_n == 20:
        adj = _build_adj(best_edges, 20)
        result = _try_extend(adj, 20, 20, deadline - 5)
        if result[0] is not None:
            new_edges, new_adj = result
            best_n = 21
            best_edges = best_edges + new_edges
    
    if best_n < 10:
        # Fallback: return a small valid graph
        best_n = 5
        best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
