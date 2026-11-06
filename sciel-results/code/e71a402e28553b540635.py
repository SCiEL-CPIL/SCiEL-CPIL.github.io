"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _max_book(edges, n=40):
    """Compute the maximum book size for a given edge list."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    max_b = 0
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = len(adj[u] & adj[v])
                if common > max_b:
                    max_b = common
    return max_b

def _random_graph(n=40, m=401):
    """Generate a random graph with n vertices and m edges."""
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    return all_edges[:m]

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_edges = None
    best_score = float('inf')
    
    # Try many random graphs
    while time.time() < deadline:
        edges = _random_graph(40, 401)
        score = _max_book(edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            if best_score <= 7:
                break  # Theoretical lower bound is 7
    
    # Fallback: return a valid graph if none found
    if best_edges is None:
        best_edges = _random_graph(40, 401)
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
