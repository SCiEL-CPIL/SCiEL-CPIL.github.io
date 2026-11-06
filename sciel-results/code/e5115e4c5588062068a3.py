"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _bron_kerbosch(adj, R, P, X, cliques):
    """Bron-Kerbosch with pivoting to enumerate maximal cliques."""
    if not P and not X:
        if len(R) >= 2:
            cliques.append(frozenset(R))
        return
    
    # Choose pivot: vertex in P ∪ X with max |P ∩ adj[v]|
    P_union_X = P | X
    if P_union_X:
        # Pick vertex with max neighbors in P
        pivot = max(P_union_X, key=lambda v: len(P & adj[v]))
        candidates = P - adj[pivot]
    else:
        candidates = P
    
    for v in list(candidates):
        _bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
        P.remove(v)
        X.add(v)

def _generate_graph(n, p, seed):
    """Generate random graph and return adjacency sets."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _find_min_transversal(adj, n, max_cliques, time_limit=10):
    """Find a small vertex set that hits every maximal clique of size >= 2."""
    # Convert frozensets to lists for easier handling
    clique_list = [list(c) for c in max_cliques]
    if not clique_list:
        return []
    
    # Greedy set cover: repeatedly pick vertex covering most uncovered cliques
    uncovered = list(range(len(clique_list)))
    result = set()
    
    # Precompute for each vertex, which cliques it belongs to
    vertex_cliques = {v: [] for v in range(n)}
    for i, clique in enumerate(clique_list):
        for v in clique:
            vertex_cliques[v].append(i)
    
    while uncovered:
        # Find vertex that covers most uncovered cliques
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in result:
                continue
            count = sum(1 for ci in vertex_cliques[v] if ci in uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1 or best_count == 0:
            break
        
        result.add(best_v)
        # Remove all cliques containing best_v
        uncovered = [ci for ci in uncovered if best_v not in clique_list[ci]]
    
    # Try to improve by removing vertices that aren't needed
    improved = True
    while improved:
        improved = False
        for v in list(result):
            # Check if removing v still covers all cliques
            test_result = result - {v}
            all_covered = True
            for clique in clique_list:
                if not any(x in test_result for x in clique):
                    all_covered = False
                    break
            if all_covered:
                result = test_result
                improved = True
    
    return sorted(result)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Fallback: use all vertices
            out.append(list(range(n)))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = []
        _bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        # Find transversal
        transversal = _find_min_transversal(adj, n, cliques)
        out.append(transversal)
    
    return out
# EVOLVE-BLOCK-END
