"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _generate_edges(n, p, seed):
    import random
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_cliques(n, adj):
    """Bron-Kerbosch with pivot to find all maximal cliques."""
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        if P:
            # Use first element of P as pivot for simplicity
            pivot = next(iter(P))
            candidates = [v for v in P if v not in adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    
    bronk(frozenset(), set(range(n)), frozenset())
    return cliques

def _solve_instance(n, p, seed, deadline):
    edges = _generate_edges(n, p, seed)
    adj = _build_adj(n, edges)
    
    # Find all maximal cliques
    cliques = _find_maximal_cliques(n, adj)
    
    # Convert edge to index mapping
    edge_set = set(edges)
    uncovered = set(edges)
    
    result = []
    
    # Greedy: pick clique covering most uncovered edges
    while uncovered and time.time() < deadline:
        best_clique = None
        best_count = 0
        for clique in cliques:
            cl = set(clique)
            # Count how many edges of this clique are uncovered
            count = 0
            for a, b in combinations(cl, 2):
                e = (min(a, b), max(a, b))
                if e in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_clique = cl
            elif count == best_count and best_clique is not None:
                # Prefer larger clique for tie-breaking
                if len(cl) > len(best_clique):
                    best_clique = cl
        
        if best_clique is None or best_count == 0:
            # No clique covers any uncovered edges; add remaining edges as 2-cliques
            for e in list(uncovered):
                result.append(list(e))
                uncovered.discard(e)
            break
        
        # Add this clique
        result.append(sorted(best_clique))
        # Remove its edges from uncovered
        for a, b in combinations(best_clique, 2):
            e = (min(a, b), max(a, b))
            uncovered.discard(e)
    
    # Any remaining edges
    for e in uncovered:
        result.append(list(e))
    
    return result

def construct():
    start = time.time()
    deadline = start + 140  # 140s deadline, 10s safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: just use edges as 2-cliques
            edges = _generate_edges(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        result = _solve_instance(n, p, seed, deadline)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
