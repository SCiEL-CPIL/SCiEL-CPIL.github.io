import time
from random import Random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        if time.time() > deadline:
            return []
            
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C and I
        # C is 0..k-1, I is k..n-1
        # adj_C[u] = set of neighbors in C
        # adj_I[v] = set of neighbors in C (or just check edges)
        
        # We know C is a clique.
        # We need to cover cross edges.
        # Strategy:
        # 1. One clique for C: [0, ..., k-1]
        # 2. For each u in C, if it has neighbors in I, form a clique [u] + neighbors_in_I(u)
        
        # Check if u has neighbors in I
        # We can build a list of neighbors in I for each u in C.
        
        neighbors_in_I = [set() for _ in range(k)]
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_I[u].add(v)
            elif u >= k and v < k:
                neighbors_in_I[v].add(u)
            # C-C edges are ignored for the cross-cliques construction
            # I-I edges don't exist
        
        cliques = []
        
        # 1. The main clique C
        if k >= 2:
            cliques.append(list(range(k)))
        elif k == 1:
            # If k=1, C is a single vertex. It's not a clique of size >= 2.
            # But wait, if k=1, there are no C-C edges.
            # The problem states "clique on the first k vertices".
            # If k=1, the "clique" is size 1.
            # The task says "Every listed set must be a clique with at least two vertices".
            # So if k=1, we cannot list C as a clique.
            # But are there C-C edges? No.
            # So we don't need to cover any C-C edges.
            # We only need to cover cross edges.
            # So if k=1, we just do the cross cliques.
            pass
        else: # k == 0
            pass
            
        # 2. Cross cliques
        for u in range(k):
            if neighbors_in_I[u]:
                # The set {u} U neighbors_in_I[u] is a clique.
                # u is connected to all in neighbors_in_I[u].
                # No edges within neighbors_in_I[u] (independent set).
                clique = [u] + list(neighbors_in_I[u])
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            # Ensure ints
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
