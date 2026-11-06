"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        
        if not P:
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        pivot = max(P | X, key=lambda u: len(P & adj[u]))
        candidates = [v for v in P if v not in adj[pivot]]
        
        for v in candidates:
            bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _greedy_set_cover(cliques, deadline):
    """Greedy set cover: iteratively pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    n = max(max(c) for c in cliques) + 1
    uncovered = list(range(len(cliques)))
    covered_by = {i: False for i in range(len(cliques))}
    
    # Precompute which cliques each vertex belongs to
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    solution = []
    remaining = set(range(len(cliques)))
    
    while remaining and time.time() < deadline:
        best_v = None
        best_count = 0
        for v, clique_indices in vertex_to_cliques.items():
            count = sum(1 for i in clique_indices if i in remaining)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None:
            break
        
        solution.append(best_v)
        for i in vertex_to_cliques[best_v]:
            if i in remaining:
                remaining.discard(i)
    
    return sorted(solution)

def _local_search(solution, cliques, deadline, max_iters=500):
    """Local search: try removing vertices, then swap with others."""
    if not solution:
        return solution
    
    sol_set = set(solution)
    n = max(max(c) for c in cliques) + 1
    
    def is_valid(s):
        for clique in cliques:
            if not any(v in s for v in clique):
                return False
        return True
    
    def count_uncovered(s):
        count = 0
        for clique in cliques:
            if not any(v in s for v in clique):
                count += 1
        return count
    
    best = list(solution)
    best_uncovered = count_uncovered(best)
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Try removing each vertex
        for v in best[:]:
            new_sol = [x for x in best if x != v]
            if is_valid(new_sol):
                best = new_sol
                best_uncovered = 0
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove v, add w
        for v in best:
            for w in range(n):
                if w in best:
                    continue
                new_sol = [x for x in best if x != v]
                new_sol.append(w)
                if is_valid(new_sol):
                    best = new_sol
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return sorted(best)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Not enough time left; return empty for remaining
            results.append([])
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        # Greedy set cover
        sol = _greedy_set_cover(cliques, deadline - 5)
        
        # Local search to improve
        sol = _local_search(sol, cliques, deadline - 2)
        
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
