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

def _greedy_set_cover_bitmask(cliques, n, deadline):
    """Bitmask-accelerated greedy set cover."""
    if not cliques:
        return []
    
    m = len(cliques)
    # Build bitmask for each vertex: which cliques it covers
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    full_mask = (1 << m) - 1
    remaining = full_mask
    solution = []
    
    while remaining and time.time() < deadline:
        best_v = -1
        best_count = 0
        for v in range(n):
            if vertex_mask[v] == 0:
                continue
            count = bin(remaining & vertex_mask[v]).count('1')
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v < 0:
            break
        
        solution.append(best_v)
        remaining &= ~vertex_mask[best_v]
    
    return sorted(solution)

def _local_search_bitmask(solution, cliques, n, deadline):
    """Local search with bitmask: try removals and 1-1 swaps."""
    if not solution:
        return solution
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    # Precompute vertex masks
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    def is_valid(sol):
        covered = 0
        for v in sol:
            covered |= vertex_mask[v]
        return covered == full_mask
    
    def covered_mask(sol):
        c = 0
        for v in sol:
            c |= vertex_mask[v]
        return c
    
    best = list(solution)
    best_cm = covered_mask(best)
    
    for _ in range(2000):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Try removing each vertex
        for idx, v in enumerate(best):
            new_cm = 0
            for j, u in enumerate(best):
                if j != idx:
                    new_cm |= vertex_mask[u]
            if new_cm == full_mask:
                best.pop(idx)
                best_cm = new_cm
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-1 swap: remove v, add w
        found_swap = False
        for idx, v in enumerate(best):
            rest_cm = 0
            for j, u in enumerate(best):
                if j != idx:
                    rest_cm |= vertex_mask[u]
            # Need w such that rest_cm | vertex_mask[w] == full_mask
            missing = full_mask & ~rest_cm
            if missing == 0:
                # v is redundant (shouldn't happen after removal pass)
                continue
            for w in range(n):
                if w in best:
                    continue
                if (rest_cm | vertex_mask[w]) == full_mask:
                    best[idx] = w
                    best_cm = rest_cm | vertex_mask[w]
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if not improved:
            break
    
    return sorted(best)

def _shrink_polish(solution, cliques, n, deadline, rng):
    """Try multiple random removal orders to escape local minima."""
    if not solution:
        return solution
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    def covered(sol):
        c = 0
        for v in sol:
            c |= vertex_mask[v]
        return c
    
    best = list(solution)
    best_len = len(best)
    
    # Multiple random removal passes
    for trial in range(50):
        if time.time() >= deadline:
            break
        
        cur = list(solution)
        # Shuffle order for removal attempts
        order = list(range(len(cur)))
        rng.shuffle(order)
        
        for idx in order:
            if idx >= len(cur):
                continue
            v = cur[idx]
            new_sol = [x for i, x in enumerate(cur) if i != idx]
            if covered(new_sol) == full_mask:
                cur = new_sol
        
        if len(cur) < best_len:
            best = cur
            best_len = len(cur)
    
    return sorted(best)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Not enough time left; return empty for remaining
            results.append([])
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        # Greedy set cover with bitmask
        sol = _greedy_set_cover_bitmask(cliques, n, deadline - 5)
        
        # Local search with bitmask
        sol = _local_search_bitmask(sol, cliques, n, deadline - 2)
        
        # Shrink polish: multiple random removal orders
        rng = random.Random(seed + 999)
        sol = _shrink_polish(sol, cliques, n, deadline - 1, rng)
        
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
