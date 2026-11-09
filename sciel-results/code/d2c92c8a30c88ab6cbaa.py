"""Stronger algorithm for hitting all maximal cliques of size >= 2."""
import time
import itertools
from collections import defaultdict

# Deadline guard: must finish within 120s wall-clock per evaluation.
_DEADLINE = 115.0  # safety margin before 120s
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _build_graph(n, p, seed):
    """Build edge list and adjacency sets for random_graph(n, p, seed)."""
    import random
    rng = random.Random(seed)
    edges = []
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return edges, adj

def _bron_kerbosch_maximal_cliques(adj, n):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    # Convert to bitmasks for speed
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    def bk(R, P, X):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot u from P ∪ X to minimize branching
        Px = P | X
        if Px == 0:
            return
        # Pick pivot with max degree in P
        best_u = -1
        best_count = -1
        # Iterate bits in Px
        px = Px
        while px:
            lsb = px & (-px)
            u = lsb.bit_length() - 1
            px ^= lsb
            cnt = bin(P & adj_mask[u]).count('1')
            if cnt > best_count:
                best_count = cnt
                best_u = u
        # Branch on vertices in P not adjacent to pivot
        candidates = P & ~adj_mask[best_u]
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            bk(R | (1 << v), P & adj_mask[v], X & adj_mask[v])
            P ^= lsb
            X |= lsb

    full_mask = (1 << n) - 1
    bk(0, full_mask, 0)
    return cliques

def _maximal_cliques_size_ge2(cliques):
    """Filter cliques to those with size >= 2."""
    return [c for c in cliques if len(c) >= 2]

def _greedy_hitting_set(cliques, n):
    """Greedy: repeatedly pick vertex hitting most uncovered cliques."""
    # Map each vertex to list of clique indices it belongs to
    vertex_cliques = defaultdict(list)
    for idx, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].append(idx)
    
    uncovered = set(range(len(cliques)))
    selected = set()
    
    while uncovered and _remaining() > 5:
        best_v = None
        best_count = 0
        for v in range(n):
            if v in selected:
                continue
            count = sum(1 for idx in vertex_cliques[v] if idx in uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for idx in vertex_cliques[best_v]:
            uncovered.discard(idx)
    
    return selected

def _local_search_improve(hitting_set, cliques, n, max_iterations=500):
    """Try to improve by removing vertices or swapping."""
    hitting_set = set(hitting_set)
    
    def is_valid(hs):
        for c in cliques:
            if not c & hs:
                return False
        return True
    
    improved = True
    iters = 0
    while improved and iters < max_iterations and _remaining() > 10:
        improved = False
        iters += 1
        # Try removing each vertex
        for v in sorted(hitting_set):
            new_hs = hitting_set - {v}
            if is_valid(new_hs):
                hitting_set = new_hs
                improved = True
                break
        # Try swapping: remove v, add w
        if not improved:
            for v in sorted(hitting_set):
                for w in range(n):
                    if w in hitting_set:
                        continue
                    new_hs = (hitting_set - {v}) | {w}
                    if is_valid(new_hs):
                        hitting_set = new_hs
                        improved = True
                        break
                if improved:
                    break
    
    return hitting_set

def _solve_instance(n, p, seed):
    """Solve one instance: find minimal hitting set for all maximal cliques size >= 2."""
    edges, adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_maximal_cliques(adj, n)
    cliques = _maximal_cliques_size_ge2(cliques)
    
    if not cliques:
        return []
    
    # Greedy initial solution
    hs = _greedy_hitting_set(cliques, n)
    
    # Local search improvement
    hs = _local_search_improve(hs, cliques, n)
    
    return sorted(hs)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _remaining() < 10:
            # Not enough time; return empty lists as fallback (will be invalid but avoids timeout)
            results.append([])
            continue
        hs = _solve_instance(n, p, seed)
        results.append(hs)
    
    return results
