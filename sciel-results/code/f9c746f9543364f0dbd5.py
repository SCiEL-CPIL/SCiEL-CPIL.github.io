"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def _moore_n(k, g):
    """Moore bound: minimum n for k-regular girth g."""
    if g % 2 == 0:
        r = g // 2
        return 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        return k * sum((k-1)**i for i in range(r))

def _girth_check(n, adj, k, g):
    """Check girth >= g using non-backtracking BFS. adj is list of lists."""
    max_depth = (g - 1) // 2
    for start in range(n):
        # visited[node] = dict mapping prev -> dist
        visited = [None] * n
        visited[start] = {-1: 0}
        queue = deque([(start, -1, 0)])
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nbr in adj[node]:
                if nbr == prev:
                    continue
                v = visited[nbr]
                if v is not None:
                    if node in v:
                        cycle_len = dist + 1 + v[node]
                        if cycle_len < g:
                            return False
                else:
                    visited[nbr] = {node: dist + 1}
                    queue.append((nbr, node, dist + 1))
    return True

def _make_adj(n, edges):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _random_k_regular(n, k, rng):
    """Generate random k-regular simple graph."""
    if n * k < 2 or n * k % 2 != 0 or k >= n:
        return None
    
    adj = [set() for _ in range(n)]
    
    for attempt in range(200):
        adj = [set() for _ in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        success = True
        for v in order:
            needed = k - len(adj[v])
            if needed <= 0:
                continue
            candidates = [u for u in range(n) if u != v and u not in adj[v] and len(adj[u]) < k]
            if len(candidates) < needed:
                success = False
                break
            # Prefer lower-degree vertices
            candidates.sort(key=lambda u: len(adj[u]))
            if len(candidates) > needed:
                # Take needed from top candidates with randomness
                top = candidates[:needed*2]
                chosen = rng.sample(top, needed)
            else:
                chosen = candidates
            for u in chosen:
                adj[v].add(u)
                adj[u].add(v)
        if success:
            if all(len(adj[i]) == k for i in range(n)):
                edge_set = set()
                for v in range(n):
                    for u in adj[v]:
                        if v < u:
                            edge_set.add((v, u))
                return list(edge_set)
    return None

def construct(k, g):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    n_moore = _moore_n(k, g)
    heuristic_n = (k-1) ** (g // 2) if g % 2 == 0 else (k-1) ** ((g-1) // 2)
    heuristic_n = max(n_moore, int(heuristic_n * 1.5))
    
    best_n = None
    best_edges = None
    
    # Try n values
    n_values = []
    for n in range(max(n_moore, heuristic_n - 80), heuristic_n + 160, 2):
        if n * k % 2 == 0 and n >= k + 1:
            n_values.append(n)
    
    for n in n_values:
        if time.time() > deadline - 20:
            break
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            edges = _random_k_regular(n, k, rng)
            if edges is None:
                continue
            adj = _make_adj(n, edges)
            if _girth_check(n, adj, k, g):
                best_n = n
                best_edges = edges
                break
        if best_n is not None:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: larger n
    for n in range(heuristic_n + 160, heuristic_n + 400, 4):
        if time.time() > deadline - 5:
            break
        if n * k % 2 != 0:
            continue
        for attempt in range(15):
            if time.time() > deadline - 3:
                break
            edges = _random_k_regular(n, k, rng)
            if edges is None:
                continue
            adj = _make_adj(n, edges)
            if _girth_check(n, adj, k, g):
                return n, edges
    
    # Last resort
    n = max(n_moore, 2*k)
    edges = _random_k_regular(n, k, rng)
    if edges is not None:
        return n, edges
    return 0, []
# EVOLVE-BLOCK-END
