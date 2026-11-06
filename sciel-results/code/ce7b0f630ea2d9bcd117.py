import time
import numpy as np
from itertools import combinations

def _has_c5(adj: list[set]) -> bool:
    """Exact check for a C5 subgraph."""
    n = len(adj)
    for v0 in range(n):
        for v4 in adj[v0]:
            if v4 <= v0:
                continue
            n_v0 = adj[v0] - {v4}
            n_v4 = adj[v4] - {v0}
            for v1 in n_v0:
                for v3 in n_v4:
                    if v1 == v3:
                        continue
                    common = adj[v1] & adj[v3]
                    common -= {v0, v1, v3, v4}
                    if common:
                        return True
    return False

def _independent_set_size(adj: list[set], target: int) -> bool:
    """Check if there exists an independent set of size >= target."""
    n = len(adj)
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    
    found = [False]
    
    def search(clique_sz, candidates):
        if found[0]:
            return
        if clique_sz >= target:
            found[0] = True
            return
        if not candidates:
            return
        if clique_sz + len(candidates) < target:
            return
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        nc = candidates & comp[v]
        search(clique_sz + 1, nc)
        if found[0]:
            return
        c2 = candidates - {v}
        search(clique_sz, c2)
    
    search(0, frozenset(range(n)), )
    return found[0]

def construct():
    # Construct 8 disjoint K4s (32 vertices)
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Verify (optional but good for safety)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    if not _has_c5(adj) and not _independent_set_size(adj, 9):
        return {'n': n, 'edges': edges}
    else:
        # This should never happen for 8 K4s
        return {'n': 0, 'edges': []}
