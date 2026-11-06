import time
import numpy as np
from itertools import combinations

def _has_c5(adj: list[set]) -> bool:
    """Exact check for a C5 subgraph."""
    n = len(adj)
    # Enumerate paths of length 4: v0-v1-v2-v3-v4
    # Check if v4 is adjacent to v0.
    # To avoid duplicates and ensure distinct vertices, we enforce v0 < v1 < v2 < v3 < v4 is NOT required for the path,
    # but we can just check all 5-tuples of distinct vertices.
    # A more efficient way:
    # For each edge (v0, v4), check if there is a path of length 3 from v0 to v4.
    # Path v0-v1-v2-v3-v4.
    # v1 in N(v0), v3 in N(v4), v2 in N(v1) \cap N(v3).
    # And v0, v1, v2, v3, v4 must be distinct.
    
    for v0 in range(n):
        for v4 in adj[v0]:
            if v4 <= v0:
                continue
            # We need v1, v2, v3 such that v0-v1-v2-v3-v4 is a path.
            # v1 is a neighbor of v0, v1 != v4
            # v3 is a neighbor of v4, v3 != v0
            # v2 is a common neighbor of v1 and v3
            # v2 != v0, v1, v3, v4
            n_v0 = adj[v0] - {v4}
            n_v4 = adj[v4] - {v0}
            for v1 in n_v0:
                for v3 in n_v4:
                    if v1 == v3:
                        continue
                    # Check if there is a v2 in N(v1) \cap N(v3)
                    # such that v2 is not in {v0, v1, v3, v4}
                    common = adj[v1] & adj[v3]
                    # Remove forbidden vertices
                    common -= {v0, v1, v3, v4}
                    if common:
                        # Found a C5
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
        # Heuristic: pick vertex with max degree in candidates
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        # Include v
        nc = candidates & comp[v]
        search(clique_sz + 1, nc)
        if found[0]:
            return
        # Exclude v
        c2 = candidates - {v}
        search(clique_sz, c2)
    
    search(0, frozenset(range(n)), )
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Base: 8 disjoint K4s
    n_base = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Add 3 more vertices: 32, 33, 34
    # Connect 32 to K4 block 0 (0,1,2,3)
    # Connect 33 to K4 block 1 (4,5,6,7)
    # Connect 34 to K4 block 2 (8,9,10,11)
    
    for v in [0, 1, 2, 3]:
        edges.append([32, v])
    for v in [4, 5, 6, 7]:
        edges.append([33, v])
    for v in [8, 9, 10, 11]:
        edges.append([34, v])
        
    n = 35
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify
    if not _has_c5(adj) and not _independent_set_size(adj, 9):
        return {'n': n, 'edges': edges}
    else:
        # Fallback to n=32 if logic fails (should not happen)
        n = 32
        edges_fallback = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges_fallback.append([base + i, base + j])
        return {'n': n, 'edges': edges_fallback}
