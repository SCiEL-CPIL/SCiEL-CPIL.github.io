import time
import random
import itertools
from collections import deque

def _has_c5(n, adj):
    """Check if graph has a C5 (5-cycle) as a subgraph. Returns True if C5 exists."""
    adj_masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_masks[u] = mask

    for subset in itertools.combinations(range(n), 5):
        a, b, c, d, e = subset
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> c) & 1 and (adj_masks[c] >> d) & 1 and (adj_masks[d] >> e) & 1 and (adj_masks[e] >> a) & 1:
            return True
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> c) & 1 and (adj_masks[c] >> e) & 1 and (adj_masks[e] >> d) & 1 and (adj_masks[d] >> a) & 1:
            return True
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> d) & 1 and (adj_masks[d] >> c) & 1 and (adj_masks[c] >> e) & 1 and (adj_masks[e] >> a) & 1:
            return True
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> d) & 1 and (adj_masks[d] >> e) & 1 and (adj_masks[e] >> c) & 1 and (adj_masks[c] >> a) & 1:
            return True
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> e) & 1 and (adj_masks[e] >> c) & 1 and (adj_masks[c] >> d) & 1 and (adj_masks[d] >> a) & 1:
            return True
        if (adj_masks[a] >> b) & 1 and (adj_masks[b] >> e) & 1 and (adj_masks[e] >> d) & 1 and (adj_masks[d] >> c) & 1 and (adj_masks[c] >> a) & 1:
            return True
        
        edge_count = 0
        verts = [a, b, c, d, e]
        for i in range(5):
            for j in range(i+1, 5):
                if (adj_masks[verts[i]] >> verts[j]) & 1:
                    edge_count += 1
        if edge_count < 5:
            continue
            
        for p1, p2, p3, p4 in itertools.permutations((b, c, d, e)):
            if ((adj_masks[a] >> p1) & 1 and 
                (adj_masks[p1] >> p2) & 1 and 
                (adj_masks[p2] >> p3) & 1 and 
                (adj_masks[p3] >> p4) & 1 and 
                (adj_masks[p4] >> a) & 1):
                return True

    return False

def _has_independent_set_9(n, adj):
    """Check if graph has an independent set of size 9. Returns True if it does."""
    adj_sets = [set(adj[i]) for i in range(n)]
    found = [False]
    
    def _search(chosen, candidates):
        if found[0]:
            return
        if len(chosen) >= 9:
            found[0] = True
            return
        if len(chosen) + len(candidates) < 9:
            return
        if not candidates:
            return
        
        min_deg = float('inf')
        min_v = None
        for v in candidates:
            deg = len(adj_sets[v] & candidates)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        v = min_v
        
        new_candidates = candidates - {v} - adj_sets[v]
        _search(chosen + [v], new_candidates)
        if found[0]:
            return
        
        _search(chosen, candidates - {v})
        
    _search([], set(range(n)))
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    
    adj_base = [set() for _ in range(n_base)]
    for u, v in edges_base:
        adj_base[u].add(v)
        adj_base[v].add(u)
        
    if not _has_c5(n_base, adj_base) and not _has_independent_set_9(n_base, adj_base):
        best_n = n_base
        best_edges = edges_base
    else:
        best_n = 0
        best_edges = []

    return {'n': best_n, 'edges': best_edges}
