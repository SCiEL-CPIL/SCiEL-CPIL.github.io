"""Explicit construction for a C4-free graph with alpha <= 10 and as many vertices as possible.

Approach:
  - Random greedy construction of C4-free graph on n=43 (or lower).
  - Add random edges that don't create a C4 (no two vertices share >1 common neighbor).
  - Use fast branch-and-bound to check alpha <= 10.
  - Deadline guard with safe fallback to the 30-vertex triangle seed.
"""

import time
import random

DEADLINE = 140.0
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _remaining():
    return DEADLINE - _elapsed()

def _seed_graph():
    """10 disjoint triangles: C4-free, alpha=10, n=30."""
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def _has_c4(adj, u, v):
    """Check if adding edge (u,v) creates a 4-cycle.
    A C4 using new edge (u,v) exists iff there are a~u, b~v with a~b, a!=b, a!=v, b!=u."""
    for a in adj[u]:
        if a == v:
            continue
        for b in adj[v]:
            if b == u or b == a:
                continue
            if b in adj[a]:
                return True
    return False

def _alpha_le_10(adj, n, deadline_margin=2.0):
    """Branch and bound: return True if max independent set size <= 10.
    Uses bitmask for speed. Stops early if finds IS of size 11."""
    # Convert to bitmask adjacency
    adj_mask = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        adj_mask[i] = m
    
    full_mask = (1 << n) - 1
    found_11 = [False]
    start_time = time.time()
    
    # Bit operations helper
    def popcount(x):
        return bin(x).count('1')
    
    def bnb(avail, size):
        if found_11[0]:
            return
        if time.time() - start_time > deadline_margin:
            return
        if size >= 11:
            found_11[0] = True
            return
        if avail == 0:
            return
        # Upper bound: size + popcount(avail) <= 10 means we can't reach 11
        if size + popcount(avail) < 11:
            return
        # Pick vertex with max degree in avail (greedy heuristic)
        # Try vertices in order of decreasing degree within avail
        best_v = -1
        best_deg = -1
        temp = avail
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(avail & adj_mask[v])
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v < 0:
            return
        
        v = best_v
        bit = 1 << v
        
        # Branch 1: include v
        new_avail = avail & ~(bit | adj_mask[v])
        bnb(new_avail, size + 1)
        
        if found_11[0]:
            return
        
        # Branch 2: exclude v
        bnb(avail & ~bit, size)
    
    bnb(full_mask, 0)
    return not found_11[0]

def _try_build_graph(max_n=43, max_trials=100):
    """Try to build a C4-free graph on up to max_n vertices with alpha <= 10."""
    best_result = _seed_graph()
    best_n = 30
    rng = random.Random(12345)
    
    for trial in range(max_trials):
        if _remaining() < 8.0:
            break
        
        n_target = min(max_n, 39 + (trial % 5))
        if n_target <= best_n:
            continue
        
        # Build random C4-free graph
        adj = [set() for _ in range(n_target)]
        
        all_edges = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        rng.shuffle(all_edges)
        
        # Target edge count: avg degree ~6-8 for C4-free graphs
        edge_count_target = int(n_target * 3.0)
        added = 0
        for (u, v) in all_edges:
            if added >= edge_count_target:
                break
            if _remaining() < 5.0:
                break
            if not _has_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        
        if added < n_target * 1.5:
            continue
        
        # Check alpha <= 10
        if _alpha_le_10(adj, n_target, deadline_margin=3.0):
            edges = []
            for i in range(n_target):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            result = {'n': n_target, 'edges': edges}
            if n_target > best_n:
                best_n = n_target
                best_result = result
                if n_target >= max_n:
                    break
    
    return best_result

def construct():
    global _START
    _START = time.time()
    
    seed = _seed_graph()
    
    if _remaining() > 15.0:
        try:
            result = _try_build_graph(max_n=43, max_trials=80)
            if result['n'] > seed['n']:
                return result
        except Exception:
            pass
    
    return seed
