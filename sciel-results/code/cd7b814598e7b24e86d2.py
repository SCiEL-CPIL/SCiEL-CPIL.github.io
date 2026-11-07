"""Optimized branch-and-bound with local search to maximize distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _compute_degrees(vertices, adj, n):
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    degs = {}
    for v in vertices:
        degs[v] = _popcount(adj[v] & mask)
    return degs


def _distinct_degree_count(vertices, adj, n):
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    degs = set()
    for v in vertices:
        degs.add(_popcount(adj[v] & mask))
    return len(degs)


def _local_search(vertices, adj, n, deadline):
    """Hill-climbing: try adding/removing/swapping vertices to increase distinct degrees."""
    best = list(vertices)
    best_score = _distinct_degree_count(best, adj, n)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best)
        current_score = best_score
        
        # Try adding a vertex
        for v in range(n):
            if v not in current and time.time() > deadline:
                break
            trial = current + [v]
            s = _distinct_degree_count(trial, adj, n)
            if s > current_score:
                current_score = s
                current = trial
                improved = True
        
        # Try removing a vertex (if size > minv)
        minv = P['minimum_vertices']
        if len(current) > minv:
            for v in list(current):
                if time.time() > deadline:
                    break
                trial = [x for x in current if x != v]
                s = _distinct_degree_count(trial, adj, n)
                if s > current_score:
                    current_score = s
                    current = trial
                    improved = True
        
        # Try swapping
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for v in range(n):
                if v not in current:
                    trial = list(current)
                    trial[i] = v
                    s = _distinct_degree_count(trial, adj, n)
                    if s > current_score:
                        current_score = s
                        current = trial
                        improved = True
                        break
        
        if improved:
            best = current
            best_score = current_score
    
    return best, best_score


def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    vertices_all = list(range(n))
    
    best_count = -1
    best_vertices = list(range(minv))
    
    # Phase 1: Branch and bound search
    # We'll do a DFS with pruning, trying to build subsets of various sizes.
    # Key insight: for a subset of size k, max distinct degrees is k (all different) 
    # but degrees range from 0 to k-1, so max is min(k, k) = k but practically limited.
    
    # Strategy: enumerate subsets by size from large to small, with pruning.
    # Use the fact that if we have a partial assignment, we can bound the max possible.
    
    # Actually, let's use a smarter approach: 
    # 1. First do a quick scan of all size-8 subsets (C(17,8)=24310) - fast
    # 2. Then size-9 (C(17,9)=24310), etc.
    # But with pruning: if best_count >= size, skip that size.
    
    # Precompute: for each vertex, its degree in the full graph
    full_degs = [_popcount(adj[v]) for v in range(n)]
    
    # Phase 1a: Enumerate subsets of size 8 first (most likely to have many distinct degrees
    # relative to size, and C(17,8) is manageable)
    # Then go up in size.
    
    # Actually, let's think: a subset of size k can have at most k distinct degrees.
    # So a size-8 subset can have at most 8 distinct degrees.
    # A size-17 subset can have at most 17 distinct degrees.
    # We want to maximize the count, so larger subsets have higher potential.
    # But the degrees must actually be distinct.
    
    # Let's just enumerate all subset sizes from minv to n, with pruning.
    # For size k, if k <= best_count, we can stop (no larger subset will be checked after).
    
    # Use random restarts with local search for the harder parts.
    
    # Phase 1: Systematic enumeration with pruning
    for size in range(n, minv - 1, -1):
        if size <= best_count:
            break
        if time.time() > deadline:
            break
        
        # For this size, enumerate combinations with early termination
        count = 0
        for combo in itertools.combinations(vertices_all, size):
            count += 1
            if time.time() > deadline:
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            degs = set()
            for v in combo:
                degs.add(_popcount(adj[v] & mask))
            
            c = len(degs)
            if c > best_count:
                best_count = c
                best_vertices = list(combo)
        
        # If we found a subset with all distinct degrees (c == size), we can't do better for this size
        # But we continue to smaller sizes unless size <= best_count
    
    # Phase 2: Local search from multiple random starting points
    if time.time() < deadline:
        random.seed(42)
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random starting subset
            size = random.randint(minv, n)
            start = random.sample(vertices_all, size)
            
            # Local search
            improved_verts, improved_score = _local_search(start, adj, n, deadline)
            
            if improved_score > best_count:
                best_count = improved_score
                best_vertices = improved_verts
    
    # Phase 3: Also try local search from the best found so far
    if time.time() < deadline:
        improved_verts, improved_score = _local_search(best_vertices, adj, n, deadline)
        if improved_score > best_count:
            best_count = improved_score
            best_vertices = improved_verts
    
    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
