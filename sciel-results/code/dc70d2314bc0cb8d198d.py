"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 110  # 10s safety margin before 120s kill

def _popcount_fast(x):
    """Count set bits in integer using Python's built-in."""
    return bin(x).count('1')

def _find_bad_6_subset(G, n):
    """
    Find a 6-vertex subset inducing a regular subgraph.
    Returns the subset as a tuple, or None if none found.
    Uses bit-parallelism: precompute for each 6-subset the degree vector.
    """
    # For n up to 22, C(22,6) = 74613 subsets - manageable
    # For each subset, compute degrees of all 6 vertices
    verts = list(range(n))
    
    # Precompute adjacency masks
    adj = G  # G[i] has bit j set if edge (i,j)
    
    for combo in combinations(verts, 6):
        a, b, c, d, e, f = combo
        # Build subset mask
        sm = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e) | (1 << f)
        
        # Compute degree of each vertex within subset
        da = _popcount_fast(adj[a] & (sm & ~(1 << a)))
        db = _popcount_fast(adj[b] & (sm & ~(1 << b)))
        dc = _popcount_fast(adj[c] & (sm & ~(1 << c)))
        dd = _popcount_fast(adj[d] & (sm & ~(1 << d)))
        de = _popcount_fast(adj[e] & (sm & ~(1 << e)))
        df = _popcount_fast(adj[f] & (sm & ~(1 << f)))
        
        if da == db == dc == dd == de == df:
            return combo
    
    return None

def _find_bad_subset_any_size(G, n):
    """
    Find any subset of size >= 6 inducing a regular subgraph.
    First check size 6 (most common), then larger.
    Returns (k, subset_tuple) or (0, None).
    """
    # Check size 6 first
    subset = _find_bad_6_subset(G, n)
    if subset is not None:
        return 6, subset
    
    # Check sizes 7 through n
    verts = list(range(n))
    for k in range(7, n + 1):
        for combo in combinations(verts, k):
            sm = 0
            for v in combo:
                sm |= (1 << v)
            
            degs = []
            for v in combo:
                d = _popcount_fast(G[v] & (sm & ~(1 << v)))
                degs.append(d)
            
            if len(set(degs)) == 1:
                return k, combo
    
    return 0, None

def _is_valid(G, n):
    """Check if G has no regular induced subgraph on >= 6 vertices."""
    k, _ = _find_bad_subset_any_size(G, n)
    return k == 0

def _count_bad_6(G, n):
    """Count number of 6-vertex subsets inducing regular subgraphs."""
    count = 0
    verts = list(range(n))
    for combo in combinations(verts, 6):
        a, b, c, d, e, f = combo
        sm = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e) | (1 << f)
        da = _popcount_fast(G[a] & (sm & ~(1 << a)))
        db = _popcount_fast(G[b] & (sm & ~(1 << b)))
        dc = _popcount_fast(G[c] & (sm & ~(1 << c)))
        dd = _popcount_fast(G[d] & (sm & ~(1 << d)))
        de = _popcount_fast(G[e] & (sm & ~(1 << e)))
        df = _popcount_fast(G[f] & (sm & ~(1 << f)))
        if da == db == dc == dd == de == df:
            count += 1
    return count

def _try_local_search(n, seed, time_limit):
    """
    Local search: start with random graph, iteratively fix bad subsets.
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    
    # Start with random graph
    p = 0.3
    G = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i] |= (1 << j)
                G[j] |= (1 << i)
    
    # Check initial
    if _is_valid(G, n):
        return G
    
    # Local search iterations
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Find a bad subset
        k, subset = _find_bad_subset_any_size(G, n)
        if k == 0:
            return G
        
        # Convert to list for easy manipulation
        subset_list = list(subset)
        
        # Compute degrees within subset
        sm = 0
        for v in subset_list:
            sm |= (1 << v)
        
        degs = {}
        for v in subset_list:
            degs[v] = _popcount_fast(G[v] & (sm & ~(1 << v)))
        
        common_deg = degs[subset_list[0]]
        
        # Try flipping edges to break regularity
        pairs = list(combinations(subset_list, 2))
        random.shuffle(pairs)
        
        flipped = False
        for u, v in pairs[:10]:
            if (G[u] >> v) & 1:
                new_degu = degs[u] - 1
                new_degv = degs[v] - 1
            else:
                new_degu = degs[u] + 1
                new_degv = degs[v] + 1
            
            if new_degu != common_deg or new_degv != common_deg:
                if (G[u] >> v) & 1:
                    G[u] &= ~(1 << v)
                    G[v] &= ~(1 << u)
                else:
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
                flipped = True
                break
        
        if not flipped:
            # Try a random flip in the subset
            u, v = random.choice(pairs)
            if (G[u] >> v) & 1:
                G[u] &= ~(1 << v)
                G[v] &= ~(1 << u)
            else:
                G[u] |= (1 << v)
                G[v] |= (1 << u)
    
    # Final check
    if _is_valid(G, n):
        return G
    return None

def _to_edges(G, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (G[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    best_n = 5
    best_G = [0] * 5  # empty graph on 5 vertices
    
    # Try to find larger graphs, starting from 20 and going down
    for n in range(20, 4, -1):
        if time.time() - start > 105:
            break
        
        # Try multiple seeds with different densities
        found = False
        for seed in range(50):
            if time.time() - start > 108:
                break
            
            # Vary the initial density
            if seed % 3 == 0:
                p = 0.25
            elif seed % 3 == 1:
                p = 0.35
            else:
                p = 0.45
            
            G = _try_local_search(n, seed, 2.0)
            if G is not None:
                best_n = n
                best_G = G
                found = True
                break
        
        if found:
            break
    
    return {'n': best_n, 'edges': _to_edges(best_G, best_n)}
# EVOLVE-BLOCK-END
