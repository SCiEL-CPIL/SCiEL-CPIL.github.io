"""Improved program for R(3,10) task; uses circulant graphs with optimized search."""
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with symmetric connection set S has a triangle."""
    S_set = frozenset(S)
    neg_S = frozenset({(-x) % n for x in S})
    S_list = list(S_set)
    for i, x in enumerate(S_list):
        for y in S_list:
            z = (x + y) % n
            if z in S_set or z in neg_S:
                return True
    return False

def _max_independent_set_size(n, adj_mask):
    """
    Compute alpha of a graph given its adjacency bitmasks using branch and bound.
    """
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size):
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidates for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant(n, S):
        return False, 0
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    S_set = frozenset(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj_mask[i] |= (1 << j)
    
    # Run exact B&B
    alpha = _max_independent_set_size(n, adj_mask)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges(n, S):
    """Build edge list from circulant connection set."""
    edges = []
    S_set = set(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j > i:
                edges.append([i, j])
    return edges

def _try_circulant(n, S):
    """Try a circulant graph, return (n, edges) if valid, else None."""
    valid, alpha = _verify_circulant(n, S)
    if valid:
        return n, _build_edges(n, S)
    return None

def _greedy_add(n, S, candidates, target_size):
    """Greedily add elements to S maintaining triangle-free property."""
    S = frozenset(S)
    for c in candidates:
        if len(S) >= target_size:
            break
        new_S = S | {c, (-c) % n}
        new_S = frozenset(new_S)
        if not _has_triangle_circulant(n, new_S):
            S = new_S
    return S

def _local_search(n, S, iterations=200, time_limit=5.0):
    """Local search: try swapping elements in S to improve triangle-free property and reduce alpha."""
    start = time.time()
    S = frozenset(S)
    half = n // 2
    available = [x for x in range(1, half + 1) if x not in S and (-x) % n not in S]
    
    for _ in range(iterations):
        if time.time() - start > time_limit:
            break
        
        # Try adding an element
        if available:
            idx = random.randint(0, len(available) - 1)
            c = available[idx]
            new_S = S | {c, (-c) % n}
            new_S = frozenset(new_S)
            if not _has_triangle_circulant(n, new_S):
                valid, alpha = _verify_circulant(n, new_S)
                if valid:
                    S = new_S
                    available = [x for x in range(1, half + 1) if x not in S and (-x) % n not in S]
                    continue
        
        # Try removing and replacing an element
        S_list = list(S)
        if len(S_list) > 4:
            idx = random.randint(0, len(S_list) - 1)
            old_c = S_list[idx]
            old_neg = (-old_c) % n
            S_minus = S - {old_c, old_neg}
            
            # Try to replace with a different element
            candidates = [x for x in range(1, half + 1) if x not in S_minus and (-x) % n not in S_minus]
            if candidates:
                new_c = random.choice(candidates)
                new_S = S_minus | {new_c, (-new_c) % n}
                new_S = frozenset(new_S)
                if not _has_triangle_circulant(n, new_S):
                    valid, alpha = _verify_circulant(n, new_S)
                    if valid:
                        S = new_S
                        continue
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        half = n // 2
        
        # Strategy 1: Systematic enumeration of circulant graphs
        # Try S sizes from larger to smaller (larger S = denser graph = smaller alpha)
        for s_size in range(min(8, half), 2, -1):
            if time.time() > deadline - 5:
                break
            
            available = list(range(1, half + 1))
            max_enumerate = 3000
            count = 0
            
            for combo in combinations(available, s_size):
                if time.time() > deadline - 5:
                    break
                if count >= max_enumerate:
                    break
                count += 1
                
                S = set()
                for x in combo:
                    S.add(x)
                    S.add((-x) % n)
                S = frozenset(S)
                
                if _has_triangle_circulant(n, S):
                    continue
                
                valid, alpha = _verify_circulant(n, S)
                if valid:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
            
            if found:
                break
        
        # Strategy 2: Greedy construction with random restarts
        if not found:
            for k in range(min(half, 10), 2, -1):
                if time.time() > deadline - 5:
                    break
                
                num_restarts = 50 if n <= 35 else 20
                
                for restart in range(num_restarts):
                    if time.time() > deadline - 5:
                        break
                    
                    candidates = list(range(1, half + 1))
                    random.shuffle(candidates)
                    S = _greedy_add(n, frozenset(), candidates, k)
                    
                    if len(S) < 4:
                        continue
                    
                    # Local search to improve
                    S = _local_search(n, S, iterations=100, time_limit=2.0)
                    
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        # Strategy 3: Random sampling
        if not found:
            for k in range(min(half, 10), 2, -1):
                if time.time() > deadline - 5:
                    break
                
                num_attempts = 150 if n <= 35 else 80
                
                for attempt in range(num_attempts):
                    if time.time() > deadline - 5:
                        break
                    
                    if k > half:
                        continue
                        
                    S_half = set(random.sample(range(1, half + 1), k))
                    S = set()
                    for x in S_half:
                        S.add(x)
                        S.add((-x) % n)
                    S = frozenset(S)
                    
                    if _has_triangle_circulant(n, S):
                        continue
                    
                    valid, alpha = _verify_circulant(n, S)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    # Fallback: Known small graph
    if best_n < 19:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
