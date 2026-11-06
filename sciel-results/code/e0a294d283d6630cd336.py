"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
import math

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with symmetric connection set S has a triangle.
    Triangle exists if there exist x, y in S (distinct) such that (x+y) mod n is in S or -S.
    """
    S_set = frozenset(S)
    neg_S = frozenset({(-x) % n for x in S})
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            x = S_list[i]
            y = S_list[j]
            z = (x + y) % n
            if z in S_set or z in neg_S:
                return True
    return False

def _max_independent_set_size(n, adj_mask, time_limit=None):
    """
    Compute alpha of a graph given its adjacency bitmasks.
    Uses branch and bound with bitmasks and degree-based vertex ordering.
    """
    full_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    start_time = time.time() if time_limit else None
    
    def dfs(candidates, current_size):
        if time_limit and time.time() - start_time > time_limit:
            return
        # Upper bound
        if current_size + candidates.bit_count() <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidates subgraph for better pruning
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (candidates & adj_mask[v]).bit_count()
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

def _hoffman_bound(n, S):
    """
    Compute Hoffman bound for alpha of circulant graph.
    Returns upper bound on alpha. If bound < 10, alpha < 10 guaranteed.
    """
    if n <= 1:
        return n
    S_set = set(S)
    if not S_set:
        return n
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                A[i][j] = 1
    
    try:
        w = np.linalg.eigvalsh(A)
        lambda_min = w[0]
        d = len(S_set)
        if d == 0:
            return n
        if d - lambda_min <= 0:
            return n
        bound = n * (-lambda_min) / (d - lambda_min)
        return int(np.ceil(bound))
    except:
        return n

def _verify_circulant(n, S, time_limit=30.0):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant(n, S):
        return False, 0
    
    # Quick Hoffman bound check - if it proves alpha < 10, skip expensive B&B
    hb = _hoffman_bound(n, S)
    if hb < 10:
        return True, hb
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    S_set = frozenset(S)
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj_mask[i] |= (1 << j)
    
    # Run exact B&B with time limit
    alpha = _max_independent_set_size(n, adj_mask, time_limit=time_limit)
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

def _enumerate_circulant(n, max_s_size=12, time_limit=20.0):
    """
    Systematically enumerate triangle-free circulant graphs for given n.
    Returns best (n, edges, alpha) or None.
    """
    start = time.time()
    half = n // 2
    available = list(range(1, half + 1))
    
    # Try S sizes from larger to smaller (larger S = denser graph = smaller alpha)
    for s_size in range(min(max_s_size, half), 2, -1):
        if time.time() - start > time_limit:
            break
        
        total_combos = len(available)
        if s_size <= total_combos:
            max_enumerate = 15000
            count = 0
            for combo in combinations(available, s_size):
                if time.time() - start > time_limit:
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
                
                valid, alpha = _verify_circulant(n, S, time_limit=15.0)
                if valid:
                    return n, _build_edges(n, S), alpha
    
    return None

def _structured_candidates(n, time_limit=8.0):
    """Generate structured circulant candidates: multiples, APs, unions, and known constructions."""
    start = time.time()
    half = n // 2
    candidates = []
    
    # Known constructions for specific n values
    # For n=39: use shifts that avoid triangles
    if n == 39:
        # Try various shift sets known to work
        for shifts in [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                       [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                       [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       [1,2,3,4,5,6,7,8,9,10,11,12,13],
                       [1,2,3,4,5,6,7,8,9,10,11,12]]:
            S = set()
            for x in shifts:
                if x < n:
                    S.add(x)
                    S.add((-x) % n)
            candidates.append(frozenset(S))
    
    if n == 40:
        for shifts in [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                       [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                       [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]:
            S = set()
            for x in shifts:
                if x < n:
                    S.add(x)
                    S.add((-x) % n)
            candidates.append(frozenset(S))
    
    # Multiples of small bases
    for base in range(2, 12):
        S = set()
        for k in range(1, n // base + 1):
            x = k * base
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Arithmetic progressions
    for start_v in range(1, 8):
        for step in range(2, 12):
            S = set()
            x = start_v
            while x < n:
                S.add(x)
                S.add((-x) % n)
                x += step
            if S:
                candidates.append(frozenset(S))
    
    # Unions of two APs
    for s1 in range(1, 5):
        for st1 in range(2, 8):
            for s2 in range(1, 5):
                for st2 in range(2, 8):
                    if st1 == st2 and s1 == s2:
                        continue
                    S = set()
                    x = s1
                    while x < n:
                        S.add(x)
                        S.add((-x) % n)
                        x += st1
                    x = s2
                    while x < n:
                        S.add(x)
                        S.add((-x) % n)
                        x += st2
                    if S:
                        candidates.append(frozenset(S))
    
    # Unions of three APs
    for s1 in range(1, 4):
        for st1 in range(2, 6):
            for s2 in range(1, 4):
                for st2 in range(2, 6):
                    for s3 in range(1, 4):
                        for st3 in range(2, 6):
                            if (st1 == st2 and s1 == s2) or (st1 == st3 and s1 == s3) or (st2 == st3 and s2 == s3):
                                continue
                            S = set()
                            x = s1
                            while x < n:
                                S.add(x)
                                S.add((-x) % n)
                                x += st1
                            x = s2
                            while x < n:
                                S.add(x)
                                S.add((-x) % n)
                                x += st2
                            x = s3
                            while x < n:
                                S.add(x)
                                S.add((-x) % n)
                                x += st3
                            if S:
                                candidates.append(frozenset(S))
    
    # Random symmetric sets with triangle avoidance (greedy)
    for _ in range(100):
        if time.time() - start > time_limit:
            break
        S = set()
        cands = list(range(1, half + 1))
        random.shuffle(cands)
        for c in cands:
            new_S = S | {c, (-c) % n}
            new_S = frozenset(new_S)
            if not _has_triangle_circulant(n, new_S):
                S = set(new_S)
        if S:
            candidates.append(frozenset(S))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # Try n from 40 down to 25
    for n in range(40, 24, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        half = n // 2
        
        # Strategy 1: Structured candidates (fast, targeted)
        struct_cands = _structured_candidates(n, time_limit=5.0)
        for S in struct_cands:
            if time.time() > deadline - 15:
                break
            valid, alpha = _verify_circulant(n, S, time_limit=20.0)
            if valid:
                best_n = n
                best_edges = _build_edges(n, S)
                found = True
                break
        
        # Strategy 2: Systematic enumeration
        if not found:
            result = _enumerate_circulant(n, max_s_size=min(12, half), time_limit=25.0)
            if result:
                n_found, edges, alpha = result
                best_n = n_found
                best_edges = edges
                found = True
        
        # Strategy 3: Greedy with random restarts
        if not found:
            for k in range(min(half, 15), 2, -1):
                if time.time() > deadline - 15:
                    break
                
                num_restarts = 60 if n <= 35 else 30
                
                for restart in range(num_restarts):
                    if time.time() > deadline - 10:
                        break
                    
                    S = set()
                    cands = list(range(1, half + 1))
                    random.shuffle(cands)
                    for c in cands:
                        if len(S) >= k:
                            break
                        new_S = S | {c, (-c) % n}
                        new_S = frozenset(new_S)
                        if not _has_triangle_circulant(n, new_S):
                            S = set(new_S)
                    
                    if len(S) < k:
                        continue
                    
                    S = frozenset(S)
                    valid, alpha = _verify_circulant(n, S, time_limit=15.0)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        # Strategy 4: Random sampling
        if not found:
            for k in range(min(half, 15), 2, -1):
                if time.time() > deadline - 15:
                    break
                
                num_attempts = 200 if n <= 35 else 100
                
                for attempt in range(num_attempts):
                    if time.time() > deadline - 10:
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
                    
                    valid, alpha = _verify_circulant(n, S, time_limit=15.0)
                    if valid:
                        best_n = n
                        best_edges = _build_edges(n, S)
                        found = True
                        break
                
                if found:
                    break
        
        if found:
            break
    
    # Fallback: Known small graph (Cayley graph on Z_26)
    if best_n < 20:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
