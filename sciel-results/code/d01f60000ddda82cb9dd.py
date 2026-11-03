"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    def elapsed():
        return time.time() - start_time

    # 1. Generate Candidate Pool
    # We use shells 1, 2, 3, 4.
    # Shell 1: 11 vectors
    # Shell 2: 220 vectors
    # Shell 3: 1716 vectors
    # Shell 4: 11880 vectors
    # Total ~ 13827
    
    vecs = []
    
    # Shell 1
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    candidates = np.array(vecs, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1).astype(np.int64)

    # 2. Construct Conflict Graph
    # Conflict if ||u - v||^2 < max(||u||^2, ||v||^2)
    # This is equivalent to:
    # ||u||^2 + ||v||^2 - 2<u,v> < max(||u||^2, ||v||^2)
    
    # To save memory and time, we compute the Gram matrix G = C @ C.T
    # G is (N, N). N ~ 13800.
    # 13800^2 * 4 bytes (float32) ~ 760 MB.
    # This is a significant memory hit, but NumPy handles it.
    # Let's try to be smarter if possible, but for now, let's assume it fits.
    # If it fails, we can fall back to a smaller subset or greedy without full matrix (slower).
    
    try:
        # Compute Gram matrix
        # Using float32 to save memory
        C_float = candidates.astype(np.float32)
        G = C_float @ C_float.T
        
        # Compute conflict matrix
        # dist_sq = ns_i + ns_j - 2*G
        # max_ns = max(ns_i, ns_j)
        # conflict = dist_sq < max_ns
        
        ns = norms_sq.astype(np.float32)
        ns_col = ns[:, np.newaxis]
        ns_row = ns[np.newaxis, :]
        
        dist_sq = ns_col + ns_row - 2 * G
        max_ns = np.maximum(ns_col, ns_row)
        
        conflict = dist_sq < max_ns
        np.fill_diagonal(conflict, False)
        
        # 3. Solve Maximum Independent Set (MIS)
        # We use a greedy heuristic with multiple restarts and different orderings.
        # Orderings:
        # 1. Ascending Degree in Conflict Graph
        # 2. Random
        # 3. Lexsort by (Norm, Degree)
        
        degrees = np.sum(conflict, axis=1)
        
        def greedy_is(order):
            available = np.ones(n_cand, dtype=bool)
            selected = []
            
            for v in order:
                if not available[v]:
                    continue
                    
                selected.append(v)
                
                # Block v and neighbors
                available[v] = False
                available[conflict[v, :]] = False
                
            return selected
        
        best_sel = []
        best_size = 0
        
        # Heuristic 1: Ascending Degree
        order1 = np.argsort(degrees)
        sel1 = greedy_is(order1)
        if len(sel1) > best_size:
            best_size = len(sel1)
            best_sel = sel1
            
        if elapsed() > deadline - 20:
            return np.array(candidates[best_sel], dtype=np.int64)
            
        # Heuristic 2: Random Restarts
        rng = np.random.default_rng(42)
        n_restarts = 50 # Tune based on time
        
        for i in range(n_restarts):
            if elapsed() > deadline - 10:
                break
                
            rand_order = rng.permutation(n_cand)
            sel_r = greedy_is(rand_order)
            if len(sel_r) > best_size:
                best_size = len(sel_r)
                best_sel = sel_r
                
        # Heuristic 3: Lexsort by Norm, then Degree
        order3 = np.lexsort((degrees, norms_sq))
        sel3 = greedy_is(order3)
        if len(sel3) > best_size:
            best_size = len(sel3)
            best_sel = sel3
            
        # Heuristic 4: Local Search / Swap?
        # Given time constraints, greedy is likely sufficient for a significant jump.
        # A more advanced solver (like Bron-Kerbosch on complement) might be too slow for N=13800.
        
    except Exception as e:
        # Fallback: Simple greedy without full conflict matrix?
        # This would be O(N^2) per step, too slow.
        # Just return a small valid set.
        # Shell 1 vectors are mutually compatible?
        # ||e_i - e_j||^2 = 2. max(norm^2) = 1. 2 >= 1. Yes.
        # So we can at least return Shell 1.
        return candidates[:11]

    return np.array(candidates[best_sel], dtype=np.int64)

# EVOLVE-BLOCK-END
