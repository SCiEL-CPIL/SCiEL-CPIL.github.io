import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 570
    
    # Generate all vectors with norm squared 3
    # Form: +/- e_i +/- e_j +/- e_k with i < j < k
    three_positions = list(combinations(range(d), 3))
    candidates = []
    cand_support = []
    
    for pos in three_positions:
        for signs in product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
            
    n_cand = len(candidates) # 1320
    
    # Precompute conflicts
    # Conflict if dot product >= 2
    # Which happens if intersection size 2 and signs match on those 2.
    # Or intersection size 3 (same support) and signs match on all 3? 
    # If same support, u=v is not a pair. u!=v means signs differ in at least 1 pos.
    # If differ in 1 pos, dot = 1.
    # If differ in 2 pos, dot = -1.
    # If differ in 3 pos, dot = -3.
    # So no conflicts within same support.
    # Conflicts only between different supports with intersection 2.
    
    # Map support to list of candidate indices
    support_to_cands = {}
    for i, s in enumerate(cand_support):
        if s not in support_to_cands:
            support_to_cands[s] = []
        support_to_cands[s].append(i)
        
    # Build adjacency list
    # For each pair of supports with intersection 2, check conflicts
    # This is efficient enough for 1320 nodes.
    # Or just brute force O(N^2) dot products? 1320^2 = 1.7M. Very fast.
    adj = [set() for _ in range(n_cand)]
    
    # Brute force conflict detection
    # u . v >= 2
    for i in range(n_cand):
        vi = candidates[i]
        for j in range(i+1, n_cand):
            vj = candidates[j]
            dot = np.dot(vi, vj)
            if dot >= 2:
                adj[i].add(j)
                adj[j].add(i)
                
    # Greedy Independent Set with random restarts
    best_size = 0
    best_indices = []
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10:
        order = rng.permutation(n_cand)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            conflict = False
            for sel in selected:
                if sel in adj[idx]:
                    conflict = True
                    break
            if not conflict:
                selected.append(idx)
                selected_set.add(idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected[:]
            
        # Optional: Local search improvement
        # Try to improve current best by swapping
        # Given time constraints, simple greedy might be sufficient to beat 220.
        # 1320 nodes, degree ~48.
        # Random greedy usually yields N / (2 * log N) ? No, N / (d+1) is lower bound.
        # Let's hope for > 300.
        
    if len(best_indices) < 2:
        # Fallback to known valid set of size 220
        # Norm 2 vectors
        pts = []
        for i in range(d):
            for j in range(i+1, d):
                v1 = np.zeros(d, dtype=np.int64)
                v1[i] = 1; v1[j] = 1
                pts.append(v1)
                v2 = -v1
                pts.append(v2)
                v3 = np.zeros(d, dtype=np.int64)
                v3[i] = 1; v3[j] = -1
                pts.append(v3)
                v4 = -v3
                pts.append(v4)
        points = pts
    else:
        points = [candidates[i] for i in best_indices]
        
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
