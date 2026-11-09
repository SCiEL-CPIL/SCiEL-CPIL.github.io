"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start with a good seed: all ±1 vectors (2^11 = 2048 candidates, but we need to select)
    # Actually, let's start with a structured set and then optimize
    
    # Generate candidate pool: sparse vectors with small integer entries
    candidates = []
    
    # Add all weight-1 vectors (2*11 = 22)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Add all weight-2 vectors (2 * C(11,2) * 4 = 880)
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Add some weight-3 vectors
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    norms = np.sqrt(norms_sq)
    
    # Greedy construction: add vectors that maximize minimum distance
    selected = []
    min_dist_sq = np.inf
    
    for idx in range(n_cand):
        if time.time() > deadline - 5.0:
            break
        if len(selected) == 0:
            selected.append(idx)
            min_dist_sq = np.inf
            continue
        
        # Compute distances to all selected
        diff = candidates[idx] - candidates[selected]
        dist_sq = np.sum(diff * diff, axis=1)
        min_d = np.min(dist_sq)
        
        if min_d >= min_dist_sq:
            selected.append(idx)
            min_dist_sq = min(min_dist_sq, min_d)
    
    # Now try to improve by swapping
    current_min_dist_sq = min_dist_sq
    
    # Local search: try replacing each selected vector with a better candidate
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for pos in range(len(selected)):
            if time.time() > deadline - 2.0:
                break
            best_alt = None
            best_min_d = current_min_dist_sq - 1
            
            for alt_idx in range(n_cand):
                if alt_idx in selected:
                    continue
                
                # Try replacing selected[pos] with alt_idx
                new_selected = selected[:pos] + [alt_idx] + selected[pos+1:]
                
                # Compute min distance for new set
                diff = candidates[alt_idx] - candidates[new_selected]
                dist_sq = np.sum(diff * diff, axis=1)
                min_d = np.min(dist_sq)
                
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_alt = alt_idx
            
            if best_alt is not None and best_min_d > current_min_dist_sq:
                selected[pos] = best_alt
                current_min_dist_sq = best_min_d
                improved = True
    
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
