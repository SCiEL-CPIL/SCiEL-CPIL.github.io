import numpy as np
import time
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
# Precompute squared distances
coords = np.array(POINTS)
# D[i, j] = squared distance between POINTS[i] and POINTS[j]
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

def construct():
    start_time = time.time()
    deadline = 280  # Safety margin for 300s limit
    
    best_set = None
    best_size = float('inf')
    
    # Initial greedy to have a baseline
    # But random restarts will likely find better
    
    while time.time() - start_time < deadline:
        perm = np.random.permutation(N_POINTS)
        S = []
        S_set = set()
        
        for p_idx in perm:
            if len(S) >= best_size:
                # We can't beat the best size if we already have best_size points
                # Actually, if we have best_size points, we stop this run?
                # No, we need to finish to ensure maximality?
                # No, the score is based on the final set size.
                # If we are constructing a set, and we already have 'best_size' points,
                # and we want a set SMALLER than best_size, we should stop adding?
                # No, we are building a maximal set. We MUST consider all points.
                # If we stop early, the set is not maximal.
                # However, if we know that the final size will be >= best_size, we can abort.
                # Can we know that?
                # If we have 'k' points, the final size will be at least 'k'.
                # If 'k' >= best_size, we cannot get a smaller set.
                # So we can break and discard this run.
                break
            
            # Check compatibility
            if len(S) == 0:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
            if len(S) == 1:
                S.append(p_idx)
                S_set.add(p_idx)
                continue
                
            S_list = list(S_set)
            # dist from p to all in S
            d_p_s = D[p_idx, S_list]
            
            # Check 1: Distances from P to S are unique
            if len(np.unique(d_p_s)) < len(d_p_s):
                continue
                
            # Check 2 & 3
            # Submatrix of distances between points in S
            # We can reuse this if we store it, but S changes.
            # For small S, recomputing is fine.
            D_S = D[np.ix_(S_list, S_list)]
            
            # Condition: D_S[i, j] == d_p_s[i]
            if np.any(D_S == d_p_s[:, None]):
                continue
            
            # Condition: D_S[i, j] == d_p_s[j]
            if np.any(D_S == d_p_s[None, :]):
                continue
            
            S.append(p_idx)
            S_set.add(p_idx)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
            
    if best_set is None:
        # Fallback to simple greedy if no time for random
        best_set = []
        for p in POINTS:
             # Simple check
             pass
             
    return best_set
