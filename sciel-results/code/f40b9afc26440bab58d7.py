"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    best_vectors = []
    best_k = 0
    
    # Try norm^2 = 1, 2, 3, 4 separately
    for target_norm_sq in [1, 2, 3, 4]:
        candidates = []
        
        if target_norm_sq == 1:
            for i in range(D):
                for s in (1, -1):
                    v = [0] * D
                    v[i] = s
                    candidates.append(v)
        elif target_norm_sq == 2:
            for i, j in combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        candidates.append(v)
        elif target_norm_sq == 3:
            for i, j, k in combinations(range(D), 3):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(v)
        elif target_norm_sq == 4:
            # Four ±1 entries
            for i, j, k, l in combinations(range(D), 4):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            for sl in (1, -1):
                                v = [0] * D
                                v[i], v[j], v[k], v[l] = si, sj, sk, sl
                                candidates.append(v)
        
        if not candidates:
            continue
        
        candidates = np.array(candidates, dtype=np.int64)
        n = len(candidates)
        
        # Greedy selection
        selected = []
        max_norm_sq = target_norm_sq  # All have same norm
        
        for v in candidates:
            if not selected:
                selected.append(v)
                continue
            
            selected_arr = np.array(selected)
            diffs = selected_arr - v
            dists_sq = np.sum(diffs ** 2, axis=1)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq >= max_norm_sq:
                selected.append(v)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_vectors = selected
    
    if not best_vectors:
        # Fallback: D11 roots
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors
    
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
