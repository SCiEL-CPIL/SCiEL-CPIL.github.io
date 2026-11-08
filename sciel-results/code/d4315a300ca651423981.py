"""Direct construction using randomized greedy set packing on D11 lattice vectors."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # Generate candidate vectors: all weight-2 and weight-3 integer vectors
    candidates = []
    
    # Weight-2 vectors: norm^2 = 2
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Weight-3 vectors: norm^2 = 3
    for i, j, k in combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    n_cand = len(candidates)
    norms_sq = np.array([int(np.dot(c, c)) for c in candidates])
    
    best_result = None
    best_count = 0
    
    # Try multiple random orderings
    rng = np.random.default_rng(42)
    n_trials = 50
    
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Random permutation
        perm = rng.permutation(n_cand)
        
        selected = []
        selected_norms_sq = []
        max_norm_sq = 0
        
        for idx in perm:
            v = candidates[idx]
            vsq = int(norms_sq[idx])
            
            # New max norm if we add this
            new_max = max(max_norm_sq, vsq)
            
            compatible = True
            for i, sv in enumerate(selected):
                ssq = selected_norms_sq[i]
                dot = int(np.dot(v, sv))
                dist_sq = vsq + ssq - 2 * dot
                if dist_sq < new_max:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_norms_sq.append(vsq)
                max_norm_sq = new_max
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = np.array(selected, dtype=np.int64)
    
    if best_result is None or best_count < 220:
        # Fallback: all weight-2 vectors (known to work)
        selected = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        selected.append(v)
        best_result = np.array(selected, dtype=np.int64)
    
    return best_result
# EVOLVE-BLOCK-END
