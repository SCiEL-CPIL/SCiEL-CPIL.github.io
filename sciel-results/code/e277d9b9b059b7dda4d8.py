import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    
    all_candidates = np.array(v4_list + v1_list, dtype=np.int64)
    N = len(all_candidates)
    
    # Compute conflict matrix: two vectors conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Multi-restart randomized greedy MIS
    best_selected = []
    best_size = 0
    
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        rng = np.random.default_rng(12345 + restart)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32).copy()
        selected = []
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Randomized selection from low-degree vertices
            degs = degrees[alive_indices]
            # Pick from bottom 15% of degrees for diversity
            threshold = np.percentile(degs, 15)
            cands = alive_indices[degs <= threshold]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflicts[chosen] & alive)[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    return all_candidates[best_selected]
