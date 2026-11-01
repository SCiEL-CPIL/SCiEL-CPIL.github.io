import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    """
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: All vectors with exactly two non-zero entries (+/-1).
    # Norm^2 = 2. Max norm = sqrt(2).
    # Min distance between distinct such vectors is sqrt(2).
    # So all 220 are valid.
    candidates_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_2.append(v)
    
    baseline = np.array(candidates_2, dtype=np.int64)
    best_set = baseline
    best_size = len(baseline)
    
    # Try Norm^2 = 5 vectors (Type A: one entry +/-2, one entry +/-1).
    # Max norm = sqrt(5).
    # Need dist^2 >= 5.
    # dist^2 = 5 + 5 - 2 dot = 10 - 2 dot.
    # 10 - 2 dot >= 5 => 2 dot <= 5 => dot <= 2.
    # Conflict if dot >= 3.
    
    candidates_5 = []
    # Type A: One +/-2, One +/-1
    for i, j in itertools.combinations(range(D), 2):
        # v[i] is +/-2, v[j] is +/-1
        for si in (2, -2):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_5.append(v)
        # v[i] is +/-1, v[j] is +/-2
        for si in (1, -1):
            for sj in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates_5.append(v)
                
    candidates_5 = np.array(candidates_5, dtype=np.int64)
    n5 = len(candidates_5)
    
    if time.time() > deadline - 20:
        return best_set
        
    # Compute conflict matrix
    # dot >= 3 is conflict
    dots = candidates_5 @ candidates_5.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, False)
    
    np.random.seed(42)
    
    best_5_set = []
    
    # Greedy Independent Set
    max_restarts = 100
    for attempt in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        remaining = np.ones(n5, dtype=bool)
        degrees = conflict.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[remaining][:, actual_remove].sum(axis=1)
                degrees[remaining] -= deg_decrease
                
            remaining[actual_remove] = False
            
            if time.time() > deadline - 5:
                break
                
        if len(selected) > len(best_5_set):
            best_5_set = selected
            
    if len(best_5_set) > best_size:
        best_set = candidates_5[best_5_set]
        best_size = len(best_5_set)
        
    return best_set
