import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Generate candidate vectors: all weight-2 vectors (±ei ± ej)
    # Plus try to add more via random search
    candidates = []
    
    # All weight-2 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Also add weight-1 vectors (±ei) - but need to check compatibility
    # Actually, let's first build with weight-2, then try adding others
    
    # Greedy selection from candidates
    selected = []
    norms_sq = []  # squared norms
    
    # Sort candidates by some heuristic - maybe just use them as is
    # For weight-2 vectors, all have norm_sq = 2
    
    for v in candidates:
        if time.time() > deadline:
            break
        n_sq = int(np.dot(v, v))
        
        # Check against all selected vectors
        ok = True
        for s in selected:
            diff = v - s
            dist_sq = int(np.dot(diff, diff))
            # Need dist >= max_norm, i.e., dist_sq >= max_norm_sq
            max_norm_sq = max(norms_sq) if norms_sq else 0
            if dist_sq < max_norm_sq:
                ok = False
                break
        
        if ok:
            selected.append(v)
            norms_sq.append(n_sq)
    
    # Now try to add more vectors via random search
    # Generate random vectors with entries in {0, ±1} of weight 2 or 3
    random.seed(42)
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 10000 == 0 and time.time() > deadline:
            break
        
        # Random weight-2 vector
        if random.random() < 0.7:
            i, j = random.sample(range(D), 2)
            v = np.zeros(D, dtype=np.int64)
            v[i] = random.choice([1, -1])
            v[j] = random.choice([1, -1])
        else:
            # Weight-3 vector
            idxs = random.sample(range(D), 3)
            v = np.zeros(D, dtype=np.int64)
            for idx in idxs:
                v[idx] = random.choice([1, -1])
        
        n_sq = int(np.dot(v, v))
        
        # Check against all selected
        ok = True
        max_norm_sq = max(norms_sq) if norms_sq else 0
        for s in selected:
            diff = v - s
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < max_norm_sq:
                ok = False
                break
        
        if ok:
            selected.append(v)
            norms_sq.append(n_sq)
    
    if not selected:
        # Fallback: return the basic 220
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
