import numpy as np
from itertools import product
import time

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 580  # Safety margin
    
    d = 11
    # Generate all 2^11 ±1 vectors
    all_vecs = np.array(list(product([0,1], repeat=d)), dtype=np.int64)
    all_vecs_pm1 = 1 - 2 * all_vecs
    
    # Try multiple random orderings, keep best
    best_selected = []
    rng = np.random.default_rng(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        perm = rng.permutation(len(all_vecs_pm1))
        shuffled = all_vecs_pm1[perm]
        
        selected = []
        for v in shuffled:
            if time.time() > deadline:
                break
            ok = True
            for s in selected:
                if np.sum(v != s) < 3:
                    ok = False
                    break
            if ok:
                selected.append(v)
        
        if len(selected) > len(best_selected):
            best_selected = selected
    
    if len(best_selected) < 2:
        return np.array([np.ones(d), -np.ones(d)], dtype=np.int64)
    
    return np.array(best_selected, dtype=np.int64)
