import numpy as np
from itertools import combinations
import random

def construct():
    D = 11
    # Use 5-sparse vectors
    # Norm squared = 5
    # Distance squared >= 5 required
    # 10 - 2*dot >= 5 => dot <= 2.5 => dot <= 2
    
    all_subsets = list(combinations(range(D), 5))
    
    # Generate all candidate vectors (subset, signs)
    candidates = []
    for subset in all_subsets:
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        for s5 in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[list(subset)] = [s1, s2, s3, s4, s5]
                            candidates.append(v)
                            
    # Shuffle candidates to break symmetry
    random.shuffle(candidates)
    
    selected = []
    # For efficiency, keep the selected vectors as a list of arrays
    # Check compatibility: dot <= 2
    
    for v in candidates:
        # Check against all selected
        # Optimization: if len(selected) is large, this is slow.
        # But 120s limit is generous for ~10k vectors.
        is_valid = True
        for u in selected:
            if np.dot(v, u) > 2:
                is_valid = False
                break
        if is_valid:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
