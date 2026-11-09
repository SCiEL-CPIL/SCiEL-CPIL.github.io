import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Add Type A points: 2*e_i and -2*e_i
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        points.append(p2)
        
    # Generate all 4-combinations of indices
    combos = list(itertools.combinations(range(d), 4))
    
    # Shuffle to potentially find a larger independent set
    # Using a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    rng.shuffle(combos)
    
    selected_combos = []
    
    # Greedy selection of combinations with pairwise intersection <= 2
    for c in combos:
        is_valid = True
        for s in selected_combos:
            # Check intersection size
            inter = set(c).intersection(s)
            if len(inter) > 2:
                is_valid = False
                break
        if is_valid:
            selected_combos.append(c)
            
    # Add Type B points for selected combinations
    # For each combo, add all 16 sign patterns of +1/-1 at those indices
    for c in selected_combos:
        indices = list(c)
        # Generate all 16 sign patterns
        for mask in range(16):
            p = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(indices):
                # Bit k of mask determines sign
                if (mask >> k) & 1:
                    p[idx] = -1
                else:
                    p[idx] = 1
            points.append(p)
            
    points_array = np.array(points, dtype=np.int64)
    
    # Validation (optional, but good for debugging if we were in dev)
    # k = len(points)
    # if k < 2: return np.array([])
    # ...
    
    return points_array
