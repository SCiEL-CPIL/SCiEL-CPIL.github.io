import numpy as np
from itertools import combinations
import time

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600  # 600s deadline
    
    all_4sets = list(combinations(range(d), 4))
    
    # Greedy selection of 4-sets such that no two share 3 or more indices
    # Try multiple random orderings to find a good packing
    import random
    
    best_supports = []
    num_tries = 50
    rng = random.Random(42)
    
    for trial in range(num_tries):
        if time.time() > deadline - 60:
            break
        shuffled = all_4sets[:]
        rng.shuffle(shuffled)
        selected_supports = []
        for s in shuffled:
            valid = True
            for s2 in selected_supports:
                inter = len(set(s) & set(s2))
                if inter >= 3:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
        if len(selected_supports) > len(best_supports):
            best_supports = selected_supports
    
    # Generate all sign combinations for each selected support
    points = []
    for s in best_supports:
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
                        
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
