import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Generate all Norm 2 vectors (squared norm 2, weight 2)
    # There are C(11, 2) * 2^2 = 55 * 4 = 220 such vectors.
    # We proved that all 220 form a valid kissing configuration.
    # R^2 = 2.
    # Min dist^2 = 2 (when dot product is 1).
    # Condition: 2 >= 2. Holds.
    
    candidates = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates.append(v)
            
    V = np.array(candidates, dtype=np.int64)
    
    # Optional: Try to see if we can add Norm 4 vectors to a subset?
    # Given the analysis, mixing is detrimental.
    # We return the 220 points.
    
    return V
