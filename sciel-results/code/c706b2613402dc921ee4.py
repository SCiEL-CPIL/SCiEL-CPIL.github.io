"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_set(s):
    """Count distinct d > 0 with a 3-term AP in set s."""
    s_set = set(s)
    count = 0
    max_d = 254  # (512-3)//2
    for d in range(1, max_d + 1):
        found = False
        for a in s_set:
            if a + 2 * d > 512:
                break
            if (a + d) in s_set and (a + 2 * d) in s_set:
                found = True
                break
        if found:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    N = 512
    SIZE = 32
    max_d = 254
    
    # Best solution
    best_set = list(range(1, 33))
    best_score = _score_set(best_set)
    
    # SA
    current = best_set[:]
    current_score = best_score
    current_set_obj = set(current)
    
    T = 3.0
    T_min = 0.01
    cooling = 0.99999
    rng = random.Random(999)
    
    while time.time() < deadline and T > T_min:
        i = rng.randint(0, SIZE - 1)
        old_val = current[i]
        
        new_val = rng.randint(1, N)
        guard = 0
        while new_val in current_set_obj and guard < 50:
            new_val = rng.randint(1, N)
            guard += 1
        if new_val in current_set_obj:
            continue
        
        # Try swap
        current[i] = new_val
        current_set_obj.discard(old_val)
        current_set_obj.add(new_val)
        
        new_score = _score_set(current)
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = current[:]
        else:
            current[i] = old_val
            current_set_obj.discard(new_val)
            current_set_obj.add(old_val)
        
        T *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
