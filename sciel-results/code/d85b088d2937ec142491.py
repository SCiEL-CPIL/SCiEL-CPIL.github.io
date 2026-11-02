"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate all candidate vectors of squared norm 4
    candidates = []
    
    # Type A: (+-2, 0, ..., 0)
    for i in range(D):
        for sign in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign * 2
            candidates.append(v)
            
    # Type B: (+-1, +-1, +-1, +-1, 0, ...)
    # There are C(11,4) = 330 supports.
    # For each support, 2^4 = 16 sign patterns.
    supports = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    supports.append((i, j, k, l))
                    
    for supp in supports:
        i, j, k, l = supp
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        candidates.append(v)
                        
    # Total candidates: 22 + 330*16 = 5302
    
    best_set = []
    best_size = 0
    
    # Greedy algorithm with multiple restarts
    num_restarts = 100000 # Will be limited by time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        random.shuffle(candidates)
        
        current_set = []
        # For speed, we can just iterate and check
        # Conflict condition: dist^2 < 4
        # dist^2 = 8 - 2*dot for Type B vs Type B
        # dist^2 = 4 + 4 - 2*dot for Type A vs Type B?
        # Type A norm^2 = 4. Type B norm^2 = 4.
        # dist^2 = 8 - 2*dot.
        # Condition dist^2 >= 4 => dot <= 2.
        
        for v in candidates:
            # Check if v conflicts with any in current_set
            valid = True
            for u in current_set:
                dot = np.dot(u, v)
                if dot > 2:
                    valid = False
                    break
            
            if valid:
                current_set.append(v)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we reached the theoretical maximum, we can stop
            if best_size == 504:
                break

    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
