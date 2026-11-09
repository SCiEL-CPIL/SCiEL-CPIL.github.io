import time
import random

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def get_conflicts(sums):
    """
    sums is a list of ints (bitsets).
    Returns number of overlapping bits between different cardinalities.
    """
    conflicts = 0
    n = len(sums)
    for i in range(n):
        for j in range(i+1, n):
            conflicts += bin(sums[i] & sums[j]).count('1')
    return conflicts

def construct():
    # Try to find a valid set of size 10
    # If not found, return size 9 (powers of 2)
    
    # Fallback: Powers of 2 up to 256
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Check if fallback is valid
    # It is, as sums are unique.
    
    best_set = fallback
    best_conflicts = 0 # 0 means valid
    
    # Search for size 10
    n_target = 10
    max_val = 256
    
    # Initialize random set
    current_set = random.sample(range(1, max_val + 1), n_target)
    
    # Compute initial conflicts
    def compute_sums(S):
        n = len(S)
        sums = [0] * (n + 1)
        sums[0] = 1 # bit 0 is set
        for x in S:
            new_sums = sums[:]
            # Iterate backwards to avoid using x multiple times in same step
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    new_sums[k+1] |= (sums[k] << x)
            sums = new_sums
        return sums

    current_sums = compute_sums(current_set)
    current_conflicts = get_conflicts(current_sums)
    
    if current_conflicts == 0:
        return current_set

    best_conflicts = current_conflicts
    best_set = current_set

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while not is_done():
        if T < T_min:
            break
            
        # Mutate
        idx = random.randint(0, n_target - 1)
        old_val = current_set[idx]
        new_val = random.randint(1, max_val)
        
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        
        # Recompute sums and conflicts
        # Optimization: Only recompute if necessary? 
        # For n=10, compute_sums is fast (10 * 10 bit ops).
        new_sums = compute_sums(current_set)
        new_conflicts = get_conflicts(new_sums)
        
        if new_conflicts < current_conflicts:
            current_conflicts = new_conflicts
            if new_conflicts == 0:
                return current_set
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_set = current_set[:]
        else:
            # Accept with probability
            delta = new_conflicts - current_conflicts
            if delta > 0:
                prob = pow(2.718, -delta / T)
                if random.random() < prob:
                    current_conflicts = new_conflicts
                else:
                    # Reject
                    current_set[idx] = old_val
            else:
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_set = current_set[:]
        
        T *= cooling
        
    # Return best found
    # If best_conflicts is 0, it's valid.
    # If not, we must return a valid set.
    # The fallback is valid.
    if best_conflicts == 0:
        return best_set
    else:
        return fallback
