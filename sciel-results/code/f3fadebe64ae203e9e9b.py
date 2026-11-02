import numpy as np

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return -1
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
        
    log_ratio = np.log2(ratio)
    log_max = np.log2(2 * max_u + 1)
    
    if log_max == 0:
        return -1
        
    return 1 + log_ratio / log_max

def construct():
    best_score = -1
    best_set = [0, 1, 3]
    
    # Search small sets
    # We can try all subsets of {0..15} of size 3 to 5
    # This is manageable
    
    import itertools
    
    max_val = 15
    candidates = list(range(max_val + 1))
    
    # We must include 0
    others = candidates[1:]
    
    for size in range(3, 6):
        for comb in itertools.combinations(others, size - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    # Also try some larger random sets
    import random
    random.seed(42)
    for _ in range(1000):
        size = random.randint(3, 8)
        max_val_rand = random.randint(10, 50)
        U = random.sample(range(1, max_val_rand + 1), size - 1)
        U.append(0)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
