import numpy as np
import itertools
import random

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
    best_set = [0, 1, 3, 7]
    
    # 1. Exhaustive search for small N and small max_val
    # N=3: C(50,2) = 1225
    # N=4: C(50,3) = 19600
    # N=5: C(50,4) = 230300
    # N=6: C(50,5) = 2118760 (might be slow, let's limit to 30)
    
    max_val_exhaustive = 30
    
    for n in range(3, 7):
        k = n - 1
        if k > max_val_exhaustive:
            continue
        for comb in itertools.combinations(range(1, max_val_exhaustive + 1), k):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # 2. Randomized Local Search for larger N or larger values
    # We try to find sets with N=5, 6, 7, 8
    random.seed(42)
    
    for target_n in range(5, 9):
        # Start with a random set
        current_set = [0] + random.sample(range(1, 200), target_n - 1)
        current_score = score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
            
        # Hill climbing
        no_improvement = 0
        max_iter = 5000
        for i in range(max_iter):
            if no_improvement > 500:
                break
                
            # Pick a random element to swap
            idx = random.randint(1, len(current_set) - 1) # Don't swap 0
            old_val = current_set[idx]
            
            # Generate a new value
            # Bias towards smaller values or values close to existing ones?
            # Let's try uniform in 1..300
            new_val = random.randint(1, 300)
            
            if new_val == old_val:
                continue
                
            # Check if new_val is already in the set
            if new_val in current_set:
                continue
                
            # Create new set
            new_set = current_set[:]
            new_set[idx] = new_val
            new_set.sort()
            
            new_score = score_set(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                no_improvement = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                no_improvement += 1

    return best_set
