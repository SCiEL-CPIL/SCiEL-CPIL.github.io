import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    """Find the largest Sidon subset size in arr (12 elements)."""
    n = len(arr)
    # Check from largest possible down
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(range(n), k):
            subset = [arr[i] for i in comb]
            if is_sidon(subset):
                found = True
                break
        if found:
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    random.seed(42)
    
    # Try structured initializations that are likely to have small max Sidon
    structured_inits = [
        # Dense block 0-11 (likely max Sidon size 4)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Dense block 1-12
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # Two clusters
        [0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69],
        # Three clusters
        [0, 1, 2, 3, 43, 44, 45, 46, 86, 87, 88, 89],
        # Dense cluster + sparse
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128],
        # AP with small difference
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Mix of dense and spread
        [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 128],
    ]
    
    for init_set in structured_inits:
        if time.time() > deadline:
            break
        val = max_sidon_size(init_set)
        if val < best_val:
            best_val = val
            best_set = init_set[:]
            if best_val <= 4:
                break
    
    # Random search with local improvements
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initialization
        current_set = random.sample(range(129), 12)
        current_val = max_sidon_size(current_set)
        
        if current_val < best_val:
            best_val = current_val
            best_set = current_set[:]
            if best_val <= 4:
                break
        
        # Local search: try single element replacements
        for step in range(200):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            old_val = current_set[i]
            new_val = random.randint(0, 128)
            if new_val == old_val or new_val in current_set:
                continue
            
            current_set[i] = new_val
            new_val_result = max_sidon_size(current_set)
            
            if new_val_result < current_val:
                current_val = new_val_result
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                    if best_val <= 4:
                        break
            else:
                current_set[i] = old_val
            
            if best_val <= 4:
                break
    
    return best_set
