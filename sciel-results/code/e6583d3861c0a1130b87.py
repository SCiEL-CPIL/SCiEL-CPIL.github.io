import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def local_search(U, max_iter=1000, deadline=None):
    """
    Local search to improve a Golomb-like set.
    Tries to reduce max(U) while maintaining distinct differences (or high ratio).
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
        
    best = current
    best_score = current_score
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        n = len(current)
        # Try to move an element to a smaller value
        # This tends to reduce max(U)
        i = random.randint(1, n - 1)
        old_val = current[i]
        
        # Try a random new value smaller than old_val
        # Or a random value in a small range
        if random.random() < 0.7:
            new_val = random.randint(1, old_val - 1) if old_val > 1 else 1
        else:
            new_val = old_val + random.randint(-5, 5)
            if new_val < 1:
                new_val = 1
                
        new_set = current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        
        # If size changed, skip (collision)
        if len(new_set) != n:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best = new_set
            
        if new_score >= current_score:
            current = new_set
            current_score = new_score
        else:
            # Accept with some probability to escape local minima
            if random.random() < 0.1:
                current = new_set
                current_score = new_score
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known optimal or near-optimal Golomb rulers
    # These minimize max(U) for a given n, which helps the denominator.
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72, 78],
    ]
    
    best_set = None
    best_score = -1
    
    # Evaluate all hardcoded rulers
    for U in golomb_rulers:
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Local search on the best few candidates
    # We want to see if we can find a set with slightly larger M but better ratio,
    # or a set with smaller M (if the hardcoded ones are not perfectly optimal for the score metric)
    
    candidates = sorted(golomb_rulers, key=lambda x: fast_score(x), reverse=True)[:5]
    
    for U in candidates:
        if time.time() > deadline - 10:
            break
        refined, refined_score = local_search(U, max_iter=2000, deadline=deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # If no set found (should not happen), fallback
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
        
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
