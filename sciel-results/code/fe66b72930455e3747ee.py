import itertools
import math
import time
import numpy as np

def _score_of(U):
    U = list(U)
    if not U or 0 not in U:
        return -1
    max_u = max(U)
    # Calculate U+U and U-U
    # Using sets for uniqueness
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return -1
        
    ratio = n_diff / n_sum
    # score = 1 + log(ratio) / log(2*max_u + 1)
    # if ratio <= 1, log is <= 0, score <= 1.
    # We want max score.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return -1
        
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    timeout = 100  # seconds, leave margin for 120s limit
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search small sets
    # n from 3 to 8
    # M from 10 to 40
    # To avoid excessive time, we can limit M based on n
    # or just iterate and break if time exceeds timeout
    
    # Pre-compute combinations? No, just iterate.
    
    for n in range(3, 9):
        for M in range(10, 41):
            # If time is up, break
            if time.time() - start_time > timeout:
                return best_set
            
            # Iterate combinations of size n-1 from 1..M
            # 0 is fixed
            # This is C(M, n-1)
            # For M=40, n=8, C(40, 7) = 18,643,560. Too many.
            # We need to be smarter.
            
            # Maybe limit M for larger n?
            # Or use random sampling for large M?
            
            # Let's restrict M to 20 for n>=6?
            # C(20, 5) = 15504. C(20, 6) = 38760. C(20, 7) = 77520.
            # This is fast.
            
            effective_M = M
            if n >= 6:
                effective_M = min(M, 25)
            if n >= 7:
                effective_M = min(M, 20)
                
            # Iterate
            # Using itertools.combinations
            for comb in itertools.combinations(range(1, effective_M + 1), n - 1):
                U = [0] + list(comb)
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                    
    return best_set
