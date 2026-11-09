import itertools
import random
import time
from math import log

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
            
    max_u = max(U)
    if max_u == 0:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
        
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
        
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Predefined candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 6, 12],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10],
        [0, 1, 5, 11],
    ]
    
    for c in candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Random search over subsets of {0, ..., 30}
    # We want to explore sets with small max U.
    # We will sample random subsets of {1, ..., 20} and add 0.
    
    base_elements = list(range(1, 21))
    
    # Try different sizes
    for n in range(2, 9):
        # Limit iterations based on time
        max_iters = 10000
        for _ in range(max_iters):
            if time.time() > deadline:
                return best_set
                
            # Randomly select n-1 elements from 1..20
            subset = random.sample(base_elements, n-1)
            U = [0] + sorted(subset)
            
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U

    # Local improvement: try adding or removing elements
    # This might be too slow, so we stick to random search + predefined.
    
    return best_set
