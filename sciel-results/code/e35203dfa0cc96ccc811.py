import math
import itertools
import time

def score_set(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
    
    sums = set()
    diffs = set()
    for u1 in U:
        for u2 in U:
            sums.add(u1 + u2)
            diffs.add(u1 - u2)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0 # Or 0? Score is 1 + log(ratio)/log(...). If ratio<1, score<1.
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Brute force small sets
    # a=1 is likely best
    max_c = 50
    for b in range(2, 20):
        for c in range(b+1, max_c):
            U = [0, 1, b, c]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline:
                return best_set
                
    # Try 5 elements with best 4
    # ... (optional, might not improve)
    
    return best_set
