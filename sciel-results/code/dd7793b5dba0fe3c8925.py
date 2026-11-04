import itertools
import math
import time

def score_set(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Check seed
    best_score = score_set(best_set)
    
    # Brute force small sets
    # M up to 10
    for M in range(1, 11):
        if time.time() > deadline:
            break
        # Subsets of {0, ..., M} containing 0 and M (to maximize max_u)
        # Actually, we can just iterate all subsets of {0, ..., M} containing 0
        # But 2^10 = 1024. Easy.
        for mask in range(1, 1 << (M + 1)):
            if time.time() > deadline:
                break
            if not (mask & 1): # must contain 0
                continue
            U = [i for i in range(M + 1) if (mask >> i) & 1]
            # Ensure max is M? Not strictly, but if max < M, it's covered in smaller M
            if max(U) < M:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Try some heuristic larger sets
    # Geometric progression
    for r in range(2, 10):
        for k in range(3, 20):
            if time.time() > deadline:
                break
            U = [0] + [r**i for i in range(1, k)]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            
    # Triangular numbers
    for k in range(3, 20):
        if time.time() > deadline:
            break
        U = [i*(i+1)//2 for i in range(k)]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set
