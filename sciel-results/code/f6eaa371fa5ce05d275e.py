import itertools
import math
import time

def score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = set(U)
    sums = {a+b for a in U for b in U}
    diffs = {a-b for a in U for b in U}
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def find_best_set(max_size=6, max_val=20, timeout=100):
    start_time = time.time()
    best_score = 0
    best_set = [0, 1, 3]
    
    # Check size 3
    for combo in itertools.combinations(range(1, max_val+1), 2):
        U = [0] + list(combo)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Check size 4
    for combo in itertools.combinations(range(1, max_val+1), 3):
        if time.time() - start_time > timeout:
            break
        U = [0] + list(combo)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Check size 5
    for combo in itertools.combinations(range(1, max_val+1), 4):
        if time.time() - start_time > timeout:
            break
        U = [0] + list(combo)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Check size 6
    for combo in itertools.combinations(range(1, max_val+1), 5):
        if time.time() - start_time > timeout:
            break
        U = [0] + list(combo)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set

def construct():
    # Hardcoded best found so far
    return [0, 1, 3, 5]
    
    # If we want to search at runtime:
    # return find_best_set()
