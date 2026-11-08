import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    # Try a few other candidates just in case
    candidates = [
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 10],
        [0, 1, 3, 9, 12],
        [0, 1, 3, 9, 18],
        [0, 1, 3, 9, 20],
        [0, 1, 3, 9, 30],
        [0, 1, 3, 9, 100],
        [0, 1, 3, 9, 1000],
        [0, 1, 3, 9, 3**4],
        [0, 1, 3, 9, 3**5],
        [0, 1, 3, 9, 3**6],
        [0, 1, 3, 9, 3**7],
    ]
    
    for U in candidates:
        if time.time() > deadline:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Also try small variations of the base set
    base = [0, 1, 3, 9]
    for add_val in range(10, 50):
        if time.time() > deadline:
            break
        U = base + [add_val]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    return best_U
