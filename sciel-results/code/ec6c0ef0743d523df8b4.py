import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    best_score = -1
    best_set = [0, 1]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Exhaustive search for small sets
    max_n = 7
    max_m = 40
    
    for n in range(2, max_n + 1):
        if time.time() > deadline - 10:
            break
        # Choose n-1 elements from 1..max_m
        for combo in combinations(range(1, max_m + 1), n - 1):
            if time.time() > deadline - 10:
                break
            cand = [0] + list(combo)
            try_set(cand)

    return list(best_set)
