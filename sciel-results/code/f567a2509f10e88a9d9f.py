"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _largest_sidon_subset(s):
    """Find the largest subset of s with all pair sums (including diagonals) unique."""
    s = sorted(s)
    n = len(s)
    best = 0
    # Try all subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        ok = False
                        break
                    sums.add(sm)
                if not ok:
                    break
            if ok:
                best = size
                found = True
                break
        if found:
            break
    return best

def _evaluate(s):
    """Return the largest Sidon subset size for set s."""
    return _largest_sidon_subset(s)

def _random_candidate():
    return sorted(random.sample(range(129), 12))

def _local_search(current, deadline):
    """Hill-climbing: try swapping one element with another from [0,128]."""
    current = sorted(current)
    best_score = _evaluate(current)
    best_set = list(current)
    
    all_vals = set(range(129))
    in_set = set(current)
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            for new_val in all_vals - in_set:
                if time.time() > deadline:
                    break
                candidate = list(current)
                candidate[i] = new_val
                candidate = sorted(candidate)
                score = _evaluate(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate
                    improved = True
                    break
            if improved or time.time() > deadline:
                break
        if not improved:
            break
        current = best_set
        in_set = set(current)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to leave margin
    
    best_set = list(range(12))
    best_score = _evaluate(best_set)
    
    # Try several random restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        candidate = _random_candidate()
        score = _evaluate(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate
            # Local search from this candidate
            best_set, best_score = _local_search(best_set, deadline)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
