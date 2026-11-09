"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
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
        return 1.0  # Can't be below 1 in a meaningful way, but guard
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5 seconds buffer
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Try various structured constructions
    
    # Strategy 1: Powers of 2 scaled
    for base in range(2, 20):
        for k in range(3, 12):
            if time.time() - start_time > time_limit:
                break
            U = [base ** i for i in range(k)]
            if 0 not in U:
                U.append(0)
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = sorted(U)
    
    # Strategy 2: Sparse arithmetic-like sets
    for a in range(1, 50):
        for b in range(a + 1, a + 50):
            if time.time() - start_time > time_limit:
                break
            for k in range(3, 8):
                U = [0] + [a * i + b * (i * (i - 1) // 2) for i in range(1, k)]
                score = compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = sorted(set(U))
    
    # Strategy 3: Random sparse sets
    import random
    random.seed(42)
    for trial in range(5000):
        if time.time() - start_time > time_limit:
            break
        k = random.randint(4, 10)
        max_val = random.randint(10, 200)
        U = [0]
        remaining = max_val
        for i in range(k - 1):
            if remaining <= 0:
                break
            max_pick = remaining // (k - 1 - i) + 1
            if max_pick < 1:
                max_pick = 1
            pick = random.randint(1, max_pick)
            U.append(U[-1] + pick)
            remaining -= pick
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_set = sorted(set(U))
    
    # Strategy 4: Fibonacci-like or other recursive constructions
    for a in range(1, 10):
        for b in range(1, 10):
            if time.time() - start_time > time_limit:
                break
            for k in range(4, 12):
                U = [0, a, b]
                for i in range(3, k):
                    U.append(U[-1] + U[-2])
                score = compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = sorted(set(U))
    
    # Strategy 5: Sets with large gaps (geometric with varying ratios)
    for r1 in range(2, 8):
        for r2 in range(2, 8):
            if time.time() - start_time > time_limit:
                break
            for k in range(4, 10):
                U = [0]
                current = 1
                for i in range(k - 1):
                    current *= r1 if i % 2 == 0 else r2
                    U.append(current)
                score = compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = sorted(set(U))
    
    return best_set
# EVOLVE-BLOCK-END
