"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Use numpy for fast sumset and diffset computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sumset = np.unique(sums)
    
    # Difference set: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset = np.unique(diffs)
    
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def _try_geometric(n_terms, base):
    """Try a geometric progression set."""
    if n_terms < 2:
        return None
    U = [0]
    val = 1
    for i in range(1, n_terms):
        val *= base
        U.append(val)
    return U

def _try_arithmetic(n_terms, step):
    """Try an arithmetic progression set."""
    if n_terms < 2:
        return None
    return list(range(0, n_terms * step, step))

def _random_search(deadline, best_U, best_score):
    """Random search for better sets."""
    import random
    
    while time.time() < deadline:
        # Try random sets of various sizes
        n = random.randint(4, 12)
        
        # Strategy 1: geometric-like with random bases
        if random.random() < 0.3:
            base = random.randint(2, 5)
            U = _try_geometric(n, base)
            if U is None:
                continue
        elif random.random() < 0.6:
            # Random increasing sequence
            U = [0]
            current = random.randint(1, 5)
            for _ in range(n - 1):
                current += random.randint(1, max(1, current // 2))
                U.append(current)
        else:
            # Random set with structure
            U = [0]
            for _ in range(n - 1):
                U.append(random.randint(1, 100))
            U = sorted(set(U))
        
        if len(U) < 2:
            continue
        
        score = _compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try systematic geometric progressions
    for base in range(2, 10):
        for n_terms in range(3, 10):
            if time.time() > deadline:
                break
            U = _try_geometric(n_terms, base)
            if U is None:
                continue
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U[:]
    
    # Try some structured random sets
    best_U, best_score = _random_search(deadline, best_U, best_score)
    
    return best_U

# EVOLVE-BLOCK-END
