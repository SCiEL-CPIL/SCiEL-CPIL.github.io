"""Optimized search for finite set U maximizing sumset/difference-set exponent."""
import time
import itertools
from functools import lru_cache

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _score_set(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1.0
    maxU = max(U)
    if maxU == 0:
        return 0.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return -1.0
    import math
    return 1.0 + math.log(n_diff / n_sum) / math.log(2 * maxU + 1)

def _try_set(U):
    """Try a specific set and return its score."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    return _score_set(U)

def _search_small_sets(deadline):
    """Exhaustive search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    # Search sets of size 1 to 6 with max element up to a reasonable bound
    # For size k, we choose k-1 positive integers from {1, ..., max_val}
    
    for size in range(2, 7):
        # For each size, try all combinations of size-1 positive integers
        # We need to bound max_val. For small sets, max_val up to ~20 should suffice
        max_val = 25 if size <= 4 else 18
        
        # Generate combinations of size-1 elements from 1..max_val
        # This is C(max_val, size-1) combinations
        if size - 1 > max_val:
            continue
            
        # For efficiency, iterate over combinations
        for combo in itertools.combinations(range(1, max_val + 1), size - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
        if time.time() > deadline:
            break
    
    return best_score, best_set

def _local_improvement(U, deadline):
    """Try to improve a set by adding or removing elements."""
    best_score = _score_set(U)
    best_set = U[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        U_set = set(U)
        maxU = max(U_set)
        
        # Try adding a new element
        for new_elem in range(1, maxU + 10):
            if new_elem in U_set:
                continue
            new_U = sorted(U_set | {new_elem})
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U_set = set(new_U)
                improved = True
                break
        
        # Try removing an element (except 0)
        for elem in list(U_set):
            if elem == 0:
                continue
            new_U = sorted(U_set - {elem})
            if len(new_U) < 2:
                continue
            s = _score_set(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_set = new_U
                U_set = set(new_U)
                improved = True
                break
    
    return best_score, best_set

def construct():
    deadline = _START_TIME + _TIME_LIMIT
    
    # Start with a good initial guess
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search over small sets
    s, U = _search_small_sets(deadline - 5.0)
    if s > best_score:
        best_score = s
        best_set = U
    
    # Phase 2: Local improvement from the best found
    if time.time() < deadline - 5.0:
        s, U = _local_improvement(best_set, deadline - 5.0)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Also try some structured sets as starting points for local improvement
    structured_starts = [
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 5],
        [0, 3, 7],
        [0, 1, 2, 5],
        [0, 2, 3, 7],
    ]
    for start in structured_starts:
        if time.time() > deadline - 5.0:
            break
        s, U = _local_improvement(start, deadline - 5.0)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_set
