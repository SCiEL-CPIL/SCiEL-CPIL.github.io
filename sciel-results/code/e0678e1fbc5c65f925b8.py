"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit; keep safety margin
_TIME_MARGIN = 15.0  # seconds to finish before the kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _score_of(U):
    """
    Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    U is a list of nonnegative integers containing 0.
    """
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    # Use sets for efficiency
    U_set = set(U)
    # U+U: {a+b for a in U for b in U}
    # U-U: {a-b for a in U for b in U}
    # For large sets, this can be expensive. Use numpy for vectorized ops if needed.
    # For now, use set comprehensions which are reasonably fast for moderate sizes.
    
    # To speed up, we can use the fact that U contains 0.
    # U+U = {0} union {a+b for a,b in U, a>0 or b>0}
    # But let's just compute directly for correctness, then optimize.
    
    # Optimization: use sorted list and two-pointer or just set comprehensions.
    # For |U| up to ~2000, |U|^2 = 4e6 operations, which is fine in Python if vectorized.
    
    # Let's use numpy for the sum and difference sets.
    import numpy as np
    U_arr = np.array(U, dtype=np.int64)
    
    # U+U
    # This creates an |U|x|U| array, which can be large.
    # For |U|=2000, that's 4e6 elements, which is fine.
    # For |U|=5000, that's 25e6, still okay.
    
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    sum_set = set(sums.flatten())
    diff_set = set(diffs.flatten())
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
    
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_max
    return score

def _local_search(U, max_iters=5000, time_budget=None):
    """
    Local search: try adding/removing elements to improve score.
    """
    U = list(set(U))
    U.sort()
    if 0 not in U:
        U.append(0)
        U.sort()
    
    current_score = _score_of(U)
    best_U = list(U)
    best_score = current_score
    
    max_u = max(U) if U else 0
    # Consider elements up to some range
    max_range = max_u * 2 + 100
    
    if time_budget is None:
        time_budget = _time_left() / 2.0
    
    start = time.time()
    iters = 0
    while iters < max_iters and time.time() - start < time_budget:
        iters += 1
        if _time_left() < _TIME_MARGIN:
            break
        
        # Try adding a random element
        if random.random() < 0.5:
            # Add a new element
            if max_range > 0:
                new_elem = random.randint(1, max_range)
                if new_elem not in U:
                    new_U = list(U) + [new_elem]
                    new_U.sort()
                    s = _score_of(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        max_u = max(U)
                        max_range = max_u * 2 + 100
        else:
            # Remove a random element (not 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)  # don't remove 0
                new_U = U[:idx] + U[idx+1:]
                s = _score_of(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    max_u = max(U) if U else 0
                    max_range = max_u * 2 + 100
        
        # Also try small perturbations: add element near existing ones
        if random.random() < 0.3 and U:
            base = random.choice(U)
            delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 7, 10])
            new_elem = base + delta
            if new_elem >= 0 and new_elem not in U:
                new_U = list(U) + [new_elem]
                new_U.sort()
                s = _score_of(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    max_u = max(U)
                    max_range = max_u * 2 + 100
    
    return best_U, best_score

def _structured_candidates():
    """
    Generate structured candidate sets.
    """
    candidates = []
    
    # Small structured sets
    for n in range(3, 15):
        # Arithmetic progression
        U = list(range(n))
        candidates.append(U)
        
        # Geometric-like
        U = [0]
        val = 1
        for _ in range(n-1):
            U.append(val)
            val = val * 2 + 1
        candidates.append(U)
    
    # Sets based on binary representations
    for bits in range(4, 10):
        U = list(range(2**bits))
        candidates.append(U)
    
    # Sparse sets
    for n in range(10, 100, 10):
        U = [0]
        val = 1
        for _ in range(n-1):
            U.append(val)
            val += random.randint(1, 5)
        candidates.append(U)
    
    return candidates

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Try structured candidates
    candidates = _structured_candidates()
    for cand in candidates:
        if _time_left() < _TIME_MARGIN:
            break
        cand = sorted(set(cand))
        if 0 not in cand:
            cand.append(0)
            cand.sort()
        s = _score_of(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    # Local search from best candidate
    if _time_left() > _TIME_MARGIN + 5:
        U, s = _local_search(best_U, max_iters=10000, time_budget=_time_left() * 0.5)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Try random restarts
    if _time_left() > _TIME_MARGIN + 5:
        for _ in range(5):
            if _time_left() < _TIME_MARGIN + 5:
                break
            # Random initial set
            size = random.randint(5, 30)
            max_val = random.randint(50, 500)
            U = [0] + random.sample(range(1, max_val), min(size-1, max_val-1))
            U.sort()
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
            
            # Local search from this random start
            if _time_left() > _TIME_MARGIN + 5:
                U2, s2 = _local_search(best_U, max_iters=5000, time_budget=_time_left() * 0.3)
                if s2 > best_score:
                    best_score = s2
                    best_U = U2
    
    # Final polish: try to improve with more iterations if time allows
    if _time_left() > _TIME_MARGIN + 10:
        U, s = _local_search(best_U, max_iters=20000, time_budget=_time_left() * 0.6)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U
# EVOLVE-BLOCK-END
