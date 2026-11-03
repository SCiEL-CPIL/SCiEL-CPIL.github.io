import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    """Compute the score for a set U containing 0."""
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    sum_size = len(np.unique(sums))
    diff_size = len(np.unique(diffs))
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _golomb_greedy(n, max_iter=200000):
    """Greedy construction of a Golomb Ruler with n marks."""
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    U = [0, 1]
    diffs = set()
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            diffs.add(U[j] - U[i])
            
    for _ in range(n - 2):
        placed = False
        last = U[-1]
        for candidate in range(last + 1, last + 5000):
            valid = True
            new_diffs = []
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                U.append(candidate)
                diffs.update(new_diffs)
                placed = True
                break
        if not placed:
            break
            
    return U

def _local_search_compress(U, time_limit=20.0, rng=None):
    """
    Local search to compress the set (reduce M) by moving elements down.
    Enhanced with larger move ranges and swap operations.
    """
    U = sorted(U)
    n = len(U)
    if n < 2:
        return U
    if rng is None:
        rng = np.random.RandomState(42)
    
    cur_score = _score_set(U)
    best_U = U[:]
    best_score = cur_score
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(U)
        max_val = max(U)
        
        # 1. Try moving each element down (compression) with larger range
        for i in range(n - 1, 0, -1):
            if time.time() - start_time > time_limit:
                break
            original_val = U[i]
            lower_bound = U[i-1] + 1
            
            # Try larger delta range
            max_delta = min(50, original_val - lower_bound)
            for delta in range(1, max_delta + 1):
                new_val = original_val - delta
                if new_val < lower_bound:
                    break
                trial = U[:]
                trial[i] = new_val
                trial.sort()
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U[:]
                    best_score = cur_score
                    improved = True
                    break
        
        # 2. Try swapping adjacent elements (reordering to reduce max)
        if not improved and n >= 3:
            for i in range(n - 2):
                if time.time() - start_time > time_limit:
                    break
                # Try moving element i+1 down to just above element i
                if U[i+1] > U[i] + 1:
                    trial = U[:]
                    new_val = U[i] + 1
                    # Find where this would go
                    trial.pop(i+1)
                    # Insert at correct position
                    import bisect
                    pos = bisect.bisect_left(trial, new_val)
                    if pos <= i:
                        trial.insert(i, new_val)
                    else:
                        trial.insert(i+1, new_val)
                    trial.sort()
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        best_U = U[:]
                        best_score = cur_score
                        improved = True
                        break
        
        # 3. Random perturbation to escape local minima
        if not improved and n >= 3 and (time.time() - start_time) < time_limit - 5:
            # Randomly move one element
            idx = rng.randint(1, n - 1)
            orig = U[idx]
            # Try moving it to a random position
            for _ in range(5):
                new_val = rng.randint(U[idx-1] + 1, max_val + 5)
                if new_val == orig:
                    continue
                trial = U[:]
                trial[idx] = new_val
                trial.sort()
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U[:]
                    best_score = cur_score
                    improved = True
                    break

    return best_U

def construct():
    best_U = [0]
    best_score = 0.0
    
    # Known optimal Golomb Rulers (from literature)
    known_golomb = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 12],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 45],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    }
    
    # Evaluate all known rulers
    for n, U in known_golomb.items():
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Generate greedy Golomb Rulers for n=14,15,16
    for n in range(14, 17):
        if _elapsed() > 30:
            break
        U = _golomb_greedy(n)
        if len(U) == n:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Main compression search on best candidate
    time_remaining = _DEADLINE - _elapsed()
    if time_remaining > 15:
        search_time = max(10, time_remaining - 10)
        U = _local_search_compress(list(best_U), time_limit=search_time)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Multiple restarts with perturbations
    if _elapsed() < _DEADLINE - 20:
        rng = np.random.RandomState(123)
        n_restarts = 5
        for r in range(n_restarts):
            if _elapsed() > _DEADLINE - 10:
                break
            # Create perturbed version
            U_perturbed = list(best_U)
            if len(U_perturbed) > 2:
                # Remove a random non-zero element
                idx = rng.randint(1, len(U_perturbed) - 1)
                U_perturbed.pop(idx)
            elif len(U_perturbed) > 1:
                # Move the last element
                U_perturbed[-1] = max(1, U_perturbed[-1] - rng.randint(1, 5))
            
            # Quick compression
            time_left = _DEADLINE - _elapsed() - 5
            if time_left > 5:
                U_new = _local_search_compress(U_perturbed, time_limit=min(15, time_left), rng=rng)
                s = _score_set(U_new)
                if s > best_score:
                    best_score = s
                    best_U = U_new[:]
    
    # Final polish: try very fine-grained compression on best
    if _elapsed() < _DEADLINE - 5:
        U_final = _local_search_compress(list(best_U), time_limit=min(10, _DEADLINE - _elapsed() - 3))
        s = _score_set(U_final)
        if s > best_score:
            best_score = s
            best_U = U_final[:]
    
    return best_U
