"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sum set and difference set
    U_arr = np.array(U, dtype=np.int64)
    # Sum set
    sums = U_arr[:, None] + U_arr[None, :]
    sum_set = np.unique(sums)
    # Difference set
    diffs = U_arr[:, None] - U_arr[None, :]
    diff_set = np.unique(diffs)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return 0.0
    import math
    log_ratio = math.log(n_diff / n_sum)
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_score = _score_set([0, 1, 3])
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search over subsets of small ranges
    # For each max range, try all subsets containing 0
    for max_val in range(3, 16):
        if time.time() > deadline:
            break
        # Try subsets of {0, 1, ..., max_val} that contain 0
        # For efficiency, iterate over subsets of {1, ..., max_val}
        others = list(range(1, max_val + 1))
        n = len(others)
        # Try all sizes from 2 to min(n, 8) elements
        for size in range(2, min(n, 8) + 1):
            if time.time() > deadline:
                break
            for subset in itertools.combinations(others, size):
                U = [0] + list(subset)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                if time.time() > deadline:
                    break
    
    # Phase 2: Local improvement - try adding/removing elements near the best set
    if time.time() < deadline:
        current = list(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            max_u = max(current)
            # Try adding elements
            for candidate in range(max_u + 1, max_u + 30):
                if time.time() > deadline:
                    break
                test_set = current + [candidate]
                s = _score_set(test_set)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = test_set
                    current = test_set
                    improved = True
                    break
            # Try removing non-zero elements
            for i in range(len(current) - 1, 0, -1):
                if time.time() > deadline:
                    break
                test_set = current[:i] + current[i+1:]
                if len(test_set) < 2:
                    continue
                s = _score_set(test_set)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = test_set
                    current = test_set
                    improved = True
                    break
            # Try replacing elements
            for i in range(1, len(current)):
                if time.time() > deadline:
                    break
                for candidate in range(1, max_u + 30):
                    if candidate == current[i]:
                        continue
                    test_set = current[:i] + [candidate] + current[i+1:]
                    if len(set(test_set)) != len(test_set):
                        continue
                    s = _score_set(test_set)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = test_set
                        current = test_set
                        improved = True
                        break
                if improved:
                    break
    
    # Phase 3: Try structured patterns (arithmetic progressions, geometric)
    if time.time() < deadline:
        # Geometric progressions
        for base in range(2, 20):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for _ in range(15):
                U.append(val)
                val *= base
                if val > 10**6:
                    break
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
        
        # Mixed progressions
        for base1 in range(2, 8):
            if time.time() > deadline:
                break
            for base2 in range(base1 + 1, base1 + 10):
                if time.time() > deadline:
                    break
                U = [0]
                val = 1
                for i in range(10):
                    U.append(val)
                    val *= base1 if i % 2 == 0 else base2
                    if val > 10**6:
                        break
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
