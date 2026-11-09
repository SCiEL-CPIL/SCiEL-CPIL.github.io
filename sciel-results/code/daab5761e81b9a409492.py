"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log

_start_time = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _compute_score(U):
    """Compute the score for a given set U (must contain 0, nonnegative ints)."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted([0] + U)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
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
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denom = log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denom

def _try_set(U):
    """Try a set, return (score, U) if valid."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted([0] + U)
    if len(U) < 2:
        return 0.0, U
    if min(U) < 0:
        return 0.0, U
    return _compute_score(U), U

def construct():
    global _start_time
    best_score = 0.0
    best_set = [0, 1]
    
    # Try some known good constructions first
    candidates = []
    
    # Small exhaustive search for n=2,3,4
    # n=2: {0, a}
    for a in range(1, 20):
        U = [0, a]
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # n=3: {0, a, b}
    for a in range(1, 15):
        for b in range(a+1, 30):
            U = [0, a, b]
            s, _ = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # n=4: {0, a, b, c}
    for a in range(1, 12):
        for b in range(a+1, 20):
            for c in range(b+1, 35):
                U = [0, a, b, c]
                s, _ = _try_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # n=5: try some structured sets
    # Quadratic
    for n in range(4, 12):
        U = [i*i for i in range(n)]
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Powers of 2
    for n in range(3, 15):
        U = [1 << i for i in range(n)]
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Arithmetic progressions
    for n in range(3, 15):
        for d in range(1, 10):
            U = [i*d for i in range(n)]
            s, _ = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Mixed: quadratic with small step
    for n in range(4, 10):
        U = [i*i for i in range(n)]
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try random sets in small ranges
    random.seed(42)
    for trial in range(2000):
        if time.time() - _start_time > TIME_LIMIT:
            break
        n = random.randint(3, 10)
        max_val = random.randint(n, 50)
        U = set([0])
        for _ in range(n-1):
            U.add(random.randint(1, max_val))
        U = sorted(U)
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try some specific known good sets
    specific_sets = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],  # squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],  # powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127, 255, 511],  # 2^k - 1
        [0, 1, 3, 4, 9, 10, 12, 13, 16, 17],  # mixed
        [0, 1, 4, 10, 20, 35, 56, 84, 120, 165],  # triangular numbers
    ]
    for U in specific_sets:
        s, _ = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Local optimization: try adding/removing elements
    if time.time() - _start_time < TIME_LIMIT:
        current = list(best_set)
        improved = True
        while improved and time.time() - _start_time < TIME_LIMIT:
            improved = False
            # Try adding a new element
            max_val = max(current) + 1
            for new_val in range(1, max_val + 5):
                if new_val not in current:
                    candidate = sorted(current + [new_val])
                    s, _ = _try_set(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = list(candidate)
                        improved = True
            # Try removing an element (keep 0)
            for i in range(1, len(current)):
                candidate = current[:i] + current[i+1:]
                if len(candidate) >= 2:
                    s, _ = _try_set(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = list(candidate)
                        improved = True
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
