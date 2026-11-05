"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import comb

def _max_sidon_size(subset):
    """Compute the largest subset of 'subset' with all pair sums (i<=j) distinct."""
    n = len(subset)
    if n <= 1:
        return n
    elems = sorted(subset)
    best = [0]
    
    def backtrack(idx, current, current_sums):
        if idx >= n:
            best[0] = max(best[0], len(current))
            return
        if len(current) + (n - idx) <= best[0]:
            return
        # Try not taking idx
        backtrack(idx + 1, current, current_sums)
        # Try taking idx
        elem = elems[idx]
        new_sums = set(current_sums)
        valid = True
        for j in range(len(current)):
            val = current[j] + elem
            if val in new_sums:
                valid = False
                break
            new_sums.add(val)
        if valid:
            val = elem + elem
            if val in new_sums:
                valid = False
            else:
                new_sums.add(val)
        if valid:
            backtrack(idx + 1, current + [elem], new_sums)
    
    backtrack(0, [], set())
    return best[0]

def _eval_candidate(cand):
    """Return the max Sidon subset size of the candidate."""
    return _max_sidon_size(cand)

def _valid(cand):
    """Check candidate is 12 distinct integers in [0,128]."""
    if len(cand) != 12:
        return False
    if len(set(cand)) != 12:
        return False
    return all(0 <= x <= 128 for x in cand)

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin within 60s limit
    
    best_cand = None
    best_score = 99
    seen = set()
    
    def consider(cand):
        nonlocal best_cand, best_score
        if not _valid(cand):
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = list(cand)
    
    # Strategy: Use 6 pairs {a*i, a*i + b} where a and b are chosen to maximize collisions
    # Key insight: if a is large and b is small, cross-pair sums like a*i + a*j = a*(i+j)
    # will collide with other combinations
    
    # Try various block structures
    for a in range(8, 25):  # stride between pairs
        for b in range(1, min(a, 10)):  # offset within pair
            cand = []
            for i in range(6):
                x1 = a * i
                x2 = a * i + b
                if x1 <= 128 and x2 <= 128:
                    cand.extend([x1, x2])
            if len(cand) == 12 and len(set(cand)) == 12:
                consider(cand)
        if time.time() > deadline:
            break
    
    # Try two-AP unions: {o1 + s1*i} + {o2 + s2*i} for i in 0..5
    for s1 in range(4, 20):
        for s2 in range(4, 20):
            for o1 in range(0, 8):
                for o2 in range(0, 8):
                    a = [o1 + s1 * i for i in range(6)]
                    b = [o2 + s2 * i for i in range(6)]
                    cand = sorted(set(a + b))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        consider(cand)
        if time.time() > deadline:
            break
    
    # Try grid structures: rows * cols = 12
    for rows in (2, 3, 4, 6):
        for cols in (2, 3, 4, 6):
            if rows * cols != 12:
                continue
            for rs in range(5, 25):
                for cs in range(2, 12):
                    cand = sorted(set([rs * r + cs * c for r in range(rows) for c in range(cols)]))
                    if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
                        consider(cand)
        if time.time() > deadline:
            break
    
    if best_cand is None:
        best_cand = list(range(12))
        best_score = _eval_candidate(best_cand)
    
    # Local search: perturb the best candidate
    rng = random.Random(42)
    for iteration in range(3000):
        if time.time() > deadline:
            break
        cand = list(best_cand)
        # Randomly perturb 1-3 elements
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, 11)
            cand[idx] = rng.randint(0, 128)
        if not _valid(cand):
            continue
        key = tuple(sorted(cand))
        if key in seen:
            continue
        seen.add(key)
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = cand
    
    return best_cand
# EVOLVE-BLOCK-END
