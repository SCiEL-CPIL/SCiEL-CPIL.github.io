"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import Counter

def _is_sidon_fast(subset):
    """Check if subset has all pair sums (i<=j) unique using numpy."""
    s = np.array(sorted(subset))
    n = len(s)
    # Compute all pair sums i<=j
    i_idx, j_idx = np.triu_indices(n)
    sums = s[i_idx] + s[j_idx]
    # Check for duplicates
    return len(np.unique(sums)) == len(sums)

def _max_sidon_size_fast(s):
    """Find max Sidon subset size using branch-and-bound."""
    s = sorted(s)
    n = len(s)
    
    # Quick check: is the whole set Sidon?
    if _is_sidon_fast(s):
        return n
    
    # Branch and bound
    best = 1
    # Try from largest k down
    for k in range(n - 1, 1, -1):
        # Use a greedy + backtrack approach
        # First try: can we find ANY k-subset that is Sidon?
        found = _exists_sidon_subset(s, k)
        if found:
            best = k
            return best
    return best

def _exists_sidon_subset(s, k):
    """Check if there exists a k-subset of s that is Sidon (branch and bound)."""
    n = len(s)
    if k <= 1:
        return True
    if k > n:
        return False
    
    # Precompute all pair sums for the full set to help pruning
    # Use recursive backtracking with pruning
    
    def backtrack(start, chosen, sums_set):
        nonlocal found
        if found:
            return
        if len(chosen) == k:
            found = True
            return
        # Pruning: not enough elements left
        if n - start < k - len(chosen):
            return
        # Try each element from start
        for i in range(start, n):
            x = s[i]
            # Check if adding x creates duplicate sums
            new_sums = []
            conflict = False
            for y in chosen:
                sm = x + y
                if sm in sums_set:
                    conflict = True
                    break
                new_sums.append(sm)
            if not conflict:
                # Also check x+x
                if x + x in sums_set:
                    conflict = True
            if not conflict:
                new_sums_set = sums_set | set(new_sums) | {x + x}
                chosen.append(x)
                backtrack(i + 1, chosen, new_sums_set)
                chosen.pop()
                if found:
                    return
    
    found = False
    backtrack(0, [], set())
    return found

def _try_candidate(cands):
    """Return (valid, max_size) for a candidate list of 12 distinct ints in [0,128]."""
    if len(cands) != 12:
        return False, 99
    if len(set(cands)) != 12:
        return False, 99
    if any(x < 0 or x > 128 for x in cands):
        return False, 99
    return True, _max_sidon_size_fast(cands)

def _build_candidates():
    """Generate candidate ambient sets designed to have small max Sidon subset."""
    candidates = []
    seen = set()
    
    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            candidates.append(c)
    
    # Base patterns designed to create many sum collisions
    bases = [
        # Arithmetic progression (max collisions)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Two clusters
        [0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37],
        # Three clusters
        [0, 1, 2, 16, 17, 18, 32, 33, 34, 48, 49, 50],
        # Near-arithmetic with perturbation
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
        # Quadratic (has structure)
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
        # Geometric + small
        [0, 1, 2, 4, 8, 16, 3, 5, 6, 10, 12, 20],
        # Mixed small values
        [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        # Dense low range
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        # Symmetric pairs
        [0, 1, 2, 3, 64, 63, 62, 61, 128, 127, 126, 125],
        # Modulo structure
        [0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121],
        # Square roots pattern
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 110],
        # Twin primes + small
        [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 25],
    ]
    
    # Apply affine transforms mod 129
    for base in bases:
        for a in range(1, 129):
            for b in range(0, 129, 1):
                c = [(x * a + b) % 129 for x in base]
                add_candidate(c)
    
    # Random candidates
    import random
    rng = random.Random(12345)
    for _ in range(5000):
        c = rng.sample(range(129), 12)
        add_candidate(c)
    
    # Targeted: pairs that sum to same value
    # e.g., {a, b, c, d} where a+b = c+d creates collision
    for _ in range(2000):
        c = []
        # Create 6 pairs with shared sums
        for _ in range(6):
            s_val = rng.randint(0, 256)
            a = rng.randint(0, 128)
            b = s_val - a
            if 0 <= b <= 128 and b != a:
                c.extend([a, b])
        # Fill remaining with random
        while len(c) < 12:
            x = rng.randint(0, 128)
            if x not in c:
                c.append(x)
        c = c[:12]
        if len(set(c)) == 12:
            add_candidate(c)
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 75  # 75s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
    import random
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    best_size = 99
    best_cand = None
    evaluated = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        valid, size = _try_candidate(cand)
        evaluated += 1
        if valid and size < best_size:
            best_size = size
            best_cand = cand
            if best_size <= 4:
                # Very good, hard to beat
                break
            if best_size <= 5:
                # Keep searching a bit more
                if evaluated > 5000:
                    break
    
    if best_cand is None:
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
