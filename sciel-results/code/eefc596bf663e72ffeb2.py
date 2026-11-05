"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import Counter
import random

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
    """Generate a small, high-quality set of candidates with high additive structure."""
    candidates = []
    seen = set()
    
    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            candidates.append(c)
    
    # 1. Arithmetic Progressions (known to have small Sidon subsets)
    for start in range(0, 10):
        for diff in range(1, 15):
            c = [start + i * diff for i in range(12)]
            if max(c) <= 128:
                add_candidate(c)
    
    # 2. Two-cluster APs
    for start1 in range(0, 5):
        for diff1 in range(1, 5):
            for start2 in range(50, 100):
                for diff2 in range(1, 5):
                    c1 = [start1 + i * diff1 for i in range(6)]
                    c2 = [start2 + i * diff2 for i in range(6)]
                    c = c1 + c2
                    if len(set(c)) == 12 and max(c) <= 128:
                        add_candidate(c)
    
    # 3. Random perturbations of APs
    rng = random.Random(123)
    for _ in range(200):
        base = [i * 10 for i in range(12)] # Large gaps to allow perturbation
        c = []
        for i, b in enumerate(base):
            p = rng.randint(-5, 5)
            c.append(b + p)
        # Ensure distinct and in range
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 4. Specific "collision-heavy" constructions
    # e.g. pairs summing to same value
    for _ in range(100):
        c = set()
        while len(c) < 12:
            # Pick a sum S and split it
            S = rng.randint(0, 256)
            a = rng.randint(0, 128)
            b = S - a
            if 0 <= b <= 128:
                c.add(a)
                c.add(b)
        if len(c) == 12:
            add_candidate(list(c))

    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin (evaluator kills at 90s)
    
    # Generate candidates
    candidates = _build_candidates()
    
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
            if best_size <= 3:
                # Theoretical minimum for N=12 is likely 3 or 4. 
                # If we hit 3, it's excellent.
                break
            if best_size <= 4:
                # Keep searching a bit more for 3
                if evaluated > 100:
                    break
    
    if best_cand is None:
        # Fallback: a simple AP
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
