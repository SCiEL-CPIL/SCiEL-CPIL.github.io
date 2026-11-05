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
    if n == 0:
        return True
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
    
    # We want to find the maximum k such that a k-subset is Sidon.
    # We iterate k from n down to 1.
    # To speed up, we can use a heuristic to prune.
    
    # Precompute all pair sums for the full set? No, that's for the subset.
    
    def backtrack(start, chosen, sums_set):
        nonlocal best
        # Pruning: if even taking all remaining elements doesn't beat best, return
        if len(chosen) + (n - start) <= best:
            return
        
        if len(chosen) > best:
            best = len(chosen)
            # If we found a valid Sidon set of size > best, update best
            # But we need to check if it's Sidon.
            # The recursion ensures we only add elements that don't conflict.
            # So any 'chosen' in the recursion path is a valid Sidon set.
            
        # If we can't add more, return
        if start == n:
            return
            
        # Option 1: Skip s[start]
        backtrack(start + 1, chosen, sums_set)
        
        # Option 2: Take s[start]
        x = s[start]
        conflict = False
        new_sums = []
        # Check against existing chosen elements
        for y in chosen:
            sm = x + y
            if sm in sums_set:
                conflict = True
                break
            new_sums.append(sm)
        
        if not conflict:
            # Check x + x
            if x + x in sums_set:
                conflict = True
            
        if not conflict:
            new_sums_set = sums_set | set(new_sums) | {x + x}
            chosen.append(x)
            backtrack(start + 1, chosen, new_sums_set)
            chosen.pop()

    backtrack(0, [], set())
    return best

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
    
    # Strategy: Partition 12 elements into small APs.
    # APs of length 3 are very dense.
    # APs of length 4 are also dense.
    # Mix of APs.
    
    # 1. Four APs of length 3
    # e.g., {0,1,2}, {10,11,12}, {20,21,22}, {30,31,32}
    for gap in range(5, 30):
        for offset in range(0, 5):
            c = []
            for i in range(4):
                start = offset + i * gap
                c.extend([start, start + 1, start + 2])
            if max(c) <= 128:
                add_candidate(c)
                
    # 2. Three APs of length 4
    for gap in range(5, 30):
        for offset in range(0, 5):
            c = []
            for i in range(3):
                start = offset + i * gap
                c.extend([start, start + 1, start + 2, start + 3])
            if max(c) <= 128:
                add_candidate(c)
                
    # 3. Two APs of length 6
    for gap in range(5, 30):
        for offset in range(0, 5):
            c = []
            for i in range(2):
                start = offset + i * gap
                c.extend([start + j for j in range(6)])
            if max(c) <= 128:
                add_candidate(c)
                
    # 4. One AP of length 12 (control)
    for start in range(0, 10):
        for diff in range(1, 10):
            c = [start + i * diff for i in range(12)]
            if max(c) <= 128:
                add_candidate(c)
                
    # 5. Random perturbations of APs of length 3
    rng = random.Random(123)
    for _ in range(100):
        base_gaps = [10, 10, 10]
        c = []
        for i in range(4):
            start = sum(base_gaps[:i]) + rng.randint(-2, 2)
            c.extend([start + j + rng.randint(-1, 1) for j in range(3)])
        if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            add_candidate(c)
            
    # 6. "Collision-heavy" random sets
    # Pick numbers close to each other
    for _ in range(100):
        c = set()
        while len(c) < 12:
            # Pick a cluster center
            center = rng.randint(0, 120)
            # Pick a few numbers around it
            for _ in range(3):
                x = center + rng.randint(-2, 2)
                if 0 <= x <= 128:
                    c.add(x)
        if len(c) == 12:
            add_candidate(list(c))

    return candidates

def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin
    
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
                if evaluated > 50:
                    break
    
    if best_cand is None:
        # Fallback: a simple AP
        return list(range(12))
    
    return best_cand
# EVOLVE-BLOCK-END
