"""Rewritten program: greedy maximal clique in lcm-compatibility graph."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    n = 512
    # Numbers 1..512
    nums = np.arange(1, n + 1, dtype=np.int64)
    
    # Build compatibility matrix: comp[i,j] = True iff lcm(nums[i], nums[j]) <= 512
    # lcm(a,b) = a*b//gcd(a,b)
    # Vectorized: compute gcd matrix, then lcm
    
    # For memory efficiency with 512x512, this is fine
    # Compute pairwise gcd using numpy
    # We'll do it in chunks or use a smart approach
    
    # Actually 512x512 = 262144 entries, very manageable
    a = nums[:, np.newaxis].astype(np.int64)
    b = nums[np.newaxis, :].astype(np.int64)
    
    # Compute gcd matrix - numpy doesn't have vectorized gcd, so use a loop or scipy
    # Let's use scipy
    from scipy.spatial.distance import cdist  # not useful
    # Just compute gcd in a loop per row, or use a trick
    # Actually, let's just do it with a nested approach but vectorized per row
    
    # Better: compute gcd for each pair
    # Since 512 is small, let's just do it efficiently
    comp = np.zeros((n, n), dtype=bool)
    for i in range(n):
        ai = nums[i]
        # gcd(ai, nums) for all nums
        # We can compute this using the property that gcd(a,b) = a if b|a, etc.
        # Just use a loop for gcd computation per row - 512*512 = 262k gcd calls, fast
        gc = np.array([gcd(ai, int(bj)) for bj in nums], dtype=np.int64)
        lcm = (ai * nums) // gc
        comp[i, :] = (lcm <= 512)
    
    # Now find a maximal clique greedily
    # Try multiple strategies:
    best_clique = set()
    
    # Strategy 1: Greedy by degree (highest degree first)
    degrees = comp.sum(axis=1)
    order = np.argsort(-degrees)
    
    def greedy_clique(order, comp, deadline):
        clique = []
        remaining = set(range(len(comp)))
        for idx in order:
            if time.time() > deadline:
                break
            if idx not in remaining:
                continue
            # Check if idx is compatible with all in clique
            if all(comp[idx, c] for c in clique):
                clique.append(idx)
                remaining.discard(idx)
        return clique
    
    clique1 = greedy_clique(order, comp, deadline)
    if len(clique1) > len(best_clique):
        best_clique = set(clique1)
    
    # Strategy 2: Greedy by smallest number first
    order2 = np.arange(n)
    clique2 = greedy_clique(order2, comp, deadline)
    if len(clique2) > len(best_clique):
        best_clique = set(clique2)
    
    # Strategy 3: Greedy by largest number first
    order3 = np.arange(n-1, -1, -1)
    clique3 = greedy_clique(order3, comp, deadline)
    if len(clique3) > len(best_clique):
        best_clique = set(clique3)
    
    # Strategy 4: Randomized restarts
    rng = np.random.default_rng(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Random ordering weighted by degree
        w = degrees.astype(float) + 1.0
        w /= w.sum()
        order_r = rng.choice(n, size=n, replace=False, p=w)
        clique_r = greedy_clique(order_r, comp, deadline)
        if len(clique_r) > len(best_clique):
            best_clique = set(clique_r)
    
    # Local improvement: try to add elements not in clique
    # For each element not in clique, check if it can be added
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(n):
            if time.time() > deadline:
                break
            if i in best_clique:
                continue
            if all(comp[i, c] for c in best_clique):
                best_clique.add(i)
                changed = True
    
    # Convert to actual numbers
    result = sorted(nums[list(best_clique)])
    
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
