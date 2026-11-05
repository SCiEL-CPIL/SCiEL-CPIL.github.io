"""Rewritten program: low-discrepancy signs on all arithmetic progressions up to n=256."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_abs_ap_sum(signs):
    """Compute max absolute partial sum over all APs using numpy."""
    n = len(signs)
    s = signs.astype(np.int32)
    max_abs = 1
    for d in range(1, n):
        # For each residue class
        for r in range(d):
            if r >= n:
                break
            # Get indices r, r+d, r+2d, ...
            idx = np.arange(r, n, d)
            if len(idx) < 2:
                continue
            # Cumulative sum
            csum = np.cumsum(s[idx])
            a = np.max(np.abs(csum))
            if a > max_abs:
                max_abs = a
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    rng = random.Random(123)
    
    # Start with a random assignment
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    best_signs = signs.copy()
    best_score = _max_abs_ap_sum(best_signs)
    
    # Local search with random restarts
    while time.time() < deadline:
        improved = False
        # Try each flip
        for i in range(n):
            if time.time() > deadline:
                break
            signs[i] = -signs[i]
            score = _max_abs_ap_sum(signs)
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = -signs[i]
        
        if not improved:
            # Perturb and continue
            for _ in range(15):
                signs[rng.randint(0, n-1)] *= -1
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
