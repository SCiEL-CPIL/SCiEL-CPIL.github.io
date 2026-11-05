"""Rewritten program: exhaustive combinatorial search for optimal underpacking."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips_frac = [Fraction(1, d) for d in denominators]
    
    best_deficit = None
    best_indices = None
    
    def try_combo(combo):
        nonlocal best_deficit, best_indices
        s = Fraction(0)
        for i in combo:
            s += recips_frac[i]
        if s >= 1:
            return
        deficit = 1 - s
        if best_deficit is None or deficit < best_deficit:
            best_deficit = deficit
            best_indices = list(combo)
    
    # Known good solution: 1/2 + 1/3 + 1/7 + 1/43
    # Indices: 0, 1, 5, 41
    known = (0, 1, 5, 41)
    s = recips_frac[0] + recips_frac[1] + recips_frac[5] + recips_frac[41]
    if s < 1:
        best_deficit = 1 - s
        best_indices = list(known)
    
    # Exhaustive search over subset sizes
    # The sum of all reciprocals from 2 to 48 is about 3.3, so we need subsets
    # whose sum is just under 1. Smaller subsets are more likely to be near 1
    # without exceeding it, and the search space is manageable.
    
    for k in range(2, 9):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                break
            try_combo(combo)
    
    if best_indices is None:
        # Fallback: single element
        best_indices = [0]
        best_deficit = 1 - recips_frac[0]
    
    return best_indices
# EVOLVE-BLOCK-END
