"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def _is_subset_sum(target, elements):
    """Check if target is a subset sum of elements (list of Fractions)."""
    if target == 0:
        return True
    if not elements:
        return False
    sums = {Fraction(0)}
    for e in elements:
        new_sums = {s + e for s in sums}
        sums = sums | new_sums
        if target in sums:
            return True
    return target in sums

def _check_valid(S):
    """Check if S satisfies the reciprocal-subsum-free condition."""
    for a in S:
        others = [Fraction(1, b) for b in S if b != a]
        target = Fraction(1, a)
        if _is_subset_sum(target, others):
            return False
    return True

def _greedy_remove(S, order):
    """Remove violators from S in the given order."""
    S = list(S)
    for a in order:
        if a not in S:
            continue
        others = [Fraction(1, b) for b in S if b != a]
        target = Fraction(1, a)
        if _is_subset_sum(target, others):
            S.remove(a)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Try multiple greedy orders
    candidates = []
    
    # Order: increasing
    S1 = _greedy_remove(range(1, 25), sorted(range(1, 25)))
    candidates.append(S1)
    
    # Order: decreasing
    S2 = _greedy_remove(range(1, 25), sorted(range(1, 25), reverse=True))
    candidates.append(S2)
    
    # Order: even first, then odd
    S3 = _greedy_remove(range(1, 25), list(range(2, 25, 2)) + list(range(1, 25, 2)))
    candidates.append(S3)
    
    # Order: odd first, then even
    S4 = _greedy_remove(range(1, 25), list(range(1, 25, 2)) + list(range(2, 25, 2)))
    candidates.append(S4)
    
    # Pick the largest valid candidate
    best = None
    for c in candidates:
        if _check_valid(c):
            if best is None or len(c) > len(best):
                best = c
    
    if best is None:
        best = list(range(13, 25))  # Fallback
    
    # Try to add back elements
    removed = [i for i in range(1, 25) if i not in best]
    for a in removed:
        if time.time() > deadline:
            break
        candidate = best + [a]
        if _check_valid(candidate):
            best = candidate
    
    # Repeat add-back until no more can be added
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        removed = [i for i in range(1, 25) if i not in best]
        for a in removed:
            if time.time() > deadline - 2:
                break
            candidate = best + [a]
            if _check_valid(candidate):
                best = candidate
                changed = True
                break
    
    return best
# EVOLVE-BLOCK-END
