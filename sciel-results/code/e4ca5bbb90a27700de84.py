"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _covers(point, chosen):
    """Check if adding 'point' to 'chosen' would create an isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return True
    return False

def _is_covered_by(point, chosen):
    """Check if point is covered by chosen (i.e., forms isosceles with some pair in chosen)."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return True
    return False

def _maximal(chosen):
    """Check if chosen is maximal: every point not in chosen is covered by chosen."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if not _is_covered_by(p, chosen):
                return False
    return True

def construct():
    deadline = time.time() + 280  # Safety margin
    
    # Strategy: Use a greedy set cover approach
    # 1. First, find a valid no-isosceles set (greedy)
    # 2. Then try to minimize it while maintaining maximality
    
    # Start with a greedy maximal set
    chosen = []
    all_points = list(itertools.product(range(N), repeat=2))
    
    for p in all_points:
        if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    
    # Now try to remove points while maintaining no-isosceles and maximality
    # A point can be removed if:
    # 1. Removing it doesn't create a new isosceles (it won't, since we're removing)
    # 2. The point itself is still "covered" by the remaining set OR it was covering others
    # Actually, we need to check maximality after removal
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing chosen[idx]
            new_chosen = chosen[:idx] + chosen[idx+1:]
            if not new_chosen:
                continue
            # Check no-isosceles (should be fine since we're removing)
            # Check maximality
            chosen_set = set(new_chosen)
            is_maximal = True
            for p in all_points:
                if p not in chosen_set:
                    if not _is_covered_by(p, new_chosen):
                        is_maximal = False
                        break
            if is_maximal:
                chosen = new_chosen
                improved = True
                break
    
    return chosen
# EVOLVE-BLOCK-END
