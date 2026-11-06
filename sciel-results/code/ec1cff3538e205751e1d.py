"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_sunflower(a, b, c):
    """Check if three 4-element sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(s, family):
    """Check if s can be added to family without creating a sunflower."""
    for i in range(len(family)):
        a = family[i]
        for j in range(i+1, len(family)):
            b = family[j]
            if _is_sunflower(a, b, s):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # All possible 4-element subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n = len(all_sets)
    
    # Greedy initial construction
    family = []
    used = [False] * n
    for idx in range(n):
        if time.time() > deadline - 10.0:
            break
        s = all_sets[idx]
        if _can_add(s, family):
            family.append(s)
            used[idx] = True
    
    # Local search with swap moves
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try removing one set and adding a different one
        for i in range(len(family)):
            if time.time() > deadline - 5.0:
                break
            removed = family[i]
            family_without = family[:i] + family[i+1:]
            # Find best replacement
            best_add = None
            best_count = -1
            for j in range(n):
                if j == i:  # skip the removed set's index
                    continue
                if used[j] and j != i:
                    continue
                s = all_sets[j]
                if _can_add(s, family_without):
                    # Count how many additional sets we could potentially add
                    # For now, just try to add it
                    best_add = j
                    break  # Just take the first valid one for speed
            if best_add is not None:
                family[i] = all_sets[best_add]
                improved = True
                break  # Restart the outer loop after a swap
    
    # Final polish: try to add more sets
    for idx in range(n):
        if time.time() > deadline - 2.0:
            break
        if not used[idx]:
            s = all_sets[idx]
            if _can_add(s, family):
                family.append(s)
                used[idx] = True
    
    return [sorted(s) for s in family]
# EVOLVE-BLOCK-END
