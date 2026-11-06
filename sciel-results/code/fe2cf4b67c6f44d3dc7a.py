"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random


def _isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_maximal(chosen):
    """Check if the chosen set is maximal: every unchosen point forms an isosceles with some pair in chosen."""
    chosen_set = set(chosen)
    all_points = set(itertools.product(range(N), repeat=2))
    unchosen = all_points - chosen_set
    if not unchosen:
        return True
    for p in unchosen:
        blocked = False
        for a, b in itertools.combinations(chosen, 2):
            if _isosceles(p, a, b):
                blocked = True
                break
        if not blocked:
            return False
    return True


def _greedy_construction(order):
    """Construct a maximal no-isosceles set using the given order of points."""
    chosen = []
    for p in order:
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return chosen


def construct():
    best_set = None
    best_size = float('inf')
    
    # Run multiple randomized greedy constructions
    num_restarts = 200
    all_points = list(itertools.product(range(N), repeat=2))
    
    for _ in range(num_restarts):
        random.shuffle(all_points)
        candidate = _greedy_construction(all_points)
        
        # The greedy construction always produces a maximal set?
        # Yes, because we add a point if it doesn't conflict. 
        # At the end, any point not in the set must have conflicted with some pair in the set 
        # at the time it was considered. 
        # Wait, if a point was considered early and rejected, it might not conflict with the final set?
        # No, the set only grows. If it conflicted with a subset, it conflicts with the superset.
        # So yes, the result of this greedy is always maximal.
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
            
            # Optional: try local search on this best set? 
            # Since removal is impossible, we can't improve size locally easily.
            # But we can try to find a smaller one via more restarts.

    return best_set
# EVOLVE-BLOCK-END
