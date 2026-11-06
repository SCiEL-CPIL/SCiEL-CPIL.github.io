"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(point, chosen):
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            if _is_isosceles(point, a, b):
                return False
    return True


def _is_maximal(chosen):
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True


def _greedy_construct(order):
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _try_remove(chosen, p):
    """Try to remove point p and see if we can still maintain maximality."""
    remaining = [q for q in chosen if q != p]
    # Check if the set without p is still no-isosceles (it should be, since it's a subset)
    # Check if it's still maximal
    if _is_maximal(remaining):
        return remaining
    return None


def _local_search(chosen, max_iterations=1000):
    """Try to remove points to minimize set size while maintaining maximality."""
    current = list(chosen)
    improved = True
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        for i, p in enumerate(current):
            new_set = _try_remove(current, p)
            if new_set is not None:
                current = new_set
                improved = True
                break
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300 second deadline with safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Try multiple random orderings
    all_points = list(itertools.product(range(N), repeat=2))
    
    # First, try a deterministic greedy to get a baseline
    baseline = _greedy_construct(all_points)
    if _is_maximal(baseline):
        baseline = _local_search(baseline, max_iterations=500)
        if len(baseline) < best_size:
            best_size = len(baseline)
            best_set = baseline
    
    # Check time
    if time.time() > deadline - 60:
        return best_set if best_set else baseline
    
    # Try randomized constructions
    for trial in range(1000):
        if time.time() > deadline - 30:
            break
            
        # Create a random ordering
        random_order = all_points.copy()
        random.shuffle(random_order)
        
        candidate = _greedy_construct(random_order)
        
        # Ensure maximality
        if not _is_maximal(candidate):
            # Try to make it maximal by adding points
            candidate_set = set(candidate)
            for p in all_points:
                if p not in candidate_set and _can_add(p, candidate):
                    candidate.append(p)
                    candidate_set.add(p)
        
        # Local search to minimize
        candidate = _local_search(candidate, max_iterations=200)
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate
    
    return best_set if best_set else baseline
# EVOLVE-BLOCK-END
