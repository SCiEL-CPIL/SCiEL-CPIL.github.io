"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_SEED = 42


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_valid_addition(p, chosen):
    """Check if point p can be added to chosen without creating an isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _isosceles(p, chosen[i], chosen[j]):
                return False
    return True


def _greedy_construct(order):
    """Construct a maximal no-isosceles set by greedily adding points in the given order."""
    chosen = []
    for p in order:
        if _is_valid_addition(p, chosen):
            chosen.append(p)
    return chosen


def _random_order():
    """Generate a random ordering of all grid points."""
    points = list(itertools.product(range(N), repeat=2))
    random.shuffle(points)
    return points


def _hill_climb(start_set, max_iters=5000, timeout=30):
    """
    Try to improve a maximal no-isosceles set by:
    1. Removing a point and re-adding points that were previously excluded
    2. Trying different removal/re-addition sequences
    Returns the best (smallest) set found.
    """
    best_set = list(start_set)
    best_len = len(best_set)
    start_time = time.time()
    
    # Precompute all points
    all_points = list(itertools.product(range(N), repeat=2))
    point_set = set(start_set)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        current_set = list(best_set)
        current_len = len(current_set)
        
        # Try removing one point and see if we can re-add fewer points to maintain maximality
        # Actually, for a maximal set, removing a point may allow us to NOT add it back,
        # but we need to check if the remaining set is still maximal.
        # A better approach: try to swap - remove a point, then try to add back previously excluded points
        # to fill in, but with a different configuration that might allow fewer total points.
        
        # Alternative: local search by trying to remove each point and checking if the set is still maximal
        # If removing a point makes it non-maximal, we need to add some point back.
        # If removing a point keeps it maximal, we have a smaller maximal set!
        
        removed_any = False
        for idx in range(len(current_set)):
            if time.time() - start_time > timeout:
                break
            
            candidate = current_set[idx]
            # Try removing this point
            reduced = current_set[:idx] + current_set[idx+1:]
            
            # Check if reduced is still maximal (no point can be added)
            is_maximal = True
            for p in all_points:
                if p in reduced:
                    continue
                if _is_valid_addition(p, reduced):
                    is_maximal = False
                    break
            
            if is_maximal:
                # We found a smaller maximal set!
                best_set = reduced
                best_len = len(reduced)
                current_set = reduced
                removed_any = True
                break
        
        if not removed_any:
            # Try a different strategy: remove two points and see if we can still have a maximal set
            # This is more expensive but might find improvements
            found_improvement = False
            for i in range(len(current_set)):
                if time.time() - start_time > timeout:
                    break
                for j in range(i + 1, len(current_set)):
                    if time.time() - start_time > timeout:
                        break
                    candidate_i = current_set[i]
                    candidate_j = current_set[j]
                    reduced = [x for k, x in enumerate(current_set) if k != i and k != j]
                    
                    # Check if reduced is still maximal
                    is_maximal = True
                    for p in all_points:
                        if p in reduced:
                            continue
                        if _is_valid_addition(p, reduced):
                            is_maximal = False
                            break
                    
                    if is_maximal:
                        best_set = reduced
                        best_len = len(reduced)
                        current_set = reduced
                        found_improvement = True
                        break
                if found_improvement:
                    break
        
        if not found_improvement and not removed_any:
            # Try a more aggressive move: remove a random point and greedily re-add
            # This might find a different local optimum
            if random.random() < 0.3:
                idx = random.randint(0, len(current_set) - 1)
                removed_point = current_set[idx]
                reduced = current_set[:idx] + current_set[idx+1:]
                
                # Now try to greedily add back points in random order to restore maximality
                # But only add if it helps (i.e., we end up with a set no larger than before)
                excluded = [p for p in all_points if p not in reduced]
                random.shuffle(excluded)
                
                new_set = list(reduced)
                for p in excluded:
                    if _is_valid_addition(p, new_set):
                        new_set.append(p)
                    if len(new_set) >= current_len:
                        break
                
                if len(new_set) < best_len:
                    best_set = new_set
                    best_len = len(new_set)
    
    return best_set


def construct():
    """
    Construct a minimal maximal no-isosceles set using:
    1. Multiple greedy constructions with different orderings
    2. Hill climbing to improve the best found set
    """
    start_time = time.time()
    deadline = start_time + 270  # Leave 30s margin before 300s limit
    
    random.seed(_SEED)
    
    # Phase 1: Try multiple greedy constructions with different orderings
    best_set = None
    best_len = float('inf')
    
    # First, try row-major order (baseline)
    row_major = list(itertools.product(range(N), repeat=2))
    set_rm = _greedy_construct(row_major)
    best_set = set_rm
    best_len = len(set_rm)
    
    # Try several random orderings
    num_random_tries = 50
    for i in range(num_random_tries):
        if time.time() > deadline - 120:  # Leave time for hill climbing
            break
        
        order = _random_order()
        s = _greedy_construct(order)
        if len(s) < best_len:
            best_set = s
            best_len = len(s)
    
    # Phase 2: Hill climb from the best set found
    if time.time() < deadline - 60:
        improved = _hill_climb(best_set, max_iters=10000, timeout=120)
        if len(improved) < best_len:
            best_set = improved
            best_len = len(best_set)
    
    # Phase 3: Try more aggressive local search if time permits
    if time.time() < deadline - 30:
        # Try removing points one at a time more systematically
        current = list(best_set)
        all_points = list(itertools.product(range(N), repeat=2))
        
        while time.time() < deadline - 10:
            progress = False
            for idx in range(len(current)):
                if time.time() > deadline - 10:
                    break
                candidate = current[idx]
                reduced = current[:idx] + current[idx+1:]
                
                # Check maximality
                is_maximal = True
                for p in all_points:
                    if p in reduced:
                        continue
                    if _is_valid_addition(p, reduced):
                        is_maximal = False
                        break
                
                if is_maximal:
                    current = reduced
                    best_set = current
                    best_len = len(current)
                    progress = True
                    break
            
            if not progress:
                break
    
    return best_set
# EVOLVE-BLOCK-END
