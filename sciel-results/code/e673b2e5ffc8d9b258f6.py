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
    # Check if adding 'point' to 'chosen' creates an isosceles triangle
    # with any two existing points in 'chosen'
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen, all_points):
    chosen_set = set(chosen)
    # For every point not in chosen, check if it can be added
    # If it can be added, the set is not maximal
    for p in all_points:
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

def _make_maximal(chosen, all_points):
    """Add points to 'chosen' until it is maximal."""
    chosen_set = set(chosen)
    # We need to add points that can be added.
    # Order of addition might matter for the final set size, but any maximal set is a local optimum.
    # We try to add points that "block" the most other points?
    # For now, just add in the order they appear in all_points or random.
    # To be deterministic-ish, we iterate all_points.
    changed = True
    while changed:
        changed = False
        for p in all_points:
            if p not in chosen_set:
                if _can_add(p, chosen):
                    chosen.append(p)
                    chosen_set.add(p)
                    changed = True
    return chosen

def _local_search(chosen, all_points, max_iterations=500, time_limit=None, start_time=None):
    """
    Local search to minimize set size.
    Moves:
    1. Remove a point if the set remains maximal.
    2. Swap a point with another if the set remains valid and maximal (size preserved, structure changed).
    """
    current = list(chosen)
    current_size = len(current)
    best_size = current_size
    best_set = current.copy()
    
    if time_limit is None:
        time_limit = 200
    if start_time is None:
        start_time = time.time()

    # Precompute all points for fast iteration
    all_points_list = list(all_points)
    
    # Helper to check validity and maximality efficiently?
    # _can_add is O(K^2). _is_maximal is O(N^2 * K^2).
    # With K ~ 20, N=256.
    # _is_maximal: 256 * 400 = 102k ops.
    # This is fast enough for hundreds of iterations per second.
    
    stall_count = 0
    max_stall = 50
    
    while max_iterations > 0 and (time.time() - start_time) < time_limit:
        max_iterations -= 1
        
        improved = False
        
        # 1. Try Removals
        # Shuffle current points to avoid bias
        removal_order = current.copy()
        random.shuffle(removal_order)
        
        for r in removal_order:
            new_set = [p for p in current if p != r]
            if _is_maximal(new_set, all_points_list):
                # Accept removal
                current = new_set
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_set = current.copy()
                improved = True
                stall_count = 0
                break # Restart loop with new set
        
        if improved:
            continue
            
        # 2. If no removal worked, try Swaps
        # Remove r, add a
        # We want to find a swap that keeps the set maximal.
        # This is harder. We can try random swaps.
        
        # Pick a random point to remove
        r = random.choice(current)
        base_set = [p for p in current if p != r]
        
        # Find candidates 'a' that can be added to base_set
        # And check if base_set + a is maximal
        candidates = []
        for a in all_points_list:
            if a in current:
                continue
            if _can_add(a, base_set):
                # Check if adding a makes it maximal
                # Note: base_set is a subset of a maximal set, so it's not maximal.
                # We need to check if base_set + a is maximal.
                new_set = base_set + [a]
                if _is_maximal(new_set, all_points_list):
                    candidates.append(a)
        
        if candidates:
            a = random.choice(candidates)
            current = base_set + [a]
            # Size is unchanged, but structure is.
            # This doesn't update best_size, but might allow future removals.
            stall_count += 1
            improved = True # Consider it a move
        else:
            # No valid swap found with single replacement.
            # Try 2-swap? Or just random restart?
            # For now, increment stall.
            stall_count += 1
            
        if stall_count > max_stall:
            # Random restart or break
            # Let's try a random restart of the local search phase
            # by shuffling the current set? No, that might break validity.
            # Just break out and let the outer loop do a new greedy.
            break
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300 second deadline with safety margin
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # 1. Deterministic Greedy Baseline
    baseline = _greedy_construct(all_points)
    baseline = _make_maximal(baseline, all_points)
    baseline = _local_search(baseline, all_points, max_iterations=200, time_limit=10, start_time=start_time)
    if len(baseline) < best_size:
        best_size = len(baseline)
        best_set = baseline.copy()
        
    # 2. Randomized Greedy with Local Search
    # Run multiple trials
    trials = 0
    while time.time() < deadline - 10:
        trials += 1
        # Create a random ordering
        random_order = all_points.copy()
        random.shuffle(random_order)
        
        candidate = _greedy_construct(random_order)
        candidate = _make_maximal(candidate, all_points)
        
        # Local search
        # Allocate remaining time minus a buffer for the next iteration
        remaining_time = (deadline - 10) - time.time()
        if remaining_time < 5:
            break
            
        # Limit local search time per trial to ensure we can try many trials
        # If we have lots of time, spend more on local search.
        # If we have little time, do quick local search.
        local_time = min(remaining_time * 0.5, 30) # Cap at 30s per local search
        if local_time < 1:
            local_time = 1
            
        candidate = _local_search(candidate, all_points, max_iterations=1000, time_limit=local_time, start_time=time.time())
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_set = candidate.copy()
            # Reset stall counter implicitly by starting fresh in next loop?
            # We don't have a global stall counter, but we can monitor improvement.
            
    return best_set if best_set else baseline
# EVOLVE-BLOCK-END
