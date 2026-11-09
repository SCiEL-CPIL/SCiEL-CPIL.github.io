"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _compute_threshold_bitmask(a_map, b_map, n):
    """Compute the exact reset word length using bitmask-based BFS on subset automaton.
    Returns (threshold, found) where found is True if synchronizing."""
    # Precompute transition tables for bitmasks
    # For each bitmask (subset of states), compute image under a and b
    # There are 2^n possible subsets, but we only visit reachable ones
    
    # Precompute single-state transitions
    a_single = a_map  # a_single[i] = state i goes to
    b_single = b_map
    
    # Precompute transition for all possible bitmasks
    # trans_a[mask] = bitmask of images under a
    # trans_b[mask] = bitmask of images under b
    size = 1 << n
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        ta = 0
        tb = 0
        for i in range(n):
            if mask & (1 << i):
                ta |= (1 << a_single[i])
                tb |= (1 << b_single[i])
        trans_a[mask] = ta
        trans_b[mask] = tb
    
    full_mask = (1 << n) - 1
    
    # BFS from full_mask, looking for singleton (power of 2)
    # dist[mask] = shortest word length to reach mask from full_mask
    # We want the shortest word that reaches ANY singleton
    
    dist = {}
    dist[full_mask] = 0
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        d = dist[mask]
        
        # Check if this is a singleton
        if mask & (mask - 1) == 0:
            return d, True
        
        # Try both transitions
        for trans in (trans_a, trans_b):
            next_mask = trans[mask]
            if next_mask not in dist:
                dist[next_mask] = d + 1
                queue.append(next_mask)
    
    return -1, False

def _pawlik_automaton(n):
    """Standard Pawlik/Cerny automaton with threshold (n-1)^2."""
    a_map = [(i + 1) % n for i in range(n)]
    b_map = [0 if i == 0 else i - 1 for i in range(n)]
    return [a_map, b_map]

def _local_search(initial_a, initial_b, n, max_iters=5000, time_limit=50):
    """Local search to find automata with high reset threshold."""
    best_a = list(initial_a)
    best_b = list(initial_b)
    best_thresh, _ = _compute_threshold_bitmask(best_a, best_b, n)
    
    start_time = time.time()
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random position and random letter to mutate
        pos = rng.randint(0, n - 1)
        letter = rng.randint(0, 1)
        old_val = best_a[pos] if letter == 0 else best_b[pos]
        new_val = rng.randint(0, n - 1)
        
        if new_val == old_val:
            continue
        
        if letter == 0:
            best_a[pos] = new_val
            a_map = best_a
            b_map = best_b
        else:
            best_b[pos] = new_val
            a_map = best_a
            b_map = best_b
        
        thresh, sync = _compute_threshold_bitmask(a_map, b_map, n)
        
        if not sync or thresh <= best_thresh:
            # Revert
            if letter == 0:
                best_a[pos] = old_val
            else:
                best_b[pos] = old_val
        else:
            best_thresh = thresh
    
    return best_a, best_b, best_thresh

def construct():
    # Start with Pawlik automaton (known threshold 121 for N=12)
    a_map, b_map = _pawlik_automaton(N)
    
    # Verify baseline
    thresh, sync = _compute_threshold_bitmask(a_map, b_map, N)
    if not sync:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Local search to try to exceed 121
    deadline = 55.0  # Safety margin before 60s evaluator kill
    start = time.time()
    
    best_a, best_b, best_thresh = a_map, b_map, thresh
    
    # Multiple restarts with different seeds
    for seed in [42, 123, 999, 777, 555]:
        if time.time() - start > deadline:
            break
        
        rng = random.Random(seed)
        # Start from Pawlik and do local search
        init_a = list(a_map)
        init_b = list(b_map)
        
        # Add some random perturbations to escape local optima
        for _ in range(rng.randint(1, 5)):
            pos = rng.randint(0, N - 1)
            letter = rng.randint(0, 1)
            if letter == 0:
                init_a[pos] = rng.randint(0, N - 1)
            else:
                init_b[pos] = rng.randint(0, N - 1)
        
        ta, sync_a = _compute_threshold_bitmask(init_a, init_b, N)
        if not sync_a:
            continue
        
        # Local search from this perturbed starting point
        la, lb, lt = _local_search(init_a, init_b, N, max_iters=2000, 
                                     time_limit=min(10, deadline - (time.time() - start)))
        
        if lt > best_thresh:
            best_a, best_b, best_thresh = la, lb, lt
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
