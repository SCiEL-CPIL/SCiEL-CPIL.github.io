"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _compute_threshold_bitmask(a_map, b_map, n):
    """Compute the exact reset word length using bitmask-based BFS on subset automaton.
    Returns (threshold, found) where found is True if synchronizing."""
    # Precompute transition for all possible bitmasks
    size = 1 << n
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        ta = 0
        tb = 0
        for i in range(n):
            if mask & (1 << i):
                ta |= (1 << a_map[i])
                tb |= (1 << b_map[i])
        trans_a[mask] = ta
        trans_b[mask] = tb
    
    full_mask = (1 << n) - 1
    
    # BFS from full_mask, looking for singleton (power of 2)
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
        next_a = trans_a[mask]
        if next_a not in dist:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        next_b = trans_b[mask]
        if next_b not in dist:
            dist[next_b] = d + 1
            queue.append(next_b)
    
    return -1, False

def _pawlik_automaton(n):
    """Standard Pawlik/Cerny automaton with threshold (n-1)^2."""
    a_map = [(i + 1) % n for i in range(n)]
    b_map = [0 if i == 0 else i - 1 for i in range(n)]
    return [a_map, b_map]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    # Start with the known optimal Pawlik automaton
    best_a, best_b = _pawlik_automaton(N)
    best_length, _ = _compute_threshold_bitmask(best_a, best_b, N)
    
    # If baseline is not valid (shouldn't happen), return it anyway
    if best_length < 0:
        return [best_a, best_b]
    
    # Local search to try to exceed the Pawlik threshold
    # The Pawlik automaton is known to be optimal for binary automata up to certain sizes,
    # but we perform a quick local search to be sure and potentially find something better.
    
    rng = random.Random(42)
    
    # We'll do a few rounds of local search starting from Pawlik
    # and also from random perturbations of Pawlik.
    
    for seed in [42, 123, 999, 777, 555, 111, 222, 333, 444, 556]:
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Start from Pawlik with some random perturbations
        init_a = list(best_a)
        init_b = list(best_b)
        
        # Add 1-3 random mutations
        for _ in range(rng.randint(1, 3)):
            pos = rng.randint(0, N - 1)
            letter = rng.randint(0, 1)
            if letter == 0:
                init_a[pos] = rng.randint(0, N - 1)
            else:
                init_b[pos] = rng.randint(0, N - 1)
        
        # Evaluate the perturbed automaton
        thresh, sync = _compute_threshold_bitmask(init_a, init_b, N)
        
        if not sync:
            continue
        
        # If this is already better, update best
        if thresh > best_length:
            best_length = thresh
            best_a = init_a
            best_b = init_b
        
        # Now do local search from this point
        cur_a = list(init_a)
        cur_b = list(init_b)
        cur_thresh = thresh
        
        for it in range(200):
            if time.time() > deadline:
                return [best_a, best_b]
            
            # Pick a random position and letter to mutate
            pos = rng.randint(0, N - 1)
            letter = rng.randint(0, 1)
            old_val = cur_a[pos] if letter == 0 else cur_b[pos]
            new_val = rng.randint(0, N - 1)
            
            if new_val == old_val:
                continue
            
            if letter == 0:
                cur_a[pos] = new_val
            else:
                cur_b[pos] = new_val
            
            thresh, sync = _compute_threshold_bitmask(cur_a, cur_b, N)
            
            if not sync or thresh <= cur_thresh:
                # Revert
                if letter == 0:
                    cur_a[pos] = old_val
                else:
                    cur_b[pos] = old_val
            else:
                cur_thresh = thresh
                # Update global best
                if cur_thresh > best_length:
                    best_length = cur_thresh
                    best_a = list(cur_a)
                    best_b = list(cur_b)
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
