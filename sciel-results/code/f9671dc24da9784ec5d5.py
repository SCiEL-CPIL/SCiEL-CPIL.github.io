# EVOLVE-BLOCK-START
import numpy as np
import time
import random

M = 205

def _compute_forbidden():
    forbidden = set()
    for x in range(1, M):
        val = pow(x, 2, M)
        if val != 0:
            forbidden.add(val)
    return forbidden

FORBIDDEN = _compute_forbidden()

# Precompute conflict masks for each vertex
# conflict_mask[i] is a bitset of all j such that i and j cannot coexist
CONFLICT_MASKS = [0] * M
for i in range(M):
    mask = 0
    for j in range(M):
        if i != j and (i - j) % M in FORBIDDEN:
            mask |= (1 << j)
    CONFLICT_MASKS[i] = mask

def _can_add(cand, sel_bits):
    """Check if cand can be added to sel_bits"""
    return (sel_bits & CONFLICT_MASKS[cand]) == 0

def _popcount(x):
    return bin(x).count('1')

def _greedy_init():
    """Generate a good initial solution using greedy with random ordering"""
    sel = 0
    remaining = list(range(M))
    random.shuffle(remaining)
    for v in remaining:
        if _can_add(v, sel):
            sel |= (1 << v)
    return sel

def _simulated_annealing(start_bits, deadline):
    """
    Run Simulated Annealing starting from start_bits until deadline.
    Returns the best solution found.
    """
    current = start_bits
    current_size = _popcount(current)
    best = current
    best_size = current_size
    
    # Initial temperature
    T = 1.0
    # Cooling rate
    alpha = 0.9999
    # We will adjust alpha to ensure we use the time well
    # Estimate steps: ~10^7 steps in 600s is too slow in Python.
    # Let's aim for ~10^5 to 10^6 steps per second? 
    # Bit operations are fast. Let's benchmark mentally.
    # 1 step: pick 2 randoms, check conflict, update bits, popcount (maybe cache size).
    # Popcount is expensive. We should track size incrementally.
    
    # Reset size tracking
    cur_size = current_size
    
    # Precompute list of vertices for random selection
    vertices = list(range(M))
    
    # Main loop
    while time.time() < deadline:
        # Pick a random move: Swap u (in S) and v (not in S)
        # Or Add/Remove
        # Swap is more effective for maintaining size.
        
        # To pick u in S and v not in S efficiently:
        # We can just pick two random distinct vertices u, v.
        # If u in S and v not in S -> Swap
        # If u not in S and v in S -> Swap
        # If both in S or both not in S -> skip or try add/remove
        
        u = random.choice(vertices)
        v = random.choice(vertices)
        
        if u == v:
            continue
            
        u_in = (current >> u) & 1
        v_in = (current >> v) & 1
        
        if u_in == v_in:
            # Try add or remove
            if u_in:
                # Remove u
                new_state = current & ~(1 << u)
                new_size = cur_size - 1
                delta_e = -1 # We want to maximize size, so removing is bad (delta E > 0 in minimization, or -1 in maximization)
                # Energy = -size. Delta E = -new_size - (-cur_size) = cur_size - new_size = 1.
                # Accept if rand < exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    current = new_state
                    cur_size = new_size
            else:
                # Add u
                if _can_add(u, current):
                    new_state = current | (1 << u)
                    new_size = cur_size + 1
                    # Delta E = -1. Always accept.
                    current = new_state
                    cur_size = new_size
                    if cur_size > best_size:
                        best = current
                        best_size = cur_size
        else:
            # Swap: u in S, v not in S (or vice versa)
            # Let u be the one in S, v the one not in S
            if u_in:
                out_v = u
                in_v = v
            else:
                out_v = v
                in_v = u
            
            # Check if in_v can be added after removing out_v
            # New set = (S \ {out_v}) U {in_v}
            # Conflict check: in_v must not conflict with (S \ {out_v})
            # This is equivalent to: (current & ~(1<<out_v)) & CONFLICT_MASKS[in_v] == 0
            temp_state = current & ~(1 << out_v)
            if (temp_state & CONFLICT_MASKS[in_v]) == 0:
                # Valid swap
                # Size change: 0 (remove 1, add 1)
                # So delta E = 0. Always accept?
                # If we accept all zero-delta moves, we might cycle.
                # But it explores the space.
                # Actually, we can just accept it.
                current = temp_state | (1 << in_v)
                # cur_size remains same
                # We don't update best_size here as it didn't change
            else:
                # Invalid swap, do nothing
                pass

        # Cooling
        T *= alpha
        if T < 1e-6:
            T = 1e-6 # Prevent underflow

    return best, best_size

def furstenberg_sarkozy_205() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0
    
    random.seed(42)
    
    best_global = 0
    best_global_size = 0
    
    # Phase 1: Multiple Greedy Starts
    # Run several greedy constructions to find a good starting point
    for i in range(500):
        if time.time() > deadline - 100:
            break
        sel = _greedy_init()
        cnt = _popcount(sel)
        if cnt > best_global_size:
            best_global = sel
            best_global_size = cnt
            
    # Phase 2: Simulated Annealing from the best greedy start
    # We can also run SA from other good starts if time permits
    # Let's run SA on the best one found so far
    if time.time() < deadline - 50:
        best_sa, best_sa_size = _simulated_annealing(best_global, deadline - 50)
        if best_sa_size > best_global_size:
            best_global = best_sa
            best_global_size = best_sa_size

    # Phase 3: If time remains, try SA from a few other random greedy starts
    # This helps if the first SA got stuck in a local optimum far from the global one
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Split remaining time among a few runs
        num_runs = 5
        time_per_run = (remaining_time - 10) / num_runs
        for i in range(num_runs):
            if time.time() > deadline - 10:
                break
            sel = _greedy_init()
            cnt = _popcount(sel)
            # Only run SA if this start is reasonably good
            if cnt >= best_global_size - 5: 
                end_time = time.time() + time_per_run
                best_sa, best_sa_size = _simulated_annealing(sel, end_time)
                if best_sa_size > best_global_size:
                    best_global = best_sa
                    best_global_size = best_sa_size

    result = [i for i in range(M) if (best_global >> i) & 1]
    return np.array(sorted(result), dtype=np.int64)

# EVOLVE-BLOCK-END
