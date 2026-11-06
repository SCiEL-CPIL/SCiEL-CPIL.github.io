"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _build_ap_matrix(n=256):
    """
    Build a dense binary incidence matrix where each row corresponds to an
    arithmetic progression within [1, n] (1-indexed) and each column to a position.
    Entry (i, j) is 1 if position j+1 is in AP i.
    """
    aps = []
    for start in range(1, n + 1):
        for step in range(1, n - start + 1):
            # Generate all elements of the AP starting at 'start' with given 'step'
            k = start
            while k <= n:
                if len(aps) == 0:
                    current_ap = [k]
                else:
                    # We just append to the list of lists
                    pass
                k += step
            # Actually, simpler: just generate the list
            ap_list = list(range(start, n + 1, step))
            if len(ap_list) >= 1:
                aps.append(ap_list)
    
    m = len(aps)
    mat = np.zeros((m, n), dtype=np.int8)
    for i, ap in enumerate(aps):
        for pos in ap:
            mat[i, pos - 1] = 1
    return mat

def _evaluate(signs, ap_mat):
    """
    Calculate the maximum absolute sum over all arithmetic progressions.
    signs: vector of length n with values -1 or 1.
    ap_mat: matrix of shape (m, n).
    """
    sums = ap_mat @ signs
    return np.max(np.abs(sums))

def construct():
    n = 256
    # 1. Build the AP matrix (cached or built once). 
    # Building takes a few seconds. We can build it inside construct but it might be slow if called repeatedly?
    # The prompt says "Each evaluation must finish within 60 s". 
    # Building the matrix for n=256:
    # Number of APs is roughly sum_{step=1}^{255} (256 - step) ~ 256^2 / 2 = 32768.
    # Matrix size 32768 x 256.
    # Construction time: Python loop over ~32k APs, each filling a few entries. Should be < 1-2 seconds.
    # Let's check if we can cache it. Since construct() is the entry point, we can use a global variable or default arg.
    
    global _AP_MAT_CACHE
    if '_AP_MAT_CACHE' not in globals():
        _AP_MAT_CACHE = _build_ap_matrix(n)
    
    ap_mat = _AP_MAT_CACHE
    m, _ = ap_mat.shape
    
    # 2. Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Initial solution: random signs
    current_signs = rng.choice([-1, 1], size=n)
    current_cost = _evaluate(current_signs, ap_mat)
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.1
    alpha = 0.995  # Cooling rate
    iterations = 50000 # Number of steps
    
    start_time = time.time()
    max_time = 50.0 # Leave 10s buffer
    
    T = T_start
    step = 0
    while step < iterations and (time.time() - start_time) < max_time:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        # Calculate delta cost efficiently?
        # Flipping sign at idx changes the sum of every AP containing idx by 2 * old_sign[idx].
        # Specifically, if we flip s[idx] from +1 to -1, the sum of AP i decreases by 2 if idx in AP i, else 0.
        # This is hard to update the max abs sum incrementally without tracking all sums.
        # Given m ~ 32k and n=256, a full matrix-vector product is 32k*256 ~ 8M ops.
        # In numpy, this is very fast (< 1ms).
        # So we can just recalculate the full cost after each flip.
        
        old_sign = current_signs[idx]
        current_signs[idx] = -old_sign
        
        new_cost = _evaluate(current_signs, ap_mat)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
        else:
            # Metropolis criterion
            delta = new_cost - current_cost
            if delta == 0:
                # Accept to move around plateaus? Or reject? Usually accept to escape.
                pass
            else:
                if rng.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    # Reject: revert
                    current_signs[idx] = old_sign
        
        T *= alpha
        step += 1
    
    # Convert numpy array to list of Python integers
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
