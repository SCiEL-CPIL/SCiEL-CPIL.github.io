"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

N = 100

def construct():
    # We restrict step heights to the discrete set {0, 0.5, 1}.
    # The constraint is sum(h) = N/2.
    # We initialize with the constant sequence h_i = 0.5, which satisfies the sum constraint.
    # For this sequence, the overlap O_k = sum h_i (1-h_{i+k}) = sum 0.5 * 0.5 = N/4.
    # Since the average overlap over all shifts is N/4, the maximum overlap cannot be less than N/4.
    # Thus, the constant 0.5 sequence is a theoretical optimum for the circular overlap metric.
    
    h = np.full(N, 0.5)
    
    def get_max_overlap(h):
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    current_max_ov = get_max_overlap(h)
    
    # Perform a local search to verify optimality.
    # Moves involve swapping two 0.5s for a 0 and a 1 to explore the discrete space.
    for _ in range(100):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        
        # Only consider moves that keep values in {0, 0.5, 1} and sum constant.
        # Current values are 0.5. Moving to 0 and 1 keeps sum constant.
        old_i, old_j = h[i], h[j]
        
        # Apply move
        h[i] = 0.0
        h[j] = 1.0
        
        new_max_ov = get_max_overlap(h)
        
        if new_max_ov < current_max_ov:
            current_max_ov = new_max_ov
        else:
            # Revert
            h[i] = old_i
            h[j] = old_j
            
    return h

# EVOLVE-BLOCK-END
