"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

N = 100

def construct():
    # Restrict step heights to {0, 0.5, 1} with sum = N/2.
    # This means we need exactly N/2 units of "0.5-mass".
    # If we have x positions at 1, y positions at 0.5, z positions at 0:
    # x + y + z = N
    # x*1 + y*0.5 + z*0 = N/2
    # => 2x + y = N
    # => y = N - 2x
    # So for any x in [0, N/2], y = N - 2x, z = N - x - y = N - x - (N - 2x) = x
    # So we have x ones, (N-2x) halfs, and x zeros.
    
    def get_max_overlap(h):
        max_ov = 0.0
        for k in range(1, N):
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    best_h = None
    best_score = float('inf')
    
    # Try different numbers of 1s (and correspondingly 0s)
    # x = number of 1s, also number of 0s
    # y = N - 2x = number of 0.5s
    for x in range(0, N // 2 + 1):
        # Create initial configuration: x ones, N-2x halfs, x zeros
        h = np.zeros(N)
        # Place ones and zeros first, fill rest with 0.5
        indices = list(range(N))
        random.shuffle(indices)
        for i in range(x):
            h[indices[i]] = 1.0
        for i in range(x, 2 * x):
            h[indices[i]] = 0.0
        for i in range(2 * x, N):
            h[indices[i]] = 0.5
        
        current_max_ov = get_max_overlap(h)
        
        # Local search: swap pairs to maintain sum constraint
        # Valid moves:
        # 1. Swap (1, 0) -> (0, 1): no change in sum
        # 2. Swap (1, 0.5) -> (0.5, 1): no change in sum
        # 3. Swap (0, 0.5) -> (0.5, 0): no change in sum
        # 4. More complex: (1, 1, 0, 0) -> (0.5, 0.5, 0.5, 0.5) changes composition
        #    But simpler: just swap two elements if their sum is preserved
        # Actually, any swap of two elements preserves the sum automatically.
        # But we need to ensure values stay in {0, 0.5, 1}.
        
        # Let's do a simple local search with random swaps
        for iteration in range(200):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            old_i, old_j = h[i], h[j]
            
            # Swap
            h[i] = old_j
            h[j] = old_i
            
            new_max_ov = get_max_overlap(h)
            
            if new_max_ov < current_max_ov:
                current_max_ov = new_max_ov
            else:
                # Revert
                h[i] = old_i
                h[j] = old_j
        
        if current_max_ov < best_score:
            best_score = current_max_ov
            best_h = h.copy()
    
    # If no valid configuration found (shouldn't happen), fall back to constant 0.5
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    return best_h

# EVOLVE-BLOCK-END
