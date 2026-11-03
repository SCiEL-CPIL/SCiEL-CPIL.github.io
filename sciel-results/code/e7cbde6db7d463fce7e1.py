"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    # Allow 110 seconds for computation to be safe
    deadline = start_time + 110.0
    
    # The theoretical lower bound for the maximum overlap of a binary sequence
    # with sum N/2 is N/4.
    # For N=100, this is 25.
    # The seed score corresponds to value 50.
    # So the target score is 50/25 = 2.0.
    
    # Initialize with a random binary sequence
    h = np.zeros(N)
    indices = np.random.choice(N, N//2, replace=False)
    h[indices] = 1.0
    
    def get_max_overlap(h):
        """
        Calculates the maximum overlap of h with shifted 1-h.
        For binary h, overlap_k = count(i) where h[i]=1 and h[(i+k)%N]=0.
        """
        max_ov = 0
        # We can optimize this, but N=100 is small
        for k in range(N):
            shifted = np.roll(h, -k)
            # We want min(h[i], 1-shifted[i])
            # If h[i]=1, 1-shifted[i]=1 => shifted[i]=0. Min(1,1)=1.
            # If h[i]=0, 1-shifted[i]=1 => shifted[i]=0. Min(0,1)=0.
            # If h[i]=1, 1-shifted[i]=0 => shifted[i]=1. Min(1,0)=0.
            # If h[i]=0, 1-shifted[i]=0 => shifted[i]=1. Min(0,0)=0.
            # So we just count where h[i]==1 and shifted[i]==0
            ov = np.sum((h == 1) & (shifted == 0))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    current_ov = get_max_overlap(h)
    best_score = current_ov
    best_h = h.copy()
    
    # If we already hit the theoretical limit, return
    if best_score <= N // 4:
        return best_h

    # Local Search: Random Swaps
    # We maintain lists of indices of 1s and 0s for efficiency
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    while time.time() < deadline:
        # Pick a random 1 and a random 0 to swap
        idx_1 = ones[np.random.randint(len(ones))]
        idx_0 = zeros[np.random.randint(len(zeros))]
        
        if idx_1 == idx_0:
            continue
        
        # Perform swap
        h[idx_1] = 0
        h[idx_0] = 1
        
        new_ov = get_max_overlap(h)
        
        if new_ov < best_score:
            best_score = new_ov
            best_h = h.copy()
            
            # Update indices lists
            # idx_1 became 0, idx_0 became 1
            # We need to update the arrays 'ones' and 'zeros'
            # This is slightly expensive if done naively, but N is small.
            # Alternatively, just recompute them if we accept a new best.
            # But we need them for the next iteration regardless.
            # Actually, we can just keep h and recompute ones/zeros if needed, 
            # or update in place.
            
            # Efficient update:
            # Find position of idx_1 in ones, replace with idx_0? No, idx_0 was in zeros.
            # It's safer to just recompute if we accept, or use a set.
            # Given N=100, recomputing is fine.
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            
            if best_score <= N // 4:
                return best_h
        else:
            # Revert swap
            h[idx_1] = 1
            h[idx_0] = 0
            
    return best_h
# EVOLVE-BLOCK-END
