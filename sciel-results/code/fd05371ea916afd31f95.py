"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    Strategy: 
    1. The problem is likely about linear cross-correlation (based on Inspiration 2's score).
    2. Uniform h=0.5 gives max linear correlation ~24.75.
    3. We aim for a value around 19.
    4. Binary sequences with structured patterns (long runs) minimize local correlation.
    5. Use Simulated Annealing on binary sequences to minimize the peak correlation.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    def compute_max_corr(h):
        """
        Compute the maximum linear cross-correlation between h and 1-h.
        h is a binary array (0 or 1).
        """
        g = 1.0 - h
        # np.correlate computes linear correlation
        # mode='full' gives length 2N-1
        # We want shifts k = -(N-1) to N-1, excluding 0
        c = np.correlate(h, g, mode='full')
        # c[N-1] is k=0
        # Exclude k=0
        # The max will be among the non-zero shifts
        # Due to symmetry c[k] = c[-k] for real signals? 
        # corr(h,g,k) = sum h[i] g[i+k]. corr(g,h,k) = sum g[i] h[i+k] = corr(h,g,-k).
        # So we just need to look at half.
        max_val = 0.0
        # We can just take the max of the whole array excluding the center
        # But center is max? Not necessarily.
        # Let's just take max of the whole thing, but ignore the center index N-1.
        # Actually, for non-negative h, g, the correlation at k=0 is sum h_i g_i.
        # This is usually the largest magnitude.
        # But we must EXCLUDE k=0.
        
        # Efficiently find max excluding center
        # c has length 2N-1. Center is N-1.
        # We can split
        left = c[:N-1]
        right = c[N:]
        
        if len(left) > 0:
            max_val = max(max_val, np.max(left))
        if len(right) > 0:
            max_val = max(max_val, np.max(right))
            
        return max_val

    def init_binary():
        """
        Initialize a random binary sequence with sum N/2.
        """
        h = np.zeros(N, dtype=int)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1
        return h

    def swap_indices(h):
        """
        Pick a random 1 and a random 0, and return their indices.
        """
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            return None
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        return i, j

    # Simulated Annealing
    best_h = init_binary()
    best_score = compute_max_corr(best_h)
    
    current_h = best_h.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Number of iterations based on time
    # Estimate cost of one iteration: compute_max_corr is O(N^2) ~ 10^4 ops.
    # 140s / 10^4 ~ 14000 iterations? 
    # NumPy correlate is fast. Let's aim for 100,000 iterations.
    
    while time.time() < deadline:
        # Batch processing to reduce Python overhead
        # We'll run a batch of 1000 swaps
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            idxs = swap_indices(current_h)
            if idxs is None:
                continue
            i, j = idxs
            
            # Swap
            current_h[i], current_h[j] = current_h[j], current_h[i]
            
            new_score = compute_max_corr(current_h)
            
            # Acceptance
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
            else:
                # Reject, swap back
                current_h[i], current_h[j] = current_h[j], current_h[i]
        
        T *= cooling_rate
        if T < T_min:
            # Reheat? Or just stop?
            # Let's try a few restarts if time permits
            break

    # Local Search Polish
    # Try all possible swaps? O(N^2) swaps. N=100 -> 2500 swaps.
    # Each swap costs O(N^2) to evaluate. Total O(N^4) ~ 10^8. Too slow in Python.
    # But we can be smarter.
    
    # Let's do a simple local search: try swapping each 1 with every 0? No, too slow.
    # Instead, we can try to improve the worst correlations.
    
    # Return the best binary sequence found
    # Convert to float for return
    return best_h.astype(float)

# EVOLVE-BLOCK-END
