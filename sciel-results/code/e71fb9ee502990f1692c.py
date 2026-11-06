import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N, d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Function to calculate score (number of monochromatic 4-APs)
    def get_score(col):
        # col is a numpy array of 0s and 1s
        # For each AP, check if all 4 elements have the same color
        # c0 == c1 == c2 == c3
        # This is equivalent to: (c0 == c1) & (c1 == c2) & (c2 == c3)
        # Or more efficiently using XOR:
        # If all are same, then c0^c1=0, c1^c2=0, c2^c3=0
        # So (c0^c1) | (c1^c2) | (c2^c3) == 0
        c0 = col[aps[:, 0]]
        c1 = col[aps[:, 1]]
        c2 = col[aps[:, 2]]
        c3 = col[aps[:, 3]]
        
        # Monochromatic if all equal
        # Using integer arithmetic: if all equal, sum of differences is 0? No.
        # Simplest vectorized check:
        # mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        # return np.sum(mask)
        
        # Alternative: 
        # If c0==c1==c2==c3, then c0+c1+c2+c3 is either 0 or 4.
        # But mixed colors give 1, 2, or 3.
        # So we can check if (c0==c1) and (c1==c2) and (c2==c3).
        # Let's stick to the boolean mask for clarity and speed.
        mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return np.sum(mask)

    # Initialize coloring
    # Start with a random coloring or a simple pattern
    # Random is often better for SA to escape local minima early
    col = np.random.randint(0, 2, size=N, dtype=np.int8)
    current_score = get_score(col)
    
    best_col = col.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute indices for faster access? 
    # The bottleneck is get_score. 
    # We can optimize get_score by pre-computing the AP indices once (done).
    
    # To speed up, we can try to update score incrementally, 
    # but for N=256, num_aps is ~ 256*85/3 ~ 7000? 
    # Let's estimate num_aps.
    # d=1: 253, d=2: 250, ... d=85: 1
    # Sum_{d=1}^{85} (256 - 3d) = 85*256 - 3*(85*86/2) = 21760 - 3*3655 = 21760 - 10965 = 10795.
    # 10,795 APs.
    # Vectorized check on 10k elements is very fast in NumPy (< 1ms).
    # So we can afford many iterations.
    
    rng = np.random.default_rng()
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with higher T if stuck? Or just stop.
            # Let's restart with a new random coloring if we've cooled down
            T = 50.0
            col = np.random.randint(0, 2, size=N, dtype=np.int8)
            current_score = get_score(col)
            if current_score < best_score:
                best_score = current_score
                best_col = col.copy()
            continue
            
        # Pick a random index to flip
        idx = rng.integers(0, N)
        old_val = col[idx]
        col[idx] = 1 - old_val
        
        new_score = get_score(col)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_col = col.copy()
        else:
            # Reject
            col[idx] = old_val
            
        T *= cooling_rate
        
    # Return best found
    return [int(c) for c in best_col]
# EVOLVE-BLOCK-END
