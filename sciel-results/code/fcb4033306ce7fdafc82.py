import numpy as np
import time

N = 100

def max_overlap_linear(h):
    # h is array of length N
    # S_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    # We want max_k S_k
    max_val = 0
    for k in range(N):
        # Vectorized calculation for shift k
        # i goes from 0 to N-1-k
        # h[i] and h[i+k]
        prod = h[:N-k] * (1 - h[k:])
        s_k = np.sum(prod)
        if s_k > max_val:
            max_val = s_k
    return max_val

def construct():
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_score = max_overlap_linear(best_h)
    
    # Try to improve using a simple parameterized family
    # h_i = 0.5 + A * cos(2*pi*i/N)
    # Check constraint h_i in [0,1] => A <= 0.5
    
    # We will do a small grid search over A
    # Since the function is smooth, a coarse grid should find the optimum if it exists
    # Note: If circular shifts are used, the optimum is A=0.
    # If linear shifts are used, A>0 might help reduce S_0 at the cost of other S_k.
    
    # Deadline guard
    start_time = time.time()
    deadline = 100 # 100 seconds
    
    # Grid search for A
    # A in [0, 0.5]
    # Also try a phase shift? cos(2*pi*i/N + phi)
    # Sum of cos is 0, so sum h_i = N/2 is maintained.
    
    best_A = 0.0
    best_phi = 0.0
    
    # Coarse grid
    for A in np.linspace(0, 0.5, 11):
        for phi in np.linspace(0, 2*np.pi, 10):
            if time.time() > deadline:
                break
            h = 0.5 + A * np.cos(2*np.pi*np.arange(N)/N + phi)
            # Clip to [0,1] just in case, though A<=0.5 ensures it
            h = np.clip(h, 0, 1)
            # Ensure sum is exactly N/2?
            # Sum of cos is 0, so sum is N/2.
            # Floating point errors might make it slightly off.
            # The problem says "sum is exactly len/2".
            # We should normalize or adjust.
            # If we clip, sum changes.
            # Better to use a family that naturally sums to N/2 and stays in [0,1].
            # A * cos is safe for A <= 0.5.
            
            score = max_overlap_linear(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                best_A = A
                best_phi = phi

    # If no improvement found, return seed
    return best_h
