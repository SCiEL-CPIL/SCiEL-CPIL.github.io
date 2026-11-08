import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_x = None
    best_val = float('inf')
    
    # Helper to compute initial R
    def compute_R(x):
        # Linear autocorrelation
        # x is 1D array of length N
        # R[t] = sum_{i=0}^{N-1-t} x[i] * x[i+t]
        # We only need t=1 to N-1 for the objective, but let's keep 0..N-1
        R = np.zeros(N)
        for t in range(N):
            R[t] = np.dot(x[:N-t], x[t:])
        return R

    # Helper to update R when flipping x[k]
    # x is the current state BEFORE flip
    # k is the index to flip
    # Returns the change in R
    def get_delta_R(x, k):
        delta = np.zeros(N)
        # Flipping x[k] from v to -v
        # Term x[i]x[i+t] is affected if i=k or i+t=k
        # 1. i=k: term x[k]x[k+t]. Change: -2*x[k]*x[k+t]
        # 2. i+t=k => i=k-t: term x[k-t]x[k]. Change: -2*x[k-t]*x[k]
        
        # Note: R[t] = sum x[i]x[i+t]
        # Contribution of x[k] to R[t]:
        # If k is in [0, N-1-t]:
        #   If i=k, term is x[k]x[k+t]. Valid if k+t < N.
        #   If i+t=k, term is x[k-t]x[k]. Valid if k-t >= 0.
        
        for t in range(N):
            change = 0
            # Case 1: i=k
            if k + t < N:
                change -= 2 * x[k] * x[k+t]
            # Case 2: i+t=k => i=k-t
            if k - t >= 0:
                change -= 2 * x[k-t] * x[k]
            delta[t] = change
            
        return delta

    # Objective: max abs R[1:]
    def get_max_sidelobe(R):
        if len(R) <= 1:
            return 0.0
        return np.max(np.abs(R[1:]))

    # Local Search
    def local_search(x, max_steps=10000):
        R = compute_R(x)
        current_max = get_max_sidelobe(R)
        
        for step in range(max_steps):
            # Check time
            if time.time() > deadline:
                return x, current_max, R
            
            # Pick random index
            k = np.random.randint(N)
            
            # Calculate change
            delta = get_delta_R(x, k)
            
            # Apply
            R_new = R + delta
            new_max = get_max_sidelobe(R_new)
            
            if new_max < current_max:
                # Accept
                x[k] *= -1
                R = R_new
                current_max = new_max
            else:
                # Reject (R remains same)
                pass
                
        return x, current_max, R

    # Main Loop: Multiple Restarts
    while time.time() < deadline:
        # Random initialization
        x = np.random.choice([-1, 1], N)
        
        # Run local search
        x, val, R = local_search(x, max_steps=20000)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # If we found a very good one, we might stop?
        # Theoretical min for random is ~ sqrt(N log N) / N?
        # For N=400, sqrt(400*6)/400 ~ 0.12.
        # If we get below 0.1, it's good.
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
