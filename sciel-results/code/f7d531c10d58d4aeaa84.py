import numpy as np

N = 100

def construct():
    # Start with seed
    h = np.full(N, 0.5)
    
    # Initial energy
    def get_value(h):
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr)
        
    current_h = h.copy()
    current_val = get_value(current_h)
    
    # Annealing parameters
    T = 1.0
    T_min = 1e-5
    alpha = 0.99
    steps = 5000
    
    for _ in range(steps):
        # Pick a random index
        i = np.random.randint(N)
        # Propose a change: random walk or swap
        # Since sum must be constant, we should swap or move mass
        # Move mass from i to j
        j = np.random.randint(N)
        if i == j: continue
        
        # Move delta mass
        delta = np.random.uniform(-0.1, 0.1)
        
        # Ensure bounds
        new_hi = current_h[i] + delta
        new_hj = current_h[j] - delta
        
        if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
            continue
            
        # Apply change
        old_hi = current_h[i]
        old_hj = current_h[j]
        current_h[i] = new_hi
        current_h[j] = new_hj
        
        new_val = get_value(current_h)
        
        # Accept if better or with probability exp(-dE/T)
        d_val = new_val - current_val
        if d_val < 0:
            current_val = new_val
        else:
            if np.random.rand() < np.exp(-d_val / T):
                current_val = new_val
            else:
                # Revert
                current_h[i] = old_hi
                current_h[j] = old_hj
        
        T *= alpha
        
    # Final projection to ensure sum is exactly 50
    diff = N / 2 - np.sum(current_h)
    current_h += diff / N
    current_h = np.clip(current_h, 0, 1)
    
    if abs(np.sum(current_h) - N/2) > 1e-9 * N:
         return [0.5] * N
         
    return list(current_h)
