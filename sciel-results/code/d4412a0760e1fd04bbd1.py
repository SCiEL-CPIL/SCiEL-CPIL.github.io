import numpy as np
import time

N = 100

def compute_value(h):
    """Compute the objective value: max correlation / N * 2"""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2

def local_search(h, max_iter=2000, time_limit=30.0):
    """
    Perform greedy local search by swapping any pair of distinct elements
    to reduce max correlation.
    """
    start_time = time.time()
    current_val = compute_value(h)
    
    # Precompute indices for potential swaps? 
    # Since values change, we just check all pairs i < j where h[i] != h[j].
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_val = current_val
        improved = False
        
        # Try all pairs
        # Optimization: We can skip pairs if they are likely not to help, 
        # but for N=100, 100*99/2 = 4950 pairs is small.
        for i in range(N):
            for j in range(i + 1, N):
                if h[i] == h[j]:
                    continue
                
                # Perform swap
                h[i], h[j] = h[j], h[i]
                
                # Compute new value
                new_val = compute_value(h)
                
                # Undo swap
                h[i], h[j] = h[j], h[i]
                
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_swap = (i, j)
                    improved = True
        
        if not improved:
            break
        
        # Apply best swap
        i, j = best_swap
        h[i], h[j] = h[j], h[i]
        current_val = best_val
    
    return h, current_val

def construct():
    # Define a symmetric multiset of values
    # 20 of each value in [0, 0.25, 0.5, 0.75, 1]
    # Sum = 20 * (0 + 0.25 + 0.5 + 0.75 + 1) = 20 * 2.5 = 50.
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    counts = [20, 20, 20, 20, 20]
    
    # Create the base array
    base_h = np.concatenate([np.full(c, v) for v, c in zip(values, counts)])
    
    best_h = None
    best_val = float('inf')
    
    # Use multiple restarts
    np.random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        # Random permutation of the base values
        h = base_h.copy()
        np.random.shuffle(h)
        
        # Run local search
        # Limit time per restart to ensure we finish in 120s total
        # 50 restarts * 2.0s = 100s. Safe.
        h_optimized, val = local_search(h, max_iter=1000, time_limit=2.0)
        
        if val < best_val:
            best_val = val
            best_h = h_optimized.copy()
            
            # If we find a very good value, we might want to spend more time polishing it
            # but for now, let's stick to the budget.
    
    return best_h
