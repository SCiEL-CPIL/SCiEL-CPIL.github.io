import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    # Ensure A is a numpy boolean array for fast bitwise operations
    A = np.array(adjacency, dtype=bool)
    
    def calc_cut(x):
        x_col = x.reshape(-1, 1)
        same_mask = (x_col == x_col.T)
        # Count edges that are present in A but NOT on the same side
        return int((A * ~same_mask).sum())

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    global_best_x = None
    global_best_cut = -1
    
    # Multi-start Simulated Annealing
    while time.time() < deadline - 5.0:
        # Random initialization
        x = np.random.randint(0, 2, n)
        current_cut = calc_cut(x)
        
        if current_cut > global_best_cut:
            global_best_cut = current_cut
            global_best_x = x.copy()
            
        T = 2.0
        # Run SA for a fixed duration or until temperature is low
        run_start = time.time()
        run_deadline = run_start + 5.0  # 5 seconds per restart
        
        while time.time() < run_deadline and T > 0.01:
            v = np.random.randint(n)
            
            # Calculate delta for flipping vertex v
            # Neighbors with same label as v
            same_count = (A[v] & (x == x[v])).sum()
            # Neighbors with different label than v
            diff_count = (A[v] & (x != x[v])).sum()
            delta = same_count - diff_count
            
            # Metropolis acceptance criterion
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                x[v] = 1 - x[v]
                current_cut += delta
                if current_cut > global_best_cut:
                    global_best_cut = current_cut
                    global_best_x = x.copy()
            
            T *= 0.9999  # Cooling rate
            
    return global_best_x
