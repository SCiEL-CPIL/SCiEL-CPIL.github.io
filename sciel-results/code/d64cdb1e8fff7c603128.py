"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank matching
    # Facilities with high flow sums should go to locations with low distance sums (central)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow sum
    # Sort locations by increasing distance sum
    heavy_facilities = np.argsort(-a_row_sums, kind="stable")
    central_locations = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Calculate initial cost
    def calc_cost(p):
        # B[np.ix_(p, p)] creates the permuted distance matrix
        # A * B_perm is element-wise product
        # Sum is the total cost
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing
    # Time limit is 120s. We should use ~110s for search.
    start_time = time.time()
    time_limit = 115.0
    
    # Initial temperature
    # Estimate typical cost difference
    # Let's start with a temperature that allows exploring ~1% of the cost range
    # Or just a fixed heuristic.
    T = 1000.0 
    cooling_rate = 0.999
    
    # We will run SA until time runs out
    # To make it more effective, we can restart if stuck? 
    # For now, standard SA.
    
    # Random seed for reproducibility in local runs, but evaluator might want diversity?
    # The task says "seed/value", implying a fixed seed might be used by evaluator or we should be deterministic?
    # "The search score is seed/value". This usually implies the evaluator runs with a specific seed or we just return the best found.
    # Standard practice: use random.
    
    while time.time() - start_time < time_limit:
        # Pick two random distinct indices
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Swap
        p[i], p[j] = p[j], p[i]
        
        new_cost = calc_cost(p)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Revert
            p[i], p[j] = p[j], p[i]
            
        T *= cooling_rate
        
        # Prevent T from becoming too small too fast, or restart if stuck?
        # If T is very small, we are doing hill climbing.
        # Let's add a simple restart mechanism if no improvement for a long time.
        # For simplicity, just keep going. The cooling rate 0.999 is slow.
        # After 10000 steps, T = 1000 * 0.999^10000 ~ 1000 * 4e-5 ~ 0.04.
        # This is reasonable.
        
        # Optional: Local Search every N iterations to escape local minima faster?
        # Let's add a lightweight local search: check a few random swaps for improvement.
        # Actually, SA with random swaps is a form of local search with noise.
        
        # To speed up, we can batch? No, cost calc is O(N^2).
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
