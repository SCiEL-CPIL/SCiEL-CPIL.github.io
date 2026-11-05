import numpy as np
import time
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    even_mask = (targets % 2 == 0)
    odd_mask = ~even_mask
    mid_indices = targets[even_mask] // 2
    
    def get_R(x):
        """Compute unordered pair representation counts R[k] for k in [128,256]."""
        conv = np.convolve(x, x, mode='full')
        conv_vals = conv[targets]
        R = np.zeros(len(targets))
        R[odd_mask] = conv_vals[odd_mask] / 2.0
        mid_vals = x[mid_indices]
        R[even_mask] = (conv_vals[even_mask] + mid_vals) / 2.0
        return R
    
    def get_cost(x):
        R = get_R(x)
        missing = np.sum(np.maximum(0, 1 - R))
        penalty = missing * 10000.0
        # Use sum of squares to minimize peak
        sq_sum = np.sum(R**2)
        return penalty + sq_sum, R
    
    def initial_construct():
        """Create an initial set with good coverage."""
        x = np.zeros(n, dtype=np.int8)
        # Start with a structured set: include elements to cover targets
        # For each target t in [128,256], we need a+b=t with a,b in A
        # A simple approach: include a range of values
        # Let's try a deterministic pattern first, then randomize
        # Include elements that help cover the range
        # If we include [48, 80], then sums range from 96 to 160 - not enough
        # We need sums up to 256, so we need elements up to 128
        # Let's try including a mix
        # Strategy: include elements from two clusters
        # Cluster 1: [64, 96] to cover mid-range
        # Cluster 2: some low and high elements
        size = random.randint(35, 55)
        # Ensure we have some structure
        # Include 64 always (helps with symmetry)
        x[64] = 1
        remaining = size - 1
        # Random sample from remaining
        candidates = list(range(n))
        candidates.remove(64)
        indices = random.sample(candidates, remaining)
        x[indices] = 1
        return x
    
    def sa_optimize(x, time_budget):
        """Run simulated annealing with time budget."""
        start_time = time.time()
        current_cost, _ = get_cost(x)
        best_x = x.copy()
        best_cost = current_cost
        
        start_temp = 50.0
        end_temp = 0.01
        max_steps = 20000
        
        i = 0
        while i < max_steps:
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (i / max_steps)
            
            # Pick a random move: flip a bit
            idx = random.randint(0, n - 1)
            old_val = x[idx]
            x[idx] = 1 - old_val
            
            new_cost, _ = get_cost(x)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            else:
                # Revert
                x[idx] = old_val
            
            i += 1
        
        return best_x, best_cost
    
    # Multiple restarts to find a good solution
    best_result_x = None
    best_result_cost = float('inf')
    
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer
    
    # First, try a few different initial constructions
    for restart in range(8):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        time_per_restart = (time_limit - elapsed) / (8 - restart)
        
        x = initial_construct()
        x, cost = sa_optimize(x, time_per_restart)
        
        if cost < best_result_cost:
            best_result_cost = cost
            best_result_x = x.copy()
    
    # Final polish: run more SA on the best solution found
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    if remaining_time > 2.0 and best_result_x is not None:
        best_result_x, best_result_cost = sa_optimize(best_result_x, remaining_time)
    
    # Convert back to list of indices
    result = np.nonzero(best_result_x)[0].tolist()
    
    # Ensure non-empty
    if not result:
        return [64]
        
    return result
