"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import random

def construct():
    n = 128
    target_low = 128
    target_high = 256
    
    # Precompute the range of 'a' for each sum s
    # a_min[s] = max(0, s-128)
    # a_max[s] = min(128, s//2)
    a_mins = np.array([max(0, s - 128) for s in range(target_low, target_high + 1)])
    a_maxs = np.array([min(128, s // 2) for s in range(target_low, target_high + 1)])
    
    def fast_eval(x_binary):
        """
        x_binary: numpy array of 0s and 1s, length 129.
        Returns (max_count, uncovered_count)
        """
        # Use convolution to compute ordered pair counts
        # y[k] = sum_{i+j=k} x[i]x[j]
        y = np.convolve(x_binary, x_binary, mode='full')
        
        max_count = 0
        uncovered = 0
        for i in range(len(a_mins)):
            s = target_low + i
            # Ordered count is y[s]
            # Unordered count logic:
            # If s is even, check if x[s/2] is 1.
            # N_eq = 1 if s % 2 == 0 and x[s//2] == 1 else 0
            # Count = (y[s] + N_eq) // 2
            
            n_eq = 0
            if s % 2 == 0:
                if x_binary[s // 2] == 1:
                    n_eq = 1
            
            count = (y[s] + n_eq) // 2
            
            if count == 0:
                uncovered += 1
            else:
                if count > max_count:
                    max_count = count
        
        return max_count, uncovered

    def objective(x):
        """
        x: continuous vector in [0, 1]^129
        """
        # Threshold to binary
        x_binary = (x > 0.5).astype(np.int8)
        
        max_count, uncovered = fast_eval(x_binary)
        
        # Heavy penalty for uncovered sums
        penalty = 1000 * uncovered
        return max_count + penalty

    bounds = [(0, 1)] * 129
    
    start_time = time.time()
    time_limit = 50  # seconds for DE
    
    # Run Differential Evolution
    # popsize=15, maxiter=50 might be too slow for 129 dims.
    # Let's try a smaller popsize and fewer iterations, relying on the penalty to guide it.
    # Also, we can use 'best1bin' strategy which is good for binary.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            strategy='best1bin', 
            popsize=10, 
            maxiter=100, 
            tol=0.01, 
            seed=42,
            polish=False  # Polish is local search, we'll do our own
        )
        
        best_x = res.x
        best_val = res.fun
        
    except Exception:
        best_x = np.zeros(129)
        best_val = 10000

    # Convert best_x to binary set
    best_binary = (best_x > 0.5).astype(np.int8)
    best_A = set(np.where(best_binary == 1)[0].tolist())
    
    # Ensure validity and try to improve with a quick local search
    # This part is a refinement, not the primary design, but necessary for robustness.
    
    def get_metrics(A_set):
        x_binary = np.zeros(129, dtype=np.int8)
        for a in A_set:
            x_binary[a] = 1
        return fast_eval(x_binary)
    
    current_max, current_uncovered = get_metrics(best_A)
    
    # If still uncovered, try to fix by adding elements
    if current_uncovered > 0:
        # Simple greedy repair
        for s in range(target_low, target_high + 1):
            # Check if s is covered
            x_binary = np.zeros(129, dtype=np.int8)
            for a in best_A:
                x_binary[a] = 1
            y = np.convolve(x_binary, x_binary, mode='full')
            n_eq = 1 if (s % 2 == 0 and x_binary[s//2] == 1) else 0
            count = (y[s] + n_eq) // 2
            if count == 0:
                # Find a pair (a, b) such that a+b=s, a,b not in A? 
                # Or just add a random element that helps?
                # Let's try adding s//2 if it helps, or other pairs.
                # For now, add s//2 if valid
                mid = s // 2
                if mid not in best_A and 0 <= mid <= 128:
                    best_A.add(mid)
                # If still not covered, add s - 64 and 64? 
                # This is a heuristic repair.
                # To be safe, let's just add a bunch of random elements until covered?
                # Better: iterate through all possible a, check if adding a covers s.
                for a in range(max(0, s-128), min(128, s//2) + 1):
                    b = s - a
                    if a not in best_A and b not in best_A:
                        # Adding both might be expensive, but let's try adding one
                        # Actually, if we add a, we need b to be in A or add b.
                        # This repair is complex. Let's rely on the fact that 
                        # a dense set like [64, 128] always works.
                        pass
                
        # Fallback: if repair is too complex, revert to a known good set?
        # The seed set [64, 128] is always valid.
        # Let's check if current is valid.
        current_max, current_uncovered = get_metrics(best_A)
        if current_uncovered > 0:
             # Use the seed set as fallback
             best_A = set(range(64, 129))
             current_max, current_uncovered = get_metrics(best_A)

    # Local search to minimize max_count
    # Limit time
    local_search_time_limit = 5
    
    for _ in range(200):
        if time.time() - start_time > local_search_time_limit + 50:
            break
        
        # Try removing an element
        if len(best_A) > 1:
            elem = random.choice(list(best_A))
            new_A = best_A - {elem}
            new_max, new_uncovered = get_metrics(new_A)
            if new_uncovered == 0 and new_max <= current_max:
                best_A = new_A
                current_max = new_max
    
    # Try swapping
    for _ in range(200):
        if time.time() - start_time > local_search_time_limit + 50:
            break
        
        if len(best_A) > 1:
            elem = random.choice(list(best_A))
            candidates = [i for i in range(129) if i not in best_A]
            if candidates:
                add_elem = random.choice(candidates)
                new_A = best_A - {elem} | {add_elem}
                new_max, new_uncovered = get_metrics(new_A)
                if new_uncovered == 0 and new_max < current_max:
                    best_A = new_A
                    current_max = new_max

    return list(best_A)
# EVOLVE-BLOCK-END
