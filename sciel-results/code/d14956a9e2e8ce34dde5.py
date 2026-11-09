import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55
    
    # Precompute all APs
    all_aps = []
    for d in range(1, n):
        for r in range(d):
            indices = list(range(r, n, d))
            if indices:
                all_aps.append(indices)
    
    # Sample a subset for fast cost
    rng = random.Random(42)
    sample_aps = rng.sample(all_aps, min(2000, len(all_aps)))
    
    def get_cost_fast(s):
        max_abs = 0
        for indices in sample_aps:
            curr = 0
            for idx in indices:
                curr += s[idx]
                if curr > max_abs:
                    max_abs = curr
                elif curr < -max_abs:
                    max_abs = -curr
        return max_abs
    
    def get_cost_full(s):
        max_abs = 0
        for indices in all_aps:
            curr = 0
            for idx in indices:
                curr += s[idx]
                if curr > max_abs:
                    max_abs = curr
                elif curr < -max_abs:
                    max_abs = -curr
        return max_abs
    
    s = [rng.choice((-1, 1)) for _ in range(n)]
    current_cost = get_cost_fast(s)
    best_s = s[:]
    best_cost = get_cost_full(s)
    
    # Simulated annealing with restarts
    T = 5.0
    cooling = 0.999999
    min_T = 0.001
    
    while time.time() < deadline:
        T = 5.0
        # Reset to best found so far
        s = best_s[:]
        current_cost = get_cost_fast(s)
        
        while time.time() < deadline and T > min_T:
            pos = rng.randint(0, n - 1)
            old_val = s[pos]
            s[pos] = -old_val
            
            new_cost = get_cost_fast(s)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                # Check full cost periodically
                if new_cost < best_cost:
                    full_cost = get_cost_full(s)
                    if full_cost < best_cost:
                        best_cost = full_cost
                        best_s = s[:]
            else:
                if rng.random() < np.exp(-(new_cost - current_cost) / T):
                    current_cost = new_cost
                else:
                    s[pos] = old_val
            
            T *= cooling
        
        # If we're close to deadline, do a final full check
        if time.time() > deadline - 5:
            break
    
    # Final full cost check on best_s
    final_cost = get_cost_full(best_s)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_s
