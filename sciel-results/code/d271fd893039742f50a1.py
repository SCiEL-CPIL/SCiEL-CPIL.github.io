"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # Ensure symmetry if not already, though wil100 is symmetric
    # a = (a + a.T) / 2.0
    # b = (b + b.T) / 2.0
    
    # 1. Initialization: Rank matching
    # Heavy flows to short distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    # Helper to compute cost
    def compute_cost(p):
        # p[i] is location of facility i
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Vectorized: B_perm = b[p, :]. Cost = sum a * B_perm[p] ?
        # B_perm[i, j] = b[p[i], p[j]]
        # So cost = np.sum(a * b[np.array(p), np.array(p)])
        p_arr = np.array(p)
        return np.sum(a * b[p_arr, p_arr])

    # Helper to compute delta for swap i, j
    # p[i], p[j] are current locations
    # delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    def delta_swap(i, j, p):
        pi = p[i]
        pj = p[j]
        if pi == pj:
            return 0
        
        # Sum over k != i, j
        # We can iterate k
        sum_val = 0.0
        # Pre-extract rows of a
        ai = a[i]
        aj = a[j]
        # Pre-extract rows of b for locations pi and pj
        b_pi = b[pi]
        b_pj = b[pj]
        
        # We need sum over k of (ai[k] - aj[k]) * (b_pj[p[k]] - b_pi[p[k]])
        # But k != i, j
        # Note: p[k] is the location of facility k.
        
        # This loop is O(n). For n=100, it's fast.
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            sum_val += (ai[k] - aj[k]) * (b_pj[pk] - b_pi[pk])
            
        return 2.0 * sum_val

    # Local Search: 2-opt style until no improvement
    def local_search(p, max_iter=10000):
        improved = True
        iter_count = 0
        best_cost = compute_cost(p)
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Try all pairs? O(n^2) = 10k pairs.
            # For n=100, 4950 pairs.
            # Doing this every iteration might be slow in Python if we do many passes.
            # Better: Random local search or first-improvement with random pairs.
            # Let's do a few passes of random first-improvement.
            for _ in range(100): # 100 random tries per "pass"
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                d = delta_swap(i, j, p)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += d
                    improved = True
        return p, best_cost

    # Simulated Annealing
    def simulated_annealing(p, start_temp, end_temp, steps, deadline):
        current_cost = compute_cost(p)
        best_p = p[:]
        best_cost = current_cost
        
        t = start_temp
        # Cooling factor
        if steps > 1:
            cool_factor = (end_temp / start_temp) ** (1.0 / steps)
        else:
            cool_factor = 0.99
            
        no_improve_count = 0
        restart_threshold = 50000
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            if step % 1000 == 0:
                t = max(end_temp, t * cool_factor)
                # Adaptive restart
                if no_improve_count > restart_threshold:
                    # Reset to best
                    p = best_p[:]
                    current_cost = best_cost
                    # Increase temperature to escape
                    t = start_temp * 0.5 
                    no_improve_count = 0
                    # Re-adjust cooling factor for remaining steps? 
                    # For simplicity, just continue with current t and factor.
                    
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
                
            d = delta_swap(i, j, p)
            
            if d < 0 or random.random() < np.exp(-d / t):
                p[i], p[j] = p[j], p[i]
                current_cost += d
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

        return best_p, best_cost

    # Main execution
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial local search
    p, current_cost = local_search(p, max_iter=200)
    best_p = p[:]
    best_cost = current_cost
    
    # SA Parameters
    # Start temp: estimate max possible delta.
    # Max a ~ 100, max b ~ 100, n=100. Delta ~ 2 * 100 * 100 * 100 = 2e6?
    # Let's use a generous start temp.
    start_temp = 50000.0
    end_temp = 1.0
    
    # Run SA for remaining time
    # We don't know exact steps, so we run until deadline.
    # But the function signature of simulated_annealing takes steps.
    # Let's modify SA to run until deadline.
    
    t = start_temp
    cool_factor = 0.999 # Slow cooling
    no_improve_count = 0
    restart_threshold = 50000
    last_recheck = time.time()
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() - last_recheck > 1.0:
            last_recheck = time.time()
            if no_improve_count > restart_threshold:
                p = best_p[:]
                current_cost = best_cost
                t = start_temp * 0.5
                no_improve_count = 0
                
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
            
        d = delta_swap(i, j, p)
        
        if d < 0 or (t > 1e-9 and random.random() < np.exp(-d / t)):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cool down
        t = max(end_temp, t * cool_factor)
        
        # If we are stuck for a long time, maybe do a local search?
        # The restart mechanism handles this by jumping to best and heating up.
        
    # Final local search polish
    if time.time() < deadline - 5.0:
        best_p, best_cost = local_search(best_p, max_iter=100)
        
    return best_p

# EVOLVE-BLOCK-END
