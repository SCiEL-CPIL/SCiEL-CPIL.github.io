"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation for swapping i and j (where j in js)
            # Delta = sum_{k != i, j} A_sym[i, k] * (D[j, k] - D[i, k]) 
            #       + (a[i,i] - a[j,j]) * (D[j,j] - D[i,i]) 
            #       + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
            # S_partial = sum_{k} A_sym[i, k] * (D[j, k] - D[i, k])
            # This is (D[j] - D[i]) @ A_sym[i]
            S_partial = (D[js] - D[i]) @ A_sym[i]
            
            # Correct for k=i and k=j terms in the sum S_partial
            term_k_i = A_sym[i, i] * (D[js, i] - D[i, i])
            term_k_j = A_sym[i, js] * (D[js, js] - D[i, js])
            
            S_corrected = S_partial - term_k_i - term_k_j
            
            # Add direct terms
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S_corrected + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < -1e-9:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Use multiple random restarts as per directive (row sums are weak predictors)
    # We will run local search on several random starts and the best one found so far.
    
    # Initial phase: Try a few random starts
    num_initials = 5
    time_per_initial = (time_budget * 0.3) / num_initials
    
    for _ in range(num_initials):
        if time.time() - start_time > time_budget * 0.3:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    # Use remaining time for Iterated Local Search
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        # Check time
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        # We want to finish within the budget. 
        # Let's give it a fixed chunk or all remaining?
        # Better to give it a chunk to allow more restarts if it's fast, 
        # but LS usually runs until no improvement or time limit.
        # Let's use a time limit for LS that is 80% of remaining time, 
        # ensuring we can check time after.
        
        ls_time_limit = time.time() + (time_budget - 2 - (time.time() - start_time)) * 0.8
        # Actually, _local_search takes a start_time and a time_limit (absolute or relative?)
        # In my new _local_search, `time_limit` is used as `time.time() - start_time < time_limit`.
        # Wait, in the previous code it was `time.time() - start_time < time_limit`.
        # So `time_limit` is relative to `start_time`.
        
        # Let's pass the absolute end time for the LS run.
        # The function signature is `_local_search(a, b, p, time_limit, start_time)`.
        # It checks `time.time() - start_time < time_limit`.
        # So I should pass `time_budget - 2` as the limit? No, that's the total budget.
        # I should pass the allowed time for *this* run.
        
        # Let's modify the call to use a dynamic limit.
        # The function uses `start_time` passed to it.
        # I will pass `start_time` (global) and a `time_limit` which is the absolute time we want to stop?
        # No, the code says `while time.time() - start_time < time_limit`.
        # So `time_limit` is a duration from `start_time`.
        
        # To run until `time_budget - 2`:
        # duration = (time_budget - 2) - (time.time() - start_time)
        
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
