"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP with multiple sweep passes.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    
    # Do multiple full sweeps until no improvement or time limit
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

def _polish_pass(a, b, p, time_limit, start_time):
    """
    High-precision polish: run local search with multiple restarts from 
    small perturbations of the current solution to escape shallow local optima.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Try a few small perturbations and local search
    for k in [1, 2, 3, 5]:
        if time.time() - start_time > time_limit:
            break
        for trial in range(3):
            if time.time() - start_time > time_limit:
                break
            p_pert = p.copy()
            for _ in range(k):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            # Give this LS run a small time budget
            remaining = time_limit - (time.time() - start_time)
            if remaining < 1:
                break
            ls_time = min(remaining * 0.3, 10)  # cap at 10s per polish LS
            p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    return best_p, best_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _greedy_init(a, b, n):
    """
    Greedy constructive heuristic: build permutation by placing facilities 
    in order of increasing 'importance' (flow row sum) to locations with 
    matching 'importance' (distance row sum), but with some randomization.
    """
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow sum (ascending - place less important first)
    fac_order = np.argsort(flow_sums)
    # Sort locations by distance sum (ascending)
    loc_order = np.argsort(dist_sums)
    
    p = np.zeros(n, dtype=int)
    used_locs = set()
    
    for fac in fac_order:
        # Find the best unused location for this facility
        # Consider only unused locations, pick one with min distance sum
        candidates = [loc for loc in loc_order if loc not in used_locs]
        # Add some randomness: pick from top 5 candidates
        top_k = min(5, len(candidates))
        chosen = random.choice(candidates[:top_k])
        p[fac] = chosen
        used_locs.add(chosen)
    
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple initializations
    # Use a mix of greedy and random starts
    num_initials = 6
    time_per_initial = (time_budget * 0.25) / num_initials
    
    # First try greedy init
    p_greedy = _greedy_init(a, b, n)
    p_ls, cost_ls = _local_search(a, b, p_greedy, time_per_initial, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Then random starts
    for _ in range(num_initials - 1):
        if time.time() - start_time > time_budget * 0.25:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_per_initial, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: ILS
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 3:
        # Check time
        if time.time() - start_time > time_budget - 3:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 8:
            perturb_k = min(20, base_perturb + consecutive_no_improve // 2)
        elif consecutive_no_improve > 4:
            perturb_k = base_perturb + 2
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        duration = (time_budget - 3) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Phase 3: Final polish pass with remaining time
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 5:
        polish_time = min(remaining_time - 2, 20)  # cap polish at 20s
        best_p, best_cost = _polish_pass(a, b, best_p, polish_time, start_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
