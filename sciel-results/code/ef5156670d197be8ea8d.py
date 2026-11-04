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
        # Iterate i from 0 to n-2
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms for the diagonal and i-j interactions
            # term_i (k=i part of S that needs correction)
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # term_j (k=j part of S that needs correction)
            coeff_j = A_sym[i, js] - np.diag(A_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
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

def _greedy_init(a, b):
    """
    Greedy initialization based on pairwise interaction strength.
    1. Compute interaction weight for each facility: sum of absolute flows.
    2. Sort facilities by weight descending.
    3. Assign the highest weight facility to a random location.
    4. For subsequent facilities, assign to the location that minimizes the cost
       with all already assigned facilities.
    """
    n = len(a)
    p = np.zeros(n, dtype=int)
    
    # Interaction strength: sum of absolute flows
    # Using sum of abs(a) might be better than sum(a) if a has negatives, 
    # but QAP flows are usually non-negative. Let's use sum of rows of a.
    weights = np.sum(np.abs(a), axis=1)
    
    # Sort facilities by weight (descending)
    order = np.argsort(weights)[::-1]
    
    assigned = np.zeros(n, dtype=bool)
    
    # Assign first facility to a random location
    first_facility = order[0]
    loc = random.randint(0, n - 1)
    p[first_facility] = loc
    assigned[first_facility] = True
    
    # For the rest, pick the best location
    for idx in range(1, n):
        fac = order[idx]
        
        # Calculate cost of placing fac at each unassigned location
        # Cost contribution of fac to already assigned facilities k:
        # sum_{k assigned} a[fac, k] * b[loc, p[k]] + a[k, fac] * b[p[k], loc]
        
        # We can vectorize this.
        # Let's gather the assigned facilities indices
        assigned_facs = np.where(assigned)[0]
        
        if len(assigned_facs) == 0:
            # Should not happen
            continue
            
        # b_sub: matrix of distances from all locations to assigned locations
        # b[all_locs, assigned_locs]
        b_to_assigned = b[:, assigned_facs] # shape (n, num_assigned)
        
        # a_fac: flows from fac to assigned facs
        a_to_assigned = a[fac, assigned_facs] # shape (num_assigned,)
        a_from_assigned = a[assigned_facs, fac] # shape (num_assigned,)
        
        # Total flow interaction for a location l:
        # sum_{k} (a[fac, k] * b[l, p[k]] + a[k, fac] * b[p[k], l])
        
        # Note: b is not symmetric necessarily.
        # b[l, p[k]] is row l of b, cols p[k].
        # b[p[k], l] is row p[k] of b, col l.
        
        # cost[l] = sum_k a[fac, k] * b[l, p[k]] + sum_k a[k, fac] * b[p[k], l]
        
        # term1 = b_to_assigned @ a_to_assigned  (shape n)
        # b_to_assigned is (n, m), a_to_assigned is (m,). Result (n,).
        term1 = b_to_assigned @ a_to_assigned
        
        # term2: sum_k a[k, fac] * b[p[k], l]
        # This is equivalent to: for each l, sum_k a[k, fac] * b[p[k], l]
        # Let B_assigned_to_all = b[np.ix_(assigned_locs, all_locs)] ? No.
        # We need b[p[k], l]. This is the transpose of b[all_locs, p[k]] for the specific rows.
        # b[assigned_locs, :] has shape (m, n).
        # We want to sum over k: a[k, fac] * b[assigned_locs[k], :]
        # This is a_to_assigned.T @ b[assigned_locs, :]
        
        assigned_locs = p[assigned_facs]
        b_assigned_to_all = b[assigned_locs, :] # shape (m, n)
        
        term2 = a_from_assigned @ b_assigned_to_all # shape (n,)
        
        total_cost = term1 + term2
        
        # Mask assigned locations
        total_cost[assigned] = np.inf
        
        best_loc = np.argmin(total_cost)
        p[fac] = best_loc
        assigned[fac] = True
        
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Diverse Initializations
    # Mix of Random and Greedy starts
    
    # 1. Greedy Start
    p_greedy = _greedy_init(a, b)
    p_ls, cost_ls = _local_search(a, b, p_greedy, 5, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        
    # 2. Random Starts
    num_randoms = 4
    elapsed = time.time() - start_time
    remaining_for_inits = time_budget * 0.2 - elapsed
    if remaining_for_inits > 1:
        time_per_random = remaining_for_inits / num_randoms
        for _ in range(num_randoms):
            if time.time() - start_time > time_budget * 0.2:
                break
            p_init = np.random.permutation(n)
            p_ls, cost_ls = _local_search(a, b, p_init, time_per_random, start_time)
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
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
             break
             
        # Give LS 90% of remaining time to allow for multiple quick restarts if LS is fast?
        # No, LS usually runs until convergence or time limit.
        # If we give it all remaining time, we only do one restart.
        # Better to cap the LS time to allow more restarts?
        # For QAP, deep local search is valuable.
        # Let's give it the full remaining time minus a small buffer.
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
