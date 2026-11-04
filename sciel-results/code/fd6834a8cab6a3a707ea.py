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
    # Precompute diagonal of A for quick access
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate i from 0 to n-2
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            njs = len(js)
            
            # Vectorized delta calculation
            # diff_A = A_sym[i] - A_sym[js]  (shape: (n-i-1, n))
            # diff_D = D[js] - D[i]         (shape: (n-i-1, n))
            # S = sum(diff_A * diff_D, axis=1)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms for the diagonal and i-j interactions
            # The formula S includes k=i and k=j terms which are incorrect for the swap delta.
            # We need to subtract the contributions from k=i and k=j in the sum, 
            # and add the correct terms for the swap.
            
            # Term for k=i in sum:
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # Term for k=j in sum:
            coeff_j = A_sym[i, js] - A_sym[js, js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            # cross_term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
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

def _generate_initial_solutions(a, b, n, num_solutions):
    """
    Generate diverse initial solutions using different strategies.
    """
    solutions = []
    
    # Strategy 1: Pure random permutations
    for _ in range(num_solutions // 2):
        solutions.append(np.random.permutation(n))
    
    # Strategy 2: Based on pairwise interaction strengths
    # Compute a measure of how "important" each facility is based on flow interactions
    # and how "distant" each location is based on distance interactions
    # This does NOT rely on simple row sums but on the structure of interactions
    
    # Compute flow interaction strength: for each facility i, how much it interacts with others
    flow_strength = np.sum(a, axis=0) + np.sum(a, axis=1)  # Total flow involving facility i
    # Compute distance interaction strength: for each location j, how much it is used
    dist_strength = np.sum(b, axis=0) + np.sum(b, axis=1)  # Total distance involving location j
    
    # Create a bipartite matching based on these strengths
    # But we want to avoid the simple rank-matching that the directive says doesn't work well
    # Instead, use a randomized approach based on these strengths
    
    if num_solutions > num_solutions // 2:
        # Strategy 3: Greedy construction with randomization
        # Build assignment one facility at a time, choosing the location that minimizes
        # partial cost with some randomness
        for _ in range(num_solutions - num_solutions // 2):
            p = np.zeros(n, dtype=int)
            assigned_locations = set()
            
            # Order facilities by flow strength (with some randomness)
            order = np.argsort(flow_strength)[::-1]
            # Add some randomness to the order
            for idx in range(len(order) - 1):
                if random.random() < 0.3:
                    # Swap with a random later position
                    swap_idx = random.randint(idx + 1, len(order) - 1)
                    order[idx], order[swap_idx] = order[swap_idx], order[idx]
            
            for i in order:
                # For facility i, compute the cost of assigning it to each unassigned location
                # Cost = sum_{j already assigned} a[i,j] * b[p[i], p[j]] + sum_{j unassigned} a[i,j] * b[p[i], p[j]]
                # We approximate by considering only assigned locations for now
                
                best_loc = -1
                best_cost = np.inf
                candidates = [l for l in range(n) if l not in assigned_locations]
                
                # Sample a subset of candidates if n is large
                if len(candidates) > 50:
                    candidates = random.sample(candidates, 50)
                
                for loc in candidates:
                    # Compute partial cost
                    partial_cost = 0
                    for j in assigned_locations:
                        if j < i:  # j is a facility index
                            partial_cost += a[i, j] * b[loc, p[j]]
                        else:
                            partial_cost += a[j, i] * b[loc, p[j]]
                    
                    if partial_cost < best_cost:
                        best_cost = partial_cost
                        best_loc = loc
                
                if best_loc == -1 and candidates:
                    best_loc = candidates[0]
                
                p[i] = best_loc
                assigned_locations.add(best_loc)
            
            solutions.append(p)
    
    return solutions

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    num_initials = 8
    initial_solutions = _generate_initial_solutions(a, b, n, num_initials)
    
    # Time allocation for initial local searches
    initial_time_fraction = 0.25
    time_per_initial = (time_budget * initial_time_fraction) / num_initials
    
    for idx, p_init in enumerate(initial_solutions):
        if time.time() - start_time > time_budget * initial_time_fraction:
            break
        
        # Adjust time limit based on remaining budget
        elapsed = time.time() - start_time
        remaining_initial_time = time_budget * initial_time_fraction - elapsed
        time_limit = max(1, remaining_initial_time / (num_initials - idx))
        
        p_ls, cost_ls = _local_search(a, b, p_init, time_limit, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    max_perturb = 20
    
    while time.time() - start_time < time_budget - 2:
        # Check time
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 15:
            perturb_k = min(max_perturb, base_perturb + consecutive_no_improve // 2)
        elif consecutive_no_improve > 8:
            perturb_k = min(max_perturb, base_perturb + consecutive_no_improve // 3)
        else:
            perturb_k = base_perturb + random.randint(0, 2)
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
            break
            
        # Give most of the remaining time to this LS run
        ls_duration = duration * 0.85
        
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
