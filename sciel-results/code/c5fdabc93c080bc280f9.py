"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _calc_cost(a, b, p):
    """Calculate exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search using vectorized delta calculations.
    For each i, computes deltas for all j > i in a single NumPy operation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _calc_cost(a, b, p)
    
    # Precompute symmetric A for delta calculation
    # Delta formula for swapping i and j:
    # d = sum_k (a[i,k] - a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]]) + corrections
    # Using A_sym = a + a.T simplifies the correction terms.
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    # We maintain D = b[np.ix_(p, p)] implicitly or explicitly?
    # Explicit D is faster for updates but uses memory. n=256, 256x256 int64 is ~512KB. Very small.
    D = b[np.ix_(p, p)].copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        improved = False
        # Iterate through all i
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation for all j in js
            # diff_A = A_sym[i] - A_sym[js]  (shape: (n-i-1, n))
            # diff_D = D[js] - D[i]         (shape: (n-i-1, n))
            # S = sum(diff_A * diff_D, axis=1)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms:
            # The sum S includes terms for k=i and k=j which are not correct for the swap delta.
            # We need to remove their contribution and add the correct diagonal/cross terms.
            
            # Term for k=i in the sum: (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            # Note: A_sym is symmetric, so A_sym[j,i] = A_sym[i,j].
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # Term for k=j in the sum: (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            # diag_term = (a[i,i] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[i]])
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            # cross_term = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
            # Note: a[i,j] - a[j,i] is the antisymmetric part.
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

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    Cost[i, j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

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
    deadline = start_time + 110.0  # Safety margin before 120s limit (150s kill)
    
    best_p = None
    best_cost = np.inf
    
    # 1. Initialization Phase: Generate diverse starts
    # Start 1: Hungarian
    p_hung = _hungarian_init(a, b)
    
    # Start 2: Random
    p_rand1 = np.random.permutation(n)
    
    # Start 3: Random
    p_rand2 = np.random.permutation(n)
    
    initial_solutions = [p_hung, p_rand1, p_rand2]
    
    # 2. Initial Refinement
    # Allocate ~30% of time to refining initial solutions to find a good "anchor"
    initial_time_budget = 0.3 * (deadline - start_time)
    time_per_init = initial_time_budget / len(initial_solutions)
    
    for p_init in initial_solutions:
        if time.time() > deadline - 5:
            break
        p_ls, cost_ls = _local_search_vectorized(a, b, p_init, time_per_init, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 3. Iterated Local Search (ILS) Phase
    # Use remaining time for ILS starting from best_p
    current_p = best_p.copy()
    current_cost = best_cost
    
    stagnation_count = 0
    base_perturb = 4
    max_stagnation = 15
    
    while time.time() < deadline - 10.0:
        # Adaptive perturbation
        if stagnation_count > max_stagnation:
            perturb_k = base_perturb + (stagnation_count // 5)
            # Restart from best if stuck too long
            if stagnation_count > max_stagnation * 2:
                current_p = best_p.copy()
                current_cost = best_cost
                stagnation_count = 0
        else:
            perturb_k = base_perturb
        
        p_pert = _perturb(current_p, perturb_k)
        
        # Refine perturbed solution
        # Give it a decent chunk of time, but ensure we can do multiple iterations
        # Dynamic time limit: min(remaining_time * 0.5, 20s)
        remaining = deadline - time.time()
        ls_time = min(remaining * 0.5, 20.0)
        
        if ls_time < 1.0:
            break
            
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, ls_time, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            current_p = p_ls.copy()
            current_cost = cost_ls
            stagnation_count = 0
        else:
            stagnation_count += 1
            # If no improvement, keep the perturbed solution as current for next iteration?
            # Or go back to best? Standard ILS usually keeps the perturbed one if it's not too bad,
            # but going back to best is safer for convergence. Let's stick to best if no improvement.
            current_p = best_p.copy()
            current_cost = best_cost
    
    # Final check
    if best_p is None:
        return list(range(n))
        
    if sorted(best_p.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
