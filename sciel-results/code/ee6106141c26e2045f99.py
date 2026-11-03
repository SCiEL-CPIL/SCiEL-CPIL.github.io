"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Fast local search with vectorized delta calculations.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    a_sum = a + a.T
    a_diag = np.diag(a)
    D_diag = np.diag(D).copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Compute S vector for all j
            # S_j = a_sum[i] dot D[j] - a_sum[i] dot D[i] - a_sum[j] dot D[j] + a_sum[j] dot D[i]
            
            term1 = a_sum[i] @ D
            term2 = a_sum[i] @ D[i]
            term3 = np.sum(a_sum * D, axis=1)
            term4 = a_sum @ D[i]
            
            S_vec = term1 - term2 - term3 + term4
            
            # Compute Corrections
            # Delta = S - T_i - T_j + diag_term + cross_term
            
            # T_i = (a_sum[i, i] - a_sum[j, i]) * (D[j, i] - D[i, i])
            T_i_vec = (a_sum[i, i] - a_sum[:, i]) * (D[:, i] - D[i, i])
            
            # T_j = (a_sum[i, j] - a_sum[j, j]) * (D[j, j] - D[i, j])
            T_j_vec = (a_sum[i, :] - np.diag(a_sum)) * (D_diag - D[i, :])
            
            # diag_term = (a[i, i] - a[j, j]) * (D[j, j] - D[i, i])
            diag_term_vec = (a[i, i] - a_diag) * (D_diag - D[i, i])
            
            # cross_term = (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
            cross_term_vec = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
            
            delta_vec = S_vec - T_i_vec - T_j_vec + diag_term_vec + cross_term_vec
            
            # Exclude i itself
            delta_vec[i] = np.inf
            
            # Find improving moves
            improving = np.where(delta_vec < 0)[0]
            
            if len(improving) > 0:
                # Pick the best improvement
                best_idx = improving[np.argmin(delta_vec[improving])]
                best_delta = delta_vec[best_idx]
                
                j = best_idx
                
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                # Swap rows i and j
                D[i], D[j] = D[j].copy(), D[i].copy()
                # Swap cols i and j
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                # Update D_diag
                D_diag[i], D_diag[j] = D_diag[j], D_diag[i]
                
                current_cost += best_delta
                improved = True
                
        if not improved:
            break

    return p, current_cost

def _perturb(p, k=5):
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
    
    # Initial random permutation
    rng = np.random.default_rng(42)
    p_init = rng.permutation(n)
    
    # First local search on random init
    p_ls, cost_ls = _local_search_vectorized(a, b, p_init, time_budget - 10, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Iterated Local Search
    while time.time() - start_time < time_budget - 5:
        # Adaptive perturbation size
        k = random.randint(3, 10)
        p_pert = _perturb(best_p, k=k)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, remaining - 2, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls

    return best_p.tolist()
# EVOLVE-BLOCK-END
