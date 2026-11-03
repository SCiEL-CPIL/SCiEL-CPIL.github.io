"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _local_search_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            best_j = -1
            best_delta = 0.0
            
            for j in range(i + 1, n):
                # Compute delta for swapping positions i and j
                a_sym_i = a[i] + a.T[i]
                a_sym_j = a[j] + a.T[j]
                
                S = np.dot(a_sym_i - a_sym_j, D[j] - D[i])
                
                ci = 2*a[i,i] - a[j,i] - a[i,j]
                cj = a[i,j] + a[j,i] - 2*a[j,j]
                
                Ti = ci * (D[j,i] - D[i,i])
                Tj = cj * (D[j,j] - D[i,j])
                
                diag = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
                cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                
                delta = S - Ti - Tj + diag + cross
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta < -1e-9:
                j = best_j
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += best_delta
                improved = True
    
    return p, current_cost

def _perturb(p, k=5, rng=None):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Multiple random initializations
    num_inits = 4
    init_budget = (time_budget - 10) / num_inits
    
    for init_idx in range(num_inits):
        if (time.time() - start_time) > time_budget - 5:
            break
        
        p_init = rng.permutation(n)
        remaining = time_budget - (time.time() - start_time)
        ls_budget = min(init_budget, remaining - 2)
        
        if ls_budget < 1:
            break
        
        p_ls, cost_ls = _local_search_fast(a, b, p_init, ls_budget, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search
    while (time.time() - start_time) < time_budget - 3:
        k_pert = 3 + int(2 * rng.random())
        p_pert = _perturb(best_p, k=k_pert, rng=rng)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
