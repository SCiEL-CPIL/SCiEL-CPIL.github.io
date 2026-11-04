"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search_2opt_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps using vectorized delta computation.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    a_sym = a + a.T  # symmetric part
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta computation
            diff_a = a_sym[i] - a_sym[js]  # (n-i-1, n)
            diff_D = D[js] - D[i]  # (n-i-1, n)
            
            S = np.sum(diff_a * diff_D, axis=1)
            
            coeff_i = a_sym[i, i] - a_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = a_sym[i, js] - np.diag(a_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                p[i], p[j] = p[j], p[i]
                
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _eigen_init(a, b, n, rng):
    """
    Generate initial permutation using eigenvalue-based heuristic.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        k = min(16, n)
        idx_a = np.argsort(np.abs(eigvals_a))[::-1][:k]
        idx_b = np.argsort(np.abs(eigvals_b))[::-1][:k]
        
        score_a = np.zeros(n)
        for idx in idx_a:
            score_a += np.abs(eigvals_a[idx]) * np.abs(eigvecs_a[:, idx])
        
        score_b = np.zeros(n)
        for idx in idx_b:
            score_b += np.abs(eigvals_b[idx]) * np.abs(eigvecs_b[:, idx])
        
        fac_order = np.argsort(score_a)[::-1]
        loc_order = np.argsort(score_b)
        
        p = np.zeros(n, dtype=int)
        for i in range(n):
            p[fac_order[i]] = loc_order[i]
        
        p_rev = np.zeros(n, dtype=int)
        for i in range(n):
            p_rev[fac_order[i]] = loc_order[n - 1 - i]
        
        cost1 = np.sum(a * b[np.ix_(p, p)])
        cost2 = np.sum(a * b[np.ix_(p_rev, p_rev)])
        
        if cost1 <= cost2:
            return p
        else:
            return p_rev
            
    except:
        return rng.permutation(n)

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _perturb_swap(p, k, rng):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def _perturb_relocate(p, k, rng):
    """
    Relocate k facilities to random different positions.
    This is a k-opt relocation neighborhood, distinct from 2-swap.
    """
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            j = (j + 1) % n
        # Move facility at i to j, shifting others
        val = p[i]
        if i < j:
            p[i:j] = p[i+1:j+1]
        else:
            p[j+1:i+1] = p[j:i]
        p[j] = val
    return p

def _perturb_cycle(p, k, rng):
    """
    Perform a cyclic shift of a sub-permutation of length k.
    """
    p = p.copy()
    n = len(p)
    if k < 3 or k > n:
        return p
    start = rng.integers(0, n - k + 1)
    end = start + k
    segment = p[start:end].copy()
    shift = rng.integers(1, k)
    p[start:end] = np.roll(segment, shift)
    return p

def _vns_perturb(p, level, rng):
    """
    Variable Neighborhood Search perturbation.
    Level 1: 2-swap
    Level 2: Relocation
    Level 3: Cycle
    Higher levels: combinations
    """
    n = len(p)
    if level == 1:
        return _perturb_swap(p, 3, rng)
    elif level == 2:
        return _perturb_relocate(p, 3, rng)
    elif level == 3:
        return _perturb_cycle(p, 5, rng)
    elif level == 4:
        p_temp = _perturb_swap(p, 2, rng)
        return _perturb_relocate(p_temp, 2, rng)
    else:
        # Random mix
        p_temp = _perturb_swap(p, 2, rng)
        p_temp = _perturb_cycle(p_temp, 4, rng)
        return p_temp

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse initial solutions
    # 1. Hungarian initialization (strong structural start)
    p_hung = _hungarian_init(a, b)
    
    # 2. Eigenvalue-based initialization
    p_eigen = _eigen_init(a, b, n, rng)
    
    # 3. Random initialization
    p_random = rng.permutation(n)
    
    # 4. Another random
    p_random2 = rng.permutation(n)
    
    initial_solutions = [p_hung, p_eigen, p_random, p_random2]
    
    # Local search on each initial solution with time division
    # Allocate more time to Hungarian as it's structurally stronger
    time_per_ls = (time_budget - 20) / len(initial_solutions)
    
    for idx, p_init in enumerate(initial_solutions):
        elapsed = time.time() - start_time
        remaining = time_budget - 20 - elapsed
        if remaining < 5:
            break
        
        # Give Hungarian init slightly more time
        ls_time = remaining / (len(initial_solutions) - idx)
        if idx == 0:
            ls_time = min(ls_time * 1.2, remaining)
        
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_init, ls_time, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Variable Neighborhood Search (VNS) phase
    # Systematically expand neighborhood to escape 2-swap local optima
    consecutive_no_improve = 0
    vns_level = 1
    max_vns_level = 5
    
    while time.time() - start_time < time_budget - 5:
        elapsed = time.time() - start_time
        remaining = time_budget - 5 - elapsed
        if remaining < 10:
            break
        
        # Increase VNS level if stuck
        if consecutive_no_improve > 3:
            vns_level = min(max_vns_level, vns_level + 1)
            consecutive_no_improve = 0
        elif consecutive_no_improve > 0 and rng.random() < 0.3:
            # Occasionally try a higher level
            vns_level = min(max_vns_level, vns_level + 1)
        
        # Perturb using VNS neighborhood
        p_pert = _vns_perturb(best_p, vns_level, rng)
        
        # Local search on perturbed solution
        ls_time = remaining - 5
        if ls_time < 5:
            break
            
        p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, ls_time, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            vns_level = 1  # Reset to smallest neighborhood on improvement
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # If we've tried all levels and still no improvement, reset and try random
            if vns_level == max_vns_level:
                vns_level = 1
                if rng.random() < 0.1:
                    # Occasional full random restart
                    p_pert = rng.permutation(n)
                    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_pert, remaining - 5, start_time)
                    if cost_ls < best_cost:
                        best_cost = cost_ls
                        best_p = p_ls.copy()
                        consecutive_no_improve = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
