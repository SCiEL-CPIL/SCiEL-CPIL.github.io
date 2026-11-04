"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit (150s kill)
    
    # Helper: Calculate exact cost
    def calc_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    # Helper: Calculate delta for swapping facilities i and j in permutation p
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        
        # Vectorized computation for the sum part
        diff_a = a[i] - a[j]
        diff_b = b[lj, p] - b[li, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract terms for k=i and k=j to avoid double counting or incorrect inclusion
        term_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
        term_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
        
        off_diag_sum = total_dot - term_i - term_j
        
        diag_delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
        
        return int(diag_delta + 2 * off_diag_sum)

    # Initial Solution: Linear Assignment on Row Sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # Strategy: Iterated Local Search with Best-Improvement and Adaptive Perturbation
    current_p = p_init.copy()
    current_cost = best_cost
    
    # Track stagnation for adaptive perturbation
    stagnation_count = 0
    max_stagnation = 20
    base_perturb = 4
    
    # Best-improvement local search: find the single best swap in one pass
    def local_search_best_improvement(p, cost, time_limit):
        """Run best-improvement 2-opt local search until local optimum or time limit."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Evaluate all pairs to find the best improving swap
            for i in range(n):
                if time.time() >= time_limit:
                    break
                li = p[i]
                for j in range(i + 1, n):
                    if time.time() >= time_limit:
                        break
                    lj = p[j]
                    if li == lj:
                        continue
                    
                    # Compute delta for swap (i, j)
                    diff_a = a[i] - a[j]
                    diff_b = b[lj, p] - b[li, p]
                    total_dot = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
                    term_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
                    
                    off_diag_sum = total_dot - term_i - term_j
                    diag_delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
                    delta = int(diag_delta + 2 * off_diag_sum)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i >= 0:
                # Perform the best swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
        
        return p, cost
    
    num_iter = 0
    max_iter = 5000
    
    while time.time() < deadline - 8.0 and num_iter < max_iter:
        num_iter += 1
        
        # 1. Local Search with Best Improvement
        ls_start = time.time()
        remaining = deadline - time.time()
        ls_time_limit = min(remaining * 0.6, max(0.5, remaining / 5.0))
        
        current_p, current_cost = local_search_best_improvement(
            current_p, current_cost, ls_start + ls_time_limit
        )
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # 2. Adaptive Perturbation
        if time.time() > deadline - 10.0:
            break
        
        # Increase perturbation strength with stagnation
        perturb_strength = base_perturb + min(stagnation_count // 5, 10)
        
        # Perform random swaps
        for _ in range(perturb_strength):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                current_p[i], current_p[j] = current_p[j], current_p[i]
        
        current_cost = calc_cost(current_p)
        
        # 3. Repair: Short best-improvement local search
        repair_start = time.time()
        repair_time_limit = min(remaining * 0.15, max(0.3, remaining / 20.0))
        
        current_p, current_cost = local_search_best_improvement(
            current_p, current_cost, repair_start + repair_time_limit
        )
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # If stuck for too long, restart from best solution with stronger perturbation
        if stagnation_count > max_stagnation:
            current_p = best_p.copy()
            current_cost = best_cost
            stagnation_count = 0
            # Stronger perturbation on restart
            for _ in range(base_perturb + 8):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost = calc_cost(current_p)
    
    # Final check
    if sorted(best_p.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
