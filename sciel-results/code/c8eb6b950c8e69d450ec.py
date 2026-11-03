"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead

    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def perm_cost(p):
        """Compute cost of permutation p."""
        return int(np.sum(a_f * b_f[np.ix_(p, p)]))

    def sinkhorn(X, max_iter=50, tol=1e-10):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-12)
        
        total = X.sum()
        if total < 1e-12:
            return np.eye(n) / n
        X = X / total
        
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            X = X / row_sums
            
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-12)
            X = X / col_sums
            
            if np.max(np.abs(X.sum(axis=1) - 1.0)) < tol and np.max(np.abs(X.sum(axis=0) - 1.0)) < tol:
                break
        return X

    def swap_delta(p, i, j):
        """Compute cost change from swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        b_pi_p = b_f[pi, p]
        b_pj_p = b_f[pj, p]
        b_p_pi = b_f[p, pi]
        b_p_pj = b_f[p, pj]
        
        delta = 0.0
        delta += np.dot(a_f[i, :], b_pj_p - b_pi_p)
        delta += np.dot(a_f[j, :], b_pi_p - b_pj_p)
        delta += np.dot(a_f[:, i], b_p_pj - b_p_pi)
        delta += np.dot(a_f[:, j], b_p_pi - b_p_pj)
        
        correction = (a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) + 
                      a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj]) + 
                      a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi]) + 
                      a_f[j, i] * (b_f[pi, pi] + b_f[pj, pj] - 2 * b_f[pi, pj]))
                      
        return delta - correction

    def local_search_2opt(p, max_iter=5000, time_budget=5.0):
        """Improve permutation via pairwise swaps with adaptive sampling."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        
        no_improve_count = 0
        best_cost_in_ls = current_cost
        best_p_in_ls = p.copy()
        
        for it in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            improved = False
            # Adaptive number of random pairs to try per iteration
            num_trials = min(200, max(50, int(10000 / (it + 1))))
            
            for _ in range(num_trials):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                delta = swap_delta(p, i, j)
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    if current_cost < best_cost_in_ls:
                        best_cost_in_ls = current_cost
                        best_p_in_ls = p.copy()
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            else:
                no_improve_count = 0
        
        return best_p_in_ls, best_cost_in_ls

    def local_search_systematic(p, time_budget=3.0):
        """Systematic local search: try all swaps in order."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        best_cost = current_cost
        best_p = p.copy()
        
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - t_start > time_budget:
                    break
                
                delta = swap_delta(p, i, j)
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        return best_p, best_cost

    # Phase 1: Generate multiple initial permutations and run continuous relaxation
    # Initialize with rank matching
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central

    # Run continuous relaxation from multiple starting points
    best_relaxed_cost = np.inf
    best_X = None
    best_p_from_relax = None
    best_cost_from_relax = np.inf

    # Starting points: rank matching + random
    start_perms = [p_init]
    for _ in range(2):
        start_perms.append(np.random.permutation(n))

    relax_time_per_start = (time_limit - 20) / (len(start_perms) + 1)

    for start_p in start_perms:
        if time.time() - t0 > time_limit - 10:
            break
        
        # Initialize X from permutation
        X = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X[i, start_p[i]] = 1.0
        X = sinkhorn(X, max_iter=20)

        # Gradient descent with adaptive step size
        step_size = 0.1
        best_local_relaxed = np.inf
        best_local_X = X.copy()
        
        for iteration in range(500):
            if time.time() - t0 > time_limit - 10:
                break
            
            # Compute gradient: A X B^T + B X^T A^T
            AX = a_f @ X
            AXB_T = AX @ b_f.T
            BX_T = b_f @ X.T
            BX_T_A_T = BX_T @ a_f.T
            grad = AXB_T + BX_T_A_T
            
            # Update and project
            X_new = X - step_size * grad
            X_new = sinkhorn(X_new, max_iter=30)
            
            # Compute cost: tr(A X B X^T)
            M = X_new @ b_f @ X_new.T
            cost = np.sum(a_f * M)
            
            if cost < best_local_relaxed:
                best_local_relaxed = cost
                best_local_X = X_new.copy()
                X = X_new
                step_size = min(step_size * 1.02, 0.3)
            else:
                step_size *= 0.9
                if step_size < 1e-7:
                    break

        if best_local_relaxed < best_relaxed_cost:
            best_relaxed_cost = best_local_relaxed
            best_X = best_local_X

            # Recover permutation from relaxed solution
            row_ind, col_ind = linear_sum_assignment(-best_local_X)
            p_relaxed = np.empty(n, dtype=np.int64)
            for i, j in zip(row_ind, col_ind):
                p_relaxed[i] = j
            
            cost_relaxed = perm_cost(p_relaxed)
            if cost_relaxed < best_cost_from_relax:
                best_cost_from_relax = cost_relaxed
                best_p_from_relax = p_relaxed

    # Phase 2: Local search from best relaxed solution and other good starts
    candidates = []
    if best_p_from_relax is not None:
        candidates.append(best_p_from_relax)
    candidates.append(p_init)
    candidates.append(np.random.permutation(n))

    best_p = None
    best_cost = np.inf

    # Time budget for local search
    ls_time_remaining = time_limit - 8 - (time.time() - t0)
    if ls_time_remaining <= 0:
        # Fallback: return best available
        if best_p_from_relax is not None:
            return best_p_from_relax.tolist()
        return p_init.tolist()

    # Run local search on each candidate
    num_candidates = len(candidates)
    time_per_candidate = max(1.0, ls_time_remaining / num_candidates)

    for cand in candidates:
        if time.time() - t0 > time_limit - 5:
            break
        
        # First do systematic local search
        p_sys, cost_sys = local_search_systematic(cand, time_budget=min(2.0, time_per_candidate * 0.4))
        if cost_sys < best_cost:
            best_cost = cost_sys
            best_p = p_sys
        
        # Then do random local search from the best
        remaining_for_this = time_per_candidate - (time.time() - t0) - (time_limit - 5)
        if remaining_for_this > 0:
            p_rand, cost_rand = local_search_2opt(p_sys, max_iter=3000, time_budget=min(3.0, remaining_for_this))
            if cost_rand < best_cost:
                best_cost = cost_rand
                best_p = p_rand

    # Phase 3: If time remains, do more local search from best
    time_remaining = time_limit - 5 - (time.time() - t0)
    if time_remaining > 2 and best_p is not None:
        p_final, cost_final = local_search_2opt(best_p, max_iter=5000, time_budget=time_remaining)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final

    if best_p is None:
        best_p = p_init

    return best_p.tolist()
# EVOLVE-BLOCK-END
