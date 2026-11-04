"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    Uses a "first-improvement" strategy with early termination on time.
    This allows for more frequent restarts and diversification.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    # We will iterate through swaps. 
    # Instead of a full pass, we do a single pass of "first improvement" per i?
    # Or better: standard first-improvement 2-opt.
    # For each i, find the best j. If improvement, swap and restart i? 
    # Or just continue? 
    # Standard 2-opt usually does: for i, for j, if improve swap and break inner loop (or restart).
    # Given the vectorized nature, let's stick to the current "best improvement for i" logic 
    # but make it faster by checking time more frequently and potentially stopping earlier if no improvements in a full pass.
    
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
            
            # Correction terms
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                # We need to swap row/col i and j in D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
                # In a "first improvement" strategy, we might restart the i-loop or just continue.
                # Continuing is cheaper. Restarting i is more thorough.
                # Let's restart i to ensure we find the best swap for the new i.
                # Actually, for performance, continuing is often better in vectorized code 
                # because we already computed deltas for other j's which are now stale.
                # But since we only updated D for i and j, the deltas for other k are stale.
                # So we MUST restart the inner loop or the outer loop if we want correctness for the next i.
                # However, the current code just continues to the next i.
                # This is technically a "first improvement" heuristic that doesn't re-evaluate previous i's immediately.
                # This is acceptable for a heuristic.
                
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

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    # Surrogate cost: sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    # This is (A @ B)[i,j] + (A.T @ B.T)[i,j]
    cost_matrix = a @ b + a.T @ b.T
    
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _solve(a, b):
    n = len(a)
    start_time = time.time()
    # Hard deadline: 145s to be safe against 150s kill
    time_budget = 145.0
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: High-quality initial solutions
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    # Give it a significant chunk of time to converge
    ls_limit_hung = time_budget * 0.2
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit_hung, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Random Initialization (Diversification)
    # Try a few random starts to escape potential local minima of the Hungarian init
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.3:
        num_randoms = 3
        time_per_random = (time_budget * 0.3 - elapsed) / num_randoms
        for r in range(num_randoms):
            if time.time() - start_time > time_budget * 0.3:
                break
            p_init = np.random.permutation(n)
            p_ls_r, cost_ls_r = _local_search(a, b, p_init, time_per_random, start_time)
            if cost_ls_r < best_cost:
                best_cost = cost_ls_r
                best_p = p_ls_r.copy()
    
    # Phase 2: Iterated Local Search (ILS)
    # Use remaining time for ILS
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        return best_p.tolist()
    
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 5:
        # Check time with safety margin
        if time.time() - start_time > time_budget - 5:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(20, base_perturb + consecutive_no_improve)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        # We want to run LS until it converges or time runs out.
        # But we need to leave time for the next iteration.
        # Strategy: Give LS a fixed chunk of time, or until convergence.
        # Since LS returns when no improvement, it might be fast.
        # Let's give it a generous limit, but check time in the loop.
        
        # To ensure we don't run out of time for the ILS loop itself,
        # we can cap the LS time per iteration if we are getting close to the deadline.
        # However, LS is the heavy lifter. 
        # Let's just let it run until it finishes or hits the global deadline.
        
        p_ls, cost_ls = _local_search(a, b, p_pert, time_budget - 2, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()

def solve(a, b):
    """
    Entry point for the evaluator.
    """
    return _solve(a, b)
# EVOLVE-BLOCK-END
