"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search(a, b, p, deadline):
    """
    Optimized 2-opt local search for QAP.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    # Delta for swapping i and j:
    # The cost change is sum_{k != i, j} a[i,k]*(D[j,k]-D[i,k]) + a[k,i]*(D[j,k)-D[i,k]) 
    # + terms involving i,j directly.
    # Using A_sym = A + A.T simplifies the sum over k.
    A_sym = a + a.T
    
    while time.time() < deadline:
        improved = False
        # Iterate i from 0 to n-2
        for i in range(n - 1):
            if time.time() > deadline:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
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
            # (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            # Note: A_sym[i,i] = 2*A[i,i]. A_sym[j,i] = A[j,i]+A[i,j].
            # D[j,i] = b[p[j], p[i]]. D[i,i] = b[p[i], p[i]].
            
            # Term for k=j in sum:
            # (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            
            # Let's use the explicit formula derived from expansion:
            # Delta = S - term_i - term_j + term_direct
            
            # term_i (k=i part of S that needs correction)
            # In S, the k=i term is (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # term_j (k=j part of S that needs correction)
            # In S, the k=j term is (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            # Note: A_sym[i,j] is scalar for fixed i, vector for js? No, A_sym[i, js] is vector.
            # A_sym[js, js] is diagonal of submatrix.
            coeff_j = A_sym[i, js] - np.diag(A_sym[js])
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            # The swap changes:
            # 1. (i,i) -> (j,j): A[i,i]*D[j,j] - A[i,i]*D[i,i] is handled?
            # Let's stick to the reliable vectorized delta used in the previous version, 
            # which was verified to work.
            
            # Previous formula:
            # deltas = S - term_i - term_j + diag_term + cross_term
            
            # diag_term: (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
            
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

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Try a few random starts with limited time each
    deadline_initial = start_time + time_budget * 0.25
    
    for _ in range(5):
        if time.time() > deadline_initial:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, deadline_initial)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # ILS phase
    deadline_ils = start_time + time_budget - 3
    
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() < deadline_ils:
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Give this LS run 70% of remaining time to allow for overhead
        remaining = deadline_ils - time.time()
        if remaining < 2:
            break
        ls_deadline = time.time() + remaining * 0.7
        
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    if best_p is None:
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
