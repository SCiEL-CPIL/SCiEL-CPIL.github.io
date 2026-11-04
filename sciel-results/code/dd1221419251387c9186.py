"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_cost(a, b, p):
    """Compute exact QAP cost for assignment p."""
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _hungarian_init(a, b, n):
    """
    Initialize using Linear Assignment Problem (LAP) on row/col sums.
    This is a standard and effective heuristic for QAP, matching high flow
    facilities to low distance locations.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Calculate row sums (proxy for flow and distance weights)
    r_a = np.sum(a, axis=1)
    r_b = np.sum(b, axis=1)
    
    # Cost matrix for LAP: C[i, j] = r_a[i] * r_b[j]
    # We want to minimize sum C[i, p[i]]
    cost_matrix = np.outer(r_a, r_b)
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        return p
    except Exception:
        return np.random.permutation(n)

def _eig_init(a, b, n):
    """
    Create initial permutation using eigenvectors of symmetric parts.
    """
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Get eigenvalues and eigenvectors
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Use top eigenvectors to create ranking
        # Sort facilities by projection on top eigenvector of A
        top_a = eigvecs_a[:, -1]  # largest eigenvalue
        top_b = eigvecs_b[:, -1]
        
        # Create rankings
        rank_a = np.argsort(np.argsort(top_a))  # higher projection = higher rank
        rank_b = np.argsort(np.argsort(top_b))
        
        # Match: facility with rank r_a goes to location with rank r_b
        p = np.empty(n, dtype=int)
        for i in range(n):
            p[i] = rank_b[rank_a[i]]
        
        return p
    except Exception:
        return np.random.permutation(n)

def _local_search(a, b, p, time_limit, start_time):
    """
    Efficient 2-opt local search with precomputed cost deltas.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute for efficient delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a).copy()
    A_sym_diag = np.diag(A_sym).copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]  # (n-i-1, n)
            diff_D = D[js] - D[i]          # (n-i-1, n)
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction for k=i and k=j terms
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_sym_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D rows and columns
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _polish_pass(a, b, p, time_limit, start_time):
    """
    Final high-precision polish: multiple passes of 2-opt to ensure local optimality.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    A_sym = a + a.T
    A_diag = np.diag(a).copy()
    A_sym_diag = np.diag(A_sym).copy()
    
    # Multiple passes with decreasing time budget
    num_passes = 3
    time_per_pass = (time_limit - (time.time() - start_time)) / num_passes
    
    for pass_num in range(num_passes):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            S = np.sum(diff_A * diff_D, axis=1)
            
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_sym_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
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

def _perturb_3opt(p, k):
    """Apply k random 3-opt moves for better diversification."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        if n < 3:
            break
        i, j, k_idx = random.sample(range(n), 3)
        # Reverse the segment between i and k_idx (assuming i < k_idx)
        if i > k_idx:
            i, k_idx = k_idx, i
        p[i:k_idx+1] = p[i:k_idx+1][::-1]
    return p

def _adaptive_perturb(p, k, stuck_level):
    """
    Adaptive perturbation: use different strategies based on how stuck we are.
    Low stuck: small 2-opt swaps
    Medium stuck: 3-opt reversals  
    High stuck: larger segment reversals or multiple swaps
    """
    p = p.copy()
    n = len(p)
    
    if stuck_level < 3:
        # Small perturbations: k random swaps
        for _ in range(k):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
    elif stuck_level < 8:
        # Medium: k random 3-opt reversals
        for _ in range(k):
            if n < 3:
                break
            i, j, k_idx = random.sample(range(n), 3)
            if i > k_idx:
                i, k_idx = k_idx, i
            p[i:k_idx+1] = p[i:k_idx+1][::-1]
    else:
        # High: larger segment reversals
        for _ in range(max(1, k // 2)):
            if n < 5:
                break
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            # Reverse a segment of random length between i and j
            seg_len = min(j - i + 1, random.randint(3, min(10, n // 4)))
            start = i
            end = min(j, i + seg_len - 1)
            p[start:end+1] = p[start:end+1][::-1]
    
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for overhead
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions
    phase1_time = time_budget * 0.20
    
    # Strategy 1: Hungarian-based initialization
    p_hung = _hungarian_init(a, b, n)
    time_per_init = phase1_time / 5
    p_ls, cost_ls = _local_search(a, b, p_hung, time_per_init, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Strategy 2: Eigenvector-based initialization
    if time.time() - start_time < start_time + phase1_time:
        p_eig = _eig_init(a, b, n)
        time_remaining = phase1_time - (time.time() - start_time)
        if time_remaining > 1:
            p_ls, cost_ls = _local_search(a, b, p_eig, time_remaining, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Strategy 3: Random restarts with shorter LS
    remaining_phase1 = phase1_time - (time.time() - start_time)
    if remaining_phase1 > 2:
        num_random = min(5, int(remaining_phase1 / 2.5))
        for _ in range(num_random):
            if time.time() - start_time > start_time + phase1_time:
                break
            p_init = np.random.permutation(n)
            time_per_ls = min(2.0, (phase1_time - (time.time() - start_time)) / max(1, num_random - _))
            p_ls, cost_ls = _local_search(a, b, p_init, time_per_ls, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Phase 2: Iterated Local Search with adaptive perturbation
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5 or best_p is None:
        return best_p.tolist() if best_p is not None else np.random.permutation(n).tolist()
    
    consecutive_no_improve = 0
    base_perturb = 2
    ls_time_limit = 8.0  # Limit each LS run to allow more restarts
    
    while time.time() - start_time < time_budget - 5:
        if time.time() - start_time > time_budget - 5:
            break
        
        # Adaptive perturbation based on how stuck we are
        if consecutive_no_improve > 8:
            perturb_k = min(12, base_perturb + consecutive_no_improve // 2)
            p_pert = _adaptive_perturb(best_p, perturb_k, consecutive_no_improve)
        else:
            perturb_k = base_perturb
            p_pert = _perturb_3opt(best_p, perturb_k)
        
        # Allocate time for this LS run with a cap
        duration = min(ls_time_limit, (time_budget - 5) - (time.time() - start_time))
        if duration < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Phase 3: Final high-precision polish pass
    if time.time() - start_time < time_budget - 2:
        polish_time = time_budget - 2 - (time.time() - start_time)
        if polish_time > 1:
            best_p, best_cost = _polish_pass(a, b, best_p, polish_time, start_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
