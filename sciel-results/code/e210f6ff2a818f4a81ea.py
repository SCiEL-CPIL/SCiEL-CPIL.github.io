"""
Simulated Annealing solver for QAPLIB wil100.
Multi-start SA with spectral initialization and vectorized local search polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral embedding.
    Projects facility and location graphs into a shared low-dimensional
    Euclidean space via their Laplacian eigenvectors.
    """
    n = len(a)
    
    A_facility = a.copy()
    A_location = b.copy()
    
    d_facility = A_facility.sum(axis=1)
    d_location = A_location.sum(axis=1)
    
    d_facility[d_facility == 0] = 1
    d_location[d_location == 0] = 1
    
    D_facility_inv_sqrt = np.diag(1.0 / np.sqrt(d_facility))
    D_location_inv_sqrt = np.diag(1.0 / np.sqrt(d_location))
    
    L_facility = np.eye(n) - D_facility_inv_sqrt @ A_facility @ D_facility_inv_sqrt
    L_location = np.eye(n) - D_location_inv_sqrt @ A_location @ D_location_inv_sqrt
    
    try:
        eigvals_f, eigvecs_f = np.linalg.eigh(L_facility)
        eigvals_l, eigvecs_l = np.linalg.eigh(L_location)
        
        k = min(10, n - 1)
        X_facility = eigvecs_f[:, 1:k+1]
        X_location = eigvecs_l[:, 1:k+1]
        
        X_facility = X_facility / np.linalg.norm(X_facility, axis=0, keepdims=True)
        X_location = X_location / np.linalg.norm(X_location, axis=0, keepdims=True)
        
        Xf_sq = np.sum(X_facility**2, axis=1, keepdims=True)
        Xl_sq = np.sum(X_location**2, axis=1, keepdims=True).T
        cross = X_facility @ X_location.T
        dist_sq = Xf_sq + Xl_sq - 2 * cross
        dist_sq = np.maximum(dist_sq, 0)
        
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dist_sq)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        return _greedy_initial(a, b)

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search_polish(a, b, p, current_cost, time_limit):
    """
    Perform First-Improvement local search with multiple passes.
    Uses vectorized delta computation for speed.
    """
    n = len(p)
    start_time = time.time()
    
    B_perm = b[np.ix_(p, p)]
    M1 = a.T @ B_perm
    M3 = a @ B_perm.T
    
    A_ii = np.diag(a)
    B_ii = np.diag(B_perm)
    
    improved = True
    max_passes = 15
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Compute Delta Matrix D vectorized
        I = np.arange(n)
        P = p
        
        # T1 components
        M1_ij = M1[:, P]  # M1[i, P[j]]
        M1_ji = M1[P, :][I[:, None], I[None, :]].T  # M1[P[j], i]
        M1_ii = M1[I, P]  # M1[i, P[i]]
        
        # For T1: sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
        # This is already captured in M1 structure
        # Let's use the direct formula approach
        
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                t1_1 = M1[i, j] - a[i, i] * B_perm[i, j] - a[j, i] * B_ii[j]
                t1_2 = M1[i, i] - a[i, i] * B_ii[i] - a[j, i] * B_perm[j, i]
                t1_3 = M1[j, j] - a[j, j] * B_ii[j] - a[i, j] * B_perm[i, j]
                t1_4 = M1[j, i] - a[j, j] * B_perm[j, i] - a[i, j] * B_ii[i]
                T1 = t1_1 - t1_2 - t1_3 + t1_4
                
                t2_1 = M3[i, j] - a[i, i] * B_perm[j, i] - a[i, j] * B_ii[j]
                t2_2 = M3[i, i] - a[i, i] * B_ii[i] - a[i, j] * B_perm[j, i]
                t2_3 = M3[j, j] - a[j, j] * B_ii[j] - a[j, i] * B_perm[j, i]
                t2_4 = M3[j, i] - a[j, j] * B_perm[i, j] - a[j, i] * B_ii[i]
                T2 = t2_1 - t2_2 - t2_3 + t2_4
                
                T3 = (a[i, j] - a[j, i]) * (B_perm[j, i] - B_perm[i, j])
                
                D[i, j] = T1 + T2 + T3
                D[j, i] = D[i, j]
        
        D_upper = np.triu(D, 1)
        min_val = np.min(D_upper)
        
        if min_val < 0:
            i_idx, j_idx = np.unravel_index(np.argmin(D_upper), D.shape)
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
            current_cost += min_val
            improved = True
            
            B_perm[[i_idx, j_idx], :] = B_perm[[j_idx, i_idx], :]
            B_perm[:, [i_idx, j_idx]] = B_perm[:, [j_idx, i_idx]]
            B_ii = np.diag(B_perm)
            M1 = a.T @ B_perm
            M3 = a @ B_perm.T
        
        if time.time() - start_time > time_limit:
            break
            
    return p, current_cost

def _sa_run(a, b, p_init, time_limit, T_start, T_end):
    """
    Run Simulated Annealing for a given time limit.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        if elapsed == 0:
            T = T_start
        else:
            T = T_start * (T_end / T_start) ** (elapsed / time_limit)
            
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    num_restarts = 6
    time_per_restart = total_deadline / num_restarts
    
    best_p = None
    best_cost = float('inf')
    
    p_spectral = None
    try:
        p_spectral = _spectral_initial(a, b)
    except Exception:
        p_spectral = None
    
    for r in range(num_restarts):
        if time.time() - start_time > total_deadline - 5:
            break
            
        if r == 0:
            if p_spectral is not None:
                p_init = p_spectral.copy()
            else:
                p_init = _greedy_initial(a, b)
        elif r == 1:
            p_init = _greedy_initial(a, b)
        else:
            if p_spectral is not None and r % 2 == 0:
                p_init = p_spectral.copy()
            else:
                p_init = _greedy_initial(a, b)
            
            num_swaps = 5 + r * 2
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                p_init[i], p_init[j] = p_init[j], p_init[i]
                
        sa_time = time_per_restart * 0.75
        polish_time = time_per_restart * 0.25
        
        p_sa, cost_sa = _sa_run(a, b, p_init, sa_time, T_start=1500.0, T_end=0.5)
        p_polished, cost_polished = _local_search_polish(a, b, p_sa, cost_sa, polish_time)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
            
    if best_p is None:
        p_init = _greedy_initial(a, b)
        best_p, best_cost = _local_search_polish(a, b, p_init, _qap_cost(a, b, p_init), 5.0)
        
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 1.0:
        best_p, best_cost = _local_search_polish(a, b, best_p, best_cost, min(remaining_time - 0.5, 5.0))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
