"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with low-distance locations.
    """
    n = a.shape[0]
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        eigvals_a, eigvecs_a = eigh(a_sym)
        v_a = eigvecs_a[:, -1]
        
        eigvals_b, eigvecs_b = eigh(b_sym)
        v_b = eigvecs_b[:, -1]
        
        facility_rank = np.argsort(-v_a)
        location_rank = np.argsort(v_b)
        
        p = np.empty(n, dtype=np.int64)
        p[facility_rank] = location_rank
    except Exception:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
    return p


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)


def _local_search(a, b, p):
    """
    Vectorized 2-swap local search.
    Returns (p, cost, improved_flag)
    """
    n = len(p)
    # Precompute M = A^T B
    M = a.T @ b
    diag_a = np.diag(a)
    
    cost = _compute_cost(a, b, p)
    p = p.copy()
    
    while True:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            x = p[i]
            # Create mask for j != i
            j_indices = np.arange(n)
            j_indices = j_indices[j_indices != i]
            y = p[j_indices]
            
            # Vectorized calculation of Delta for all j given i
            # Sum_all = M[i, y] - M[i, x] + M[y, x] - M[y, y]
            M_i_y = M[i, y]
            M_i_x = M[i, x]
            M_y_x = M[y, x]
            M_y_y = M[y, y]
            
            Sum_all = M_i_y - M_i_x + M_y_x - M_y_y
            
            # Term_i: (A_ii - A_ij) * (B_iy - B_ix)
            # A_ii is scalar, A_ij is vector a[i, j_indices]
            # B_iy is vector b[i, y], B_ix is scalar b[i, x]
            A_ii = a[i, i]
            A_ij = a[i, j_indices]
            B_iy = b[i, y]
            B_ix = b[i, x]
            Term_i = (A_ii - A_ij) * (B_iy - B_ix)
            
            # Term_j: (A_ji - A_jj) * (B_jy - B_jx)
            # A_ji is vector a[j_indices, i]
            # A_jj is vector diag_a[j_indices]
            # B_jy is vector b[j_indices, y]
            # B_jx is vector b[j_indices, x]
            A_ji = a[j_indices, i]
            A_jj = diag_a[j_indices]
            B_jy = b[j_indices, y]
            B_jx = b[j_indices, x]
            Term_j = (A_ji - A_jj) * (B_jy - B_jx)
            
            Sum_neq = Sum_all - Term_i - Term_j
            
            # Delta_self
            B_yy = b[y, y]
            B_xx = b[x, x]
            B_yx = b[y, x]
            B_xy = b[x, y]
            
            A_jj_vec = A_jj
            A_ij_vec = A_ij
            A_ji_vec = A_ji
            
            Delta_self = (A_ii * (B_yy - B_xx) + 
                          A_jj_vec * (B_xx - B_yy) + 
                          A_ij_vec * (B_yx - B_xy) + 
                          A_ji_vec * (B_xy - B_yx))
            
            Delta = Sum_neq + Delta_self
            
            min_idx = np.argmin(Delta)
            delta_val = Delta[min_idx]
            
            if delta_val < best_delta:
                best_delta = delta_val
                best_i = i
                best_j = j_indices[min_idx]
        
        if best_delta < 0:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
        else:
            break
            
    return p, cost


def _iterated_local_search(a, b, p_init, time_budget=110):
    start_time = time.time()
    n = len(p_init)
    
    p_best = p_init.copy()
    cost_best = _compute_cost(a, b, p_best)
    
    p_current = p_best.copy()
    
    while time.time() - start_time < time_budget:
        # Local search
        p_current, cost_current = _local_search(a, b, p_current)
        
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        
        # Perturbation
        # Random 2-swap perturbation
        num_swaps = 1
        for _ in range(num_swaps):
            i, j = np.random.choice(n, size=2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
            
    return p_best, cost_best


def solve(a, b):
    n = a.shape[0]
    p_spectral = _spectral_init(a, b)
    p_best, cost_best = _iterated_local_search(a, b, p_spectral, time_budget=110)
    return p_best.tolist()
# EVOLVE-BLOCK-END
