"""Combined QAP solver: vectorized steepest-descent local search + ILS with spectral init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j in permutation p.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1: T1[i, j] = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # Corrections for T1
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2: T2[i, j] = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # Corrections for T2
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    np.fill_diagonal(D, 0)
    return D

def _local_search_steepest(a, b, p, start_time, time_limit, track_best=False):
    """
    Steepest descent local search using full delta matrix.
    Returns (p, cost). If track_best, also returns the best single-swap delta seen.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_single_swap = 0  # track the best (least negative or positive) delta seen
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        
        min_val = np.min(D)
        
        if min_val < 0:
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Track the second-best swap for perturbation hints
            if track_best:
                D_flat = D.ravel()
                # Use max integer value instead of infinity to avoid overflow/type errors
                max_val = np.iinfo(D.dtype).max
                D_flat[min_idx] = max_val
                second_best = np.min(D_flat)
                if second_best > best_single_swap:
                    best_single_swap = second_best
            
            p[i], p[j] = p[j], p[i]
            current_cost += int(min_val)
        else:
            if track_best and min_val > best_single_swap:
                best_single_swap = min_val
            break
            
    if track_best:
        return p, current_cost, best_single_swap
    return p, current_cost

def _spectral_init(a, b):
    """
    Spectral initialization: align principal eigenvectors of A and B.
    """
    n = len(a)
    try:
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        heavy_facilities = np.argsort(-v_a, kind="stable")
        central_locations = np.argsort(-v_b, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        return p
    except Exception:
        # Fallback: rank matching by row sums
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy = np.argsort(-flow_sums, kind="stable")
        central = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Initialize with spectral method
    p = _spectral_init(a, b)
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Track consecutive non-improvements for adaptive perturbation
    consecutive_no_improve = 0
    
    # ILS loop
    while time.time() - start_time < deadline:
        # Local search to convergence, tracking best single-swap delta
        p, current_cost, best_swap_delta = _local_search_steepest(a, b, p, start_time, deadline, track_best=True)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Check time
        if time.time() - start_time > deadline:
            break
        
        # Adaptive perturbation: more swaps when stuck, fewer when improving
        base_swaps = max(2, n // 10)
        if consecutive_no_improve >= 5:
            num_swaps = base_swaps * 2
        elif consecutive_no_improve >= 2:
            num_swaps = base_swaps + 1
        else:
            num_swaps = base_swaps
        
        # Batch all swaps, then compute cost once
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            p[i], p[j] = p[j], p[i]
        
        current_cost = _qap_cost(a, b, p)
        
        # If we've used too much time, break
        if time.time() - start_time > deadline:
            break
    
    # Final polish: run local search multiple times from best_p
    # Use remaining time for a thorough polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        polish_deadline = start_time + deadline - 2.0  # 2s safety margin
        polish_p = best_p.copy()
        polish_cost = best_cost
        # Multiple polish passes with fresh local search
        for _ in range(3):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost = _local_search_steepest(a, b, polish_p, start_time, polish_deadline)
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_p = polish_p.copy()
            # Small perturbation between polish passes
            if time.time() > polish_deadline:
                break
            for _ in range(2):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    polish_p[i], polish_p[j] = polish_p[j], polish_p[i]
            polish_cost = _qap_cost(a, b, polish_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
