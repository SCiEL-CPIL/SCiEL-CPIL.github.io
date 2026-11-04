"""
Hybrid QAP solver for wil100: Riemannian CQAP initialization + vectorized steepest-descent
local search + simulated annealing polish.
Combines the structural idea of continuous relaxation on the Stiefel manifold for a
high-quality initial permutation with the fast exact local search and SA machinery.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    # = sum_k (a[k,i]-a[k,j])*b[p[k],p[j]] - sum_k (a[k,i]-a[k,j])*b[p[k],p[i]]
    #   - (a[i,i]-a[i,j])*b[p[i],p[j]] + (a[i,i]-a[i,j])*b[p[i],p[i]]
    #   - (a[j,i]-a[j,j])*b[p[j],p[j]] + (a[j,i]-a[j,j])*b[p[j],p[i]]
    
    # Let's compute this more carefully
    # T1[i,j] = sum_k (a[k,i]-a[k,j])*(Bp[k,j]-Bp[k,i]) for k != i,j
    #         = sum_k (a[k,i]-a[k,j])*Bp[k,j] - sum_k (a[k,i]-a[k,j])*Bp[k,i]
    #           - (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i])
    #           - (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    
    # sum_k (a[k,i]-a[k,j])*Bp[k,j] = (a[:,i]-a[:,j]) @ Bp[:,j] = (a.T @ Bp)[i,j] - ... 
    # Actually: sum_k a[k,i]*Bp[k,j] = (a.T @ Bp)[i,j]
    # sum_k a[k,j]*Bp[k,j] = (a.T @ Bp)[j,j]
    # So sum_k (a[k,i]-a[k,j])*Bp[k,j] = (a.T @ Bp)[i,j] - (a.T @ Bp)[j,j]
    
    # Similarly: sum_k (a[k,i]-a[k,j])*Bp[k,i] = (a.T @ Bp)[i,i] - (a.T @ Bp)[j,i]
    
    # So T1[i,j] = [(a.T @ Bp)[i,j] - (a.T @ Bp)[j,j]] - [(a.T @ Bp)[i,i] - (a.T @ Bp)[j,i]]
    #           - (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i])
    #           - (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    
    ATBp = a.T @ Bp  # M1
    diag_ATBp = np.diag(ATBp)
    
    T1 = ATBp - diag_ATBp[None, :] - diag_ATBp[:, None] + ATBp.T
    
    # Correction terms for k=i and k=j
    # For k=i: (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i])
    # For k=j: (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    
    # Vectorized: for each (i,j), correction = (a[i,i]-a[i,j])*(Bp[i,j]-Bp[i,i]) + (a[j,i]-a[j,j])*(Bp[j,j]-Bp[j,i])
    
    # a[i,i] is diag_a[i], a[i,j] is a[i,j]
    # Bp[i,j] is Bp[i,j], Bp[i,i] is diag_Bp[i]
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # (a[i,i]-a[i,j]) for all (i,j) = diag_a[:,None] - a
    # (Bp[i,j]-Bp[i,i]) for all (i,j) = Bp - diag_Bp[:,None]
    # Product: (diag_a[:,None] - a) * (Bp - diag_Bp[:,None])
    
    # (a[j,i]-a[j,j]) for all (i,j) = a.T - diag_a[None,:]  (since a[j,i] = a.T[i,j], a[j,j] = diag_a[j])
    # (Bp[j,j]-Bp[j,i]) for all (i,j) = diag_Bp[None,:] - Bp.T
    # Product: (a.T - diag_a[None,:]) * (diag_Bp[None,:] - Bp.T)
    
    T1 -= (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    T1 -= (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    # Term 2: sum_{k != i,j} (a[i,k]-a[j,k])*(Bp[j,k]-Bp[i,k])
    # = sum_k (a[i,k]-a[j,k])*Bp[j,k] - sum_k (a[i,k]-a[j,k])*Bp[i,k]
    #   - (a[i,i]-a[j,i])*(Bp[j,i]-Bp[i,i])
    #   - (a[i,j]-a[j,j])*(Bp[j,j]-Bp[i,j])
    
    # sum_k a[i,k]*Bp[j,k] = a[i,:] @ Bp[j,:] = (a @ Bp)[i,j]
    # sum_k a[j,k]*Bp[j,k] = a[j,:] @ Bp[j,:] = (a @ Bp)[j,j]
    # So sum_k (a[i,k]-a[j,k])*Bp[j,k] = (a @ Bp)[i,j] - (a @ Bp)[j,j]
    
    # sum_k a[i,k]*Bp[i,k] = a[i,:] @ Bp[i,:] = (a @ Bp)[i,i]
    # sum_k a[j,k]*Bp[i,k] = a[j,:] @ Bp[i,:] = (a @ Bp)[j,i]
    # So sum_k (a[i,k]-a[j,k])*Bp[i,k] = (a @ Bp)[i,i] - (a @ Bp)[j,i]
    
    ABp = a @ Bp  # M2
    diag_ABp = np.diag(ABp)
    
    T2 = ABp - diag_ABp[None, :] - diag_ABp[:, None] + ABp.T
    
    # Correction for k=i: (a[i,i]-a[j,i])*(Bp[j,i]-Bp[i,i])
    # a[i,i] = diag_a[i], a[j,i] = a.T[i,j]
    # Bp[j,i] = Bp.T[i,j], Bp[i,i] = diag_Bp[i]
    # So: (diag_a[i] - a.T[i,j]) * (Bp.T[i,j] - diag_Bp[i])
    
    # Correction for k=j: (a[i,j]-a[j,j])*(Bp[j,j]-Bp[i,j])
    # a[i,j] = a[i,j], a[j,j] = diag_a[j]
    # Bp[j,j] = diag_Bp[j], Bp[i,j] = Bp[i,j]
    # So: (a[i,j] - diag_a[j]) * (diag_Bp[j] - Bp[i,j])
    
    T2 -= (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    T2 -= (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    # Term 3: (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D

def _perturb_permutation(p, k):
    """
    Perturb a permutation by performing k random swaps.
    """
    n = len(p)
    p_new = p.copy()
    for _ in range(k):
        i = np.random.randint(n)
        j = np.random.randint(n)
        p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new

def _local_search_fast(a, b, p, max_iter=None, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the full delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 1000
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            # Find the indices of the minimum
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            improved = True
            iter_count += 1
        else:
            improved = False
            
    return p, current_cost

def _aggressive_local_search(a, b, p, max_iter=None, time_limit=None, start_time=None, num_swaps_per_iter=3):
    """
    Perform local search that applies multiple best swaps per iteration.
    This can escape shallow local minima faster.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if max_iter is None:
        max_iter = 500
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        D = _compute_delta_matrix(a, b, p)
        
        for _ in range(num_swaps_per_iter):
            min_val = np.min(D)
            
            if min_val < 0:
                min_idx = np.argmin(D)
                i, j = np.unravel_index(min_idx, D.shape)
                
                # Apply swap
                p[i], p[j] = p[j], p[i]
                current_cost += min_val
                improved = True
                
                # Recompute delta matrix after swap
                D = _compute_delta_matrix(a, b, p)
            else:
                break
        
        iter_count += 1
        
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization (fast and good)
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # Local search to refine
    p, current_cost = _local_search_fast(a, b, p, max_iter=1000, time_limit=total_deadline - 10, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Random initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        p = _random_initial(n)
        current_cost = _qap_cost(a, b, p)
        
        # Local search to refine
        p, current_cost = _local_search_fast(a, b, p, max_iter=1000, time_limit=total_deadline - 10, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try more random starts to explore different basins
        num_random_starts = min(15, int(remaining_time // 6))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 8:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # Local search to refine
            p, current_cost = _local_search_fast(a, b, p, max_iter=500, time_limit=total_deadline - 8, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    # Final polish with SA on the best solution found
    # Use a larger portion of the time for SA if we have a good candidate
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 5 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        # SA parameters
        T_start = max(50.0, best_cost * 0.02)
        T_end = 0.1
        alpha = 0.9995  # geometric cooling factor
        
        # Ensure we don't run past the deadline
        sa_deadline = start_time + total_deadline - 2
        
        T = T_start
        while time.time() < sa_deadline:
            # Compute delta matrix for move selection
            D = _compute_delta_matrix(a, b, p)
            
            # Find the best improving move
            min_val = np.min(D)
            
            if min_val < 0:
                # Always accept improving moves
                min_idx = np.argmin(D)
                i, j = np.unravel_index(min_idx, D.shape)
                p[i], p[j] = p[j], p[i]
                current_cost += min_val
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # For non-improving moves, use SA acceptance
                # Pick a random move with some probability
                if np.random.rand() < 0.3:  # 30% chance to try a random move
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        delta = D[i, j]
                        if delta < 0 or np.random.rand() < np.exp(-delta / T):
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = p.copy()
            
            T *= alpha
            if T < T_end:
                T = T_end
    
    # Final high-precision polish: run aggressive local search on the best solution
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 3 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        p, current_cost = _aggressive_local_search(a, b, p, max_iter=200, time_limit=total_deadline - 1, start_time=start_time, num_swaps_per_iter=2)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
