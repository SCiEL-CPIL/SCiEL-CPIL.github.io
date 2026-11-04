"""
Optimized QAP solver for wil100 using vectorized steepest-descent local search
with a fast delta-matrix update, multi-start, and a final SA polish.
"""
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j in p.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] * Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] * Bp[k, j]
    M2 = a @ Bp
    
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # T1[i, j] = sum_{k != i,j} (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i])
    # Raw: M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    # Corrections for k=i and k=j
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    T1 = T1_raw - C1 - C2
    
    # T2[i, j] = sum_{k != i,j} (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
    # Raw: M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    T2 = T2_raw - C3 - C4
    
    # T3[i, j] = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    np.fill_diagonal(D, 0)
    return D

def _update_delta_matrix(D, a, b, p, i, j):
    """
    Update the delta matrix D after swapping positions i and j in p.
    Returns the updated D.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    Bp = b[np.ix_(p, p)]
    
    # For each position k, the delta D[k, i] and D[k, j] change.
    # Also D[i, k] and D[j, k] change (symmetry).
    # And D[i, j] and D[j, i] change.
    
    # Let's recompute the rows and columns i and j of D.
    # D[k, i] for all k:
    # D[k, i] = sum_{m != k,i} (a[m,k] - a[m,i]) * (Bp[m,i] - Bp[m,k]) + T2_part + T3_part
    # This is complex. Instead, let's just recompute the full D if n is small enough,
    # or recompute only the affected rows/cols.
    
    # Actually, recomputing the full D is O(n^3) which is ~1e6 ops for n=100.
    # Updating rows/cols is O(n^2) per row/col, so O(n^2) total.
    # Let's try the O(n^2) update.
    
    # The delta for swapping k and i (k != i, j):
    # D_new[k, i] = sum_{m != k,i} (a[m,k] - a[m,i]) * (Bp_new[m,i] - Bp_new[m,k]) + ...
    # Note that Bp_new is Bp with rows/cols i and j swapped.
    # Bp_new[m, i] = Bp[m, j] for m != i, j; Bp_new[i, i] = Bp[j, j]; Bp_new[j, i] = Bp[j, i]
    # This is getting complicated. Let's just recompute D fully. It's fast enough.
    # For n=100, D computation is ~100^3 = 1e6 multiplies, which is ~1ms.
    # With 1000 iterations, that's ~1s. We have 115s budget.
    
    return _compute_delta_matrix(a, b, p)

def _local_search(a, b, p, time_limit, start_time):
    """
    Steepest descent local search using the delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        min_val = np.min(D)
        
        if min_val < 0:
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
        else:
            break
    
    return p, current_cost

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _sa_polish(a, b, p, current_cost, time_limit, start_time):
    """
    Simulated annealing polish on the current solution.
    """
    n = len(p)
    T_start = max(1.0, current_cost * 0.01)
    T_end = 0.1
    elapsed_total = time_limit - (time.time() - start_time)
    if elapsed_total < 1:
        return p, current_cost
    
    best_p = p.copy()
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Compute delta for swap (i, j)
        pi = p[i]
        pj = p[j]
        
        # Use the full delta computation for a single swap
        # This is O(n) per swap
        Bp = b[np.ix_(p, p)]
        # D[i, j] = sum_{k != i,j} (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i]) + ...
        # Let's just compute it directly
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = Bp[:, j] - Bp[:, i]
        T1 = np.dot(diff_a_col, diff_b_row)
        T1 -= (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
        T1 -= (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
        
        diff_a_row = a[i, :] - a[j, :]
        diff_b_col = Bp[j, :] - Bp[i, :]
        T2 = np.dot(diff_a_row, diff_b_col)
        T2 -= (a[i, i] - a[j, i]) * (Bp[j, i] - Bp[i, i])
        T2 -= (a[i, j] - a[j, j]) * (Bp[j, j] - Bp[i, j])
        
        T3 = (a[i, j] - a[j, i]) * (Bp[j, i] - Bp[i, j])
        
        delta = T1 + T2 + T3
        new_cost = current_cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
        
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
    
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    p, current_cost = _local_search(a, b, p, total_deadline - 30, start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Try additional random starts if time permits
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        num_starts = min(3, int(remaining_time // 15))
        for _ in range(num_starts):
            if time.time() - start_time > total_deadline - 30:
                break
            p = _random_initial(n)
            p, current_cost = _local_search(a, b, p, total_deadline - 30, start_time)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final SA polish
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        best_p, best_cost = _sa_polish(a, b, best_p.copy(), best_cost, total_deadline - 1, start_time)
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
