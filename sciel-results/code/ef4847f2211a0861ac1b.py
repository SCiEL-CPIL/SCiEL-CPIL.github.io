"""
Simulated Annealing solver for QAPLIB wil100.
Uses spectral initialization, vectorized Iterated Local Search, and
annealing schedule that balances exploration and exploitation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral decomposition.
    Aligns eigenvectors of flow and distance matrices to find a good
    geometric initialization.
    """
    n = len(a)
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    
    # Compute top eigenvectors (use SVD for efficiency and stability)
    try:
        _, _, vt_a = np.linalg.svd(a_centered, full_matrices=False)
        _, _, vt_b = np.linalg.svd(b_centered, full_matrices=False)
        
        # Use the top few principal components
        k = min(10, n)
        U_a = vt_a[:k].T  # n x k
        U_b = vt_b[:k].T  # n x k
        
        # Compute the cross-similarity matrix
        sim = U_a @ U_b.T  # n x n
        
        # Use a simple greedy matching based on spectral alignment
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        # Sort facilities by their "importance" in the spectral space
        facility_scores = np.abs(U_a).sum(axis=1)
        facility_order = np.argsort(-facility_scores, kind="stable")
        
        for fac in facility_order:
            # Find the best available location
            available = np.where(~used)[0]
            if len(available) == 0:
                break
            
            # Score available locations based on spectral similarity
            loc_scores = sim[fac, available]
            # We want to maximize alignment, so pick the highest score
            best_loc_idx = np.argmax(loc_scores)
            best_loc = available[best_loc_idx]
            p[fac] = best_loc
            used[best_loc] = True
        
        return p
    
    except np.linalg.LinAlgError:
        # Fallback to greedy if SVD fails
        return _greedy_initial(a, b)

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

def _compute_all_swap_deltas(a, b, p):
    """
    Compute swap deltas for all pairs (i, j) with i < j.
    Returns a 2D array of shape (n, n) where entry (i,j) is the delta
    for swapping positions i and j (upper triangle only).
    """
    n = len(p)
    deltas = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            deltas[i, j] = _compute_swap_delta(a, b, p, i, j)
    
    return deltas

def _fast_local_search(a, b, p, time_limit=None, start_time=None, max_iter=5000):
    """
    Perform fast local search using precomputed delta matrix.
    Reuses the delta matrix and updates it incrementally after each swap.
    Returns the improved permutation and cost.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute all swap deltas
    deltas = _compute_all_swap_deltas(a, b, p)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find the best improving swap
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Vectorized search for best improving move
        mask = deltas < 0
        if not np.any(mask):
            break
        
        # Find the minimum delta
        flat_idx = np.argmin(deltas)
        i, j = divmod(flat_idx, n)
        best_delta = deltas[i, j]
        
        if best_delta >= 0:
            break
        
        best_i = i
        best_j = j
        
        # Perform the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Update the delta matrix incrementally
        # Only need to update rows/cols involving i and j
        # For simplicity and correctness, recompute deltas involving i or j
        for k in range(n):
            if k != best_i and k != best_j:
                deltas[best_i, k] = _compute_swap_delta(a, b, p, best_i, k)
                deltas[k, best_i] = deltas[best_i, k]
                deltas[best_j, k] = _compute_swap_delta(a, b, p, best_j, k)
                deltas[k, best_j] = deltas[best_j, k]
        deltas[best_i, best_j] = _compute_swap_delta(a, b, p, best_i, best_j)
        deltas[best_j, best_i] = deltas[best_i, best_j]
    
    return best_p, best_cost

def _iterated_local_search(a, b, p, time_limit=None, start_time=None, perturbation_strength=0.05):
    """
    Perform Iterated Local Search using fast local search for local optimization.
    """
    n = len(p)
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # First local search
    p, current_cost = _fast_local_search(a, b, p, time_limit=time_limit, start_time=start_time)
    
    best_p = p.copy()
    best_cost = current_cost
    
    # ILS loop
    while time.time() - start_time < time_limit:
        # Perturb the solution
        num_swaps = max(1, int(n * perturbation_strength))
        p_perturbed = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
        
        # Local search on perturbed solution
        p_new, cost_new = _fast_local_search(a, b, p_perturbed, time_limit=time_limit, start_time=start_time)
        
        # Accept if better
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new.copy()
            current_cost = cost_new
        elif cost_new < current_cost:
            p = p_new.copy()
            current_cost = cost_new
        else:
            # With small probability, accept worse to escape local optima
            if np.random.rand() < 0.1:
                p = p_new.copy()
                current_cost = cost_new
    
    return best_p, best_cost

def _greedy_construction(a, b):
    """
    Construct a permutation using a greedy heuristic.
    """
    n = len(a)
    p = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    flow_sums = a.sum(axis=1)
    order = np.argsort(-flow_sums)
    
    placed = []
    for i in order:
        best_loc = -1
        best_cost = float('inf')
        for l in range(n):
            if used[l]: continue
            cost = a[i,i] * b[l,l]
            for j in placed:
                cost += a[i,j] * b[l, p[j]] + a[j,i] * b[p[j], l]
            if cost < best_cost:
                best_cost = cost
                best_loc = l
        p[i] = best_loc
        used[best_loc] = True
        placed.append(i)
        
    return p

def _hungarian_initial(a, b):
    """
    Create an initial permutation using the Hungarian algorithm on a
    surrogate cost matrix derived from spectral alignment.
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(a)
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    
    # Compute top eigenvectors
    try:
        _, _, vt_a = np.linalg.svd(a_centered, full_matrices=False)
        _, _, vt_b = np.linalg.svd(b_centered, full_matrices=False)
        
        k = min(10, n)
        U_a = vt_a[:k].T
        U_b = vt_b[:k].T
        
        # Compute cross-similarity
        sim = U_a @ U_b.T
        
        # Convert to cost matrix (negate similarity to minimize)
        cost_matrix = -sim
        
        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
    
    except Exception:
        return _spectral_initial(a, b)

def solve(a, b):
    n = len(a)
    total_deadline = 140.0
    start_time = time.time()
    
    # Multi-start: try several different initializations
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Hungarian initialization (better starting point)
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    p, current_cost = _iterated_local_search(a, b, p, time_limit=total_deadline - 30, start_time=start_time)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Start 2: Spectral initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 20:
        p = _spectral_initial(a, b)
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _iterated_local_search(a, b, p, time_limit=total_deadline - 20, start_time=start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Start 3: Greedy initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        p = _greedy_initial(a, b)
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _iterated_local_search(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Start 4: Random initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        p = _random_initial(n)
        current_cost = _qap_cost(a, b, p)
        p, current_cost = _iterated_local_search(a, b, p, time_limit=total_deadline - 10, start_time=start_time)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # Final polish with SA on the best solution found, with tighter cooling
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.005)
        T_end = 0.01
        elapsed_total = total_deadline
        
        while time.time() - start_time < total_deadline - 2:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    if best_p is None:
        best_p = _spectral_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
