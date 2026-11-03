"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the QAP objective: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Efficient computation using matrix multiplication
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Let B_p[i,j] = b[p[i], p[j]], then cost = sum_{i,j} a[i,j] * B_p[i,j]
    # = np.sum(a * B_p)
    # But we can also write: cost = trace(a^T @ B_p) if symmetric, but let's be general.
    # Actually, sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's construct the permuted b matrix
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _compute_surrogate_cost(a, b, k=5):
    """
    Compute a surrogate cost matrix for the linear assignment problem
    using the top-k principal components of the flow and distance matrices.
    
    The idea: project the matrices onto their principal components to capture
    structural correlations, then build a surrogate cost for assignment.
    """
    n = len(a)
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    
    # SVD for top-k components
    # For flow matrix A: we want to capture the structure of flows
    # For distance matrix B: we want to capture the structure of distances
    
    # Use truncated SVD
    try:
        Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
    except:
        # Fallback to row sums
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        surrogate = np.outer(a_sums, b_sums)
        return surrogate
    
    # Project onto top-k components
    Ua_k = Ua[:, :k]
    Va_k = Va.T[:, :k]  # shape (n, k)
    Ub_k = Ub[:, :k]
    Vb_k = Vb.T[:, :k]  # shape (n, k)
    
    # The surrogate cost for assigning facility i to location j:
    # We want heavy flows (high a[i,:]) to go to short distances (low b[:,j])
    # Using the PC projections:
    # facility_score_i = Ua_k[i, :] weighted by singular values
    # location_score_j = Vb_k[j, :] weighted by singular values
    
    # Compute facility scores based on flow structure
    # A[i,:] projected onto principal components
    facility_scores = Ua_k * Sa[:k]  # (n, k)
    location_scores = Vb_k * Sb[:k]  # (n, k)
    
    # Surrogate cost: negative dot product (we want high flow with low distance)
    # Actually, let's think more carefully.
    # The QAP objective is sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we approximate a[i,j] ≈ sum_k u_ik * s_k * v_jk and b[p[i], p[j]] ≈ sum_l u'_pl * s'_l * v'_ql
    # This gets complicated. Let's use a simpler approach.
    
    # Simpler: use the row sums and column sums as primary, with PC correction
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Base surrogate: outer product of row sums (facility i's total flow) and location j's total distance
    # We want high flow facilities at low-distance locations
    surrogate = np.outer(a_row_sums, b_row_sums)
    
    # Add PC-based correction: facilities with high load in direction of high-distance PC should avoid those locations
    # For each PC direction, compute the "alignment" cost
    for k_idx in range(min(k, len(Sa), len(Sb))):
        # Facility i's projection on flow PC k_idx
        fi = Ua[:, k_idx] * Sa[k_idx]
        # Location j's projection on distance PC k_idx  
        lj = Vb[:, k_idx] * Sb[k_idx]
        # If both are high in the same direction, it's bad (high flow * high distance)
        surrogate += np.outer(fi, lj)
    
    return surrogate


def _local_search(a, b, p, max_iters=5000, time_limit=None):
    """
    Iterated local search with 2-opt swaps.
    """
    n = len(p)
    p = p.copy()
    
    # Precompute for fast delta computation
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap p[i] and p[j], the change involves:
    # For all k != i, j: a[k,i]*b[p[k],p[j]] + a[k,j]*b[p[k],p[i]] + a[i,k]*b[p[i],p[k]] + a[j,k]*b[p[j],p[k]]
    # minus: a[k,i]*b[p[k],p[i]] + a[k,j]*b[p[k],p[j]] + a[i,k]*b[p[i],p[k]] + a[j,k]*b[p[j],p[k]]
    # This is complex. Let's use a simpler approach: recompute affected terms.
    
    def _delta_swap(i, j):
        """Compute the change in cost if we swap positions i and j in p."""
        pi = p[i]
        pj = p[j]
        
        if pi == pj:
            return 0
        
        # The change comes from terms involving i or j
        # Old contributions involving i or j:
        # sum_k a[k,i]*b[p[k],pi] + a[i,k]*b[pi,p[k]] + a[k,j]*b[p[k],pj] + a[j,k]*b[pj,p[k]]
        # But when k=i or k=j, we have self-terms
        # New contributions:
        # sum_k a[k,i]*b[p[k],pj] + a[i,k]*b[pj,p[k]] + a[k,j]*b[p[k],pi] + a[j,k]*b[pi,p[k]]
        
        # Let's compute the difference
        # Terms where k is not i or j:
        # a[k,i]*(b[p[k],pj] - b[p[k],pi]) + a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + 
        # a[k,j]*(b[p[k],pi] - b[p[k],pj]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
        
        # Self terms (k=i or k=j):
        # When k=i: a[i,i]*(b[pj,pj] - b[pi,pi]) + a[i,i]*(b[pj,pi] - b[pi,pj]) ... this is getting messy
        
        # Let's just compute the full delta by summing over all k
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            # Old: a[k,i]*b[pk,pi] + a[i,k]*b[pi,pk] + a[k,j]*b[pk,pj] + a[j,k]*b[pj,pk]
            # New: a[k,i]*b[pk,pj] + a[i,k]*b[pj,pk] + a[k,j]*b[pk,pi] + a[j,k]*b[pi,pk]
            delta += (a[k, i] * (b[pk, pj] - b[pk, pi]) + 
                      a[i, k] * (b[pj, pk] - b[pi, pk]) + 
                      a[k, j] * (b[pk, pi] - b[pk, pj]) + 
                      a[j, k] * (b[pi, pk] - b[pj, pk]))
        
        # Now handle k=i and k=j cases
        # When k=i:
        # Old: a[i,i]*b[pi,pi] + a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        # Wait, let me be more careful.
        # The cost is sum_{x,y} a[x,y] * b[p[x], p[y]]
        # Terms involving index i as first or second argument, and similarly for j.
        
        # Let's recompute from scratch for the affected pairs
        # Affected pairs: (i, i), (i, j), (j, i), (j, j), and (k, i), (i, k), (k, j), (j, k) for k != i, j
        
        # Old cost for these pairs:
        old_ii = a[i, i] * b[pi, pi]
        old_ij = a[i, j] * b[pi, pj]
        old_ji = a[j, i] * b[pj, pi]
        old_jj = a[j, j] * b[pj, pj]
        
        # New cost for these pairs (after swap, p[i]=pj, p[j]=pi):
        new_ii = a[i, i] * b[pj, pj]
        new_ij = a[i, j] * b[pj, pi]
        new_ji = a[j, i] * b[pi, pj]
        new_jj = a[j, j] * b[pi, pi]
        
        # For k != i, j:
        # Old: a[k,i]*b[pk,pi] + a[i,k]*b[pi,pk] + a[k,j]*b[pk,pj] + a[j,k]*b[pj,pk]
        # New: a[k,i]*b[pk,pj] + a[i,k]*b[pj,pk] + a[k,j]*b[pk,pi] + a[j,k]*b[pi,pk]
        
        # We already computed the sum over k != i,j above as 'delta'
        # But wait, I think I double-counted. Let me redo.
        
        # Actually, let's just compute the total delta properly
        delta_total = 0
        
        # Pairs (i,i), (i,j), (j,i), (j,j)
        delta_total += (new_ii - old_ii) + (new_ij - old_ij) + (new_ji - old_ji) + (new_jj - old_jj)
        
        # Pairs (k,i), (i,k), (k,j), (j,k) for k != i, j
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            delta_total += (a[k, i] * (b[pk, pj] - b[pk, pi]) + 
                           a[i, k] * (b[pj, pk] - b[pi, pk]) + 
                           a[k, j] * (b[pk, pi] - b[pk, pj]) + 
                           a[j, k] * (b[pi, pk] - b[pj, pk]))
        
        return delta_total
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    improved = True
    iter_count = 0
    start_time = time.time()
    
    while improved and iter_count < max_iters:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        improved = False
        iter_count += 1
        
        # Try all pairs (i, j) and find the best improving swap
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i + 1, n):
                d = _delta_swap(i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_delta < 0:
            # Apply the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
            best_p = p.copy()
            improved = True
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110  # Leave margin for the 120s limit
    
    # Step 1: Generate initial permutation using surrogate LAP
    surrogate = _compute_surrogate_cost(a, b, k=5)
    
    # Solve LAP on the surrogate
    row_ind, col_ind = linear_sum_assignment(surrogate)
    p0 = np.empty(n, dtype=np.int64)
    p0[row_ind] = col_ind
    
    # Also try the simple rank-matching as another starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    
    # Evaluate both
    cost0 = _qap_cost(a, b, p0)
    cost1 = _qap_cost(a, b, p1)
    
    if cost0 <= cost1:
        best_p = p0.copy()
        best_cost = cost0
    else:
        best_p = p1.copy()
        best_cost = cost1
    
    # Step 2: Local search from the best starting point
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 5:
        best_p, best_cost = _local_search(a, b, best_p, max_iters=10000, time_limit=remaining - 2)
    
    # Step 3: Try a few more random restarts if time permits
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    rng = np.random.default_rng(42)
    
    while remaining > 30:
        # Random perturbation of the best solution
        p_curr = best_p.copy()
        
        # Apply a few random swaps
        n_swaps = max(1, n // 10)
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
        
        # Local search
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 10:
            break
        
        p_ls, cost_ls = _local_search(a, b, p_curr, max_iters=2000, time_limit=min(remaining - 5, 20))
        
        if cost_ls < best_cost:
            best_p = p_ls
            best_cost = cost_ls
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
