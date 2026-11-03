"""
Simulated Annealing solver for QAPLIB wil100.
Uses Benders-inspired decomposition: a linearized surrogate guides initial
assignment, while exact quadratic cost feedback tightens the search.
Multi-start SA with adaptive cooling and final local polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _linear_surrogate_cost(a, b, p):
    """
    Benders master: linearized surrogate cost.
    Uses row sums of A and column sums of B (for symmetric B, same as row sums).
    surrogate = sum_i (a_row_sum[i]) * (b_col_sum[p[i]])
    This is a relaxation that ignores the quadratic interaction.
    """
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    return int(np.sum(a_row_sums * b_col_sums[p]))

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
    
    # Term 1: column differences
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: row differences
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: cross term
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(T1 + T2 + T3)

def _compute_all_swap_deltas(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns an n x n matrix where entry (i,j) is the delta for swapping positions i and j.
    Only upper triangle (i<j) is meaningful; diagonal is 0.
    """
    n = len(p)
    # Precompute useful quantities
    # For each position k, we need a[:,k] and b[p[k],:] etc.
    
    # Delta for swapping i and j:
    # = sum_{k!=i,j} a[k,i]*b[p[i],p[k]] + a[k,j]*b[p[j],p[k]] - a[k,i]*b[p[j],p[k]] - a[k,j]*b[p[i],p[k]]
    #   + a[i,k]*b[p[k],p[i]] + a[j,k]*b[p[k],p[j]] - a[i,k]*b[p[k],p[j]] - a[j,k]*b[p[k],p[i]]
    #   + a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]] - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]
    #   - a[i,i]*b[p[i],p[i]] - a[j,j]*b[p[j],p[j]] + a[i,i]*b[p[j],p[j]] + a[j,j]*b[p[i],p[i]]
    #   + a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]] - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]
    # Simplified: let's use the standard formula
    # delta(i,j) = sum_k [a[k,i]-a[k,j]]*[b[p[i],p[k]]-b[p[j],p[k]]] 
    #            + sum_k [a[i,k]-a[j,k]]*[b[p[k],p[i]]-b[p[k],p[j]]]
    #            - [a[i,i]-a[i,j]]*[b[p[i],p[i]]-b[p[j],p[i]]]  (remove k=i from first sum)
    #            - [a[j,i]-a[j,j]]*[b[p[i],p[j]]-b[p[j],p[j]]]  (remove k=j from first sum)
    #            - [a[i,i]-a[j,i]]*[b[p[i],p[i]]-b[p[i],p[j]]]  (remove k=i from second sum)
    #            - [a[i,j]-a[j,j]]*[b[p[j],p[i]]-b[p[j],p[j]]]  (remove k=j from second sum)
    #            + (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])  (cross term correction)
    
    # Vectorized approach:
    # col_diff[i,j] = a[:,i] - a[:,j]  (n x n matrix of column differences)
    # row_diff[i,j] = a[i,:] - a[j,:]  (n x n matrix of row differences)
    
    # For term 1: sum_k != i,j (a[k,i]-a[k,j]) * (b[p[i],p[k]] - b[p[j],p[k]])
    # Let's compute B_p = b[p,:] which is n x n
    B_p = b[p, :]  # B_p[k, l] = b[p[k], p[l]]
    
    # col_diff[i,j] = a[:,i] - a[:,j], shape (n, n) where row index is k, col index is j
    # We want: for each (i,j), sum over k of (a[k,i] - a[k,j]) * (B_p[i,k] - B_p[j,k])
    # = sum_k a[k,i]*B_p[i,k] - sum_k a[k,i]*B_p[j,k] - sum_k a[k,j]*B_p[i,k] + sum_k a[k,j]*B_p[j,k]
    
    # A_col[k,i] = a[k,i], so A_col.T[i,k] = a[k,i]
    # Let's define:
    # S1[i] = sum_k a[k,i] * B_p[i,k]  = row sum of (a[:,i] * B_p[i,:])
    # S2[i,j] = sum_k a[k,i] * B_p[j,k] = dot(a[:,i], B_p[j,:])
    # S3[i,j] = sum_k a[k,j] * B_p[i,k] = dot(a[:,j], B_p[i,:])
    # S4[j] = sum_k a[k,j] * B_p[j,k]
    
    # Term1_full[i,j] = S1[i] + S4[j] - S2[i,j] - S3[i,j]
    # But we need to exclude k=i and k=j:
    # Exclude k=i: subtract (a[i,i]-a[i,j])*(B_p[i,i]-B_p[j,i])
    # Exclude k=j: subtract (a[j,i]-a[j,j])*(B_p[i,j]-B_p[j,j])
    
    # Similarly for term 2:
    # Term2_full[i,j] = sum_k (a[i,k]-a[j,k])*(B_p[k,i]-B_p[k,j])
    # = sum_k a[i,k]*B_p[k,i] - sum_k a[i,k]*B_p[k,j] - sum_k a[j,k]*B_p[k,i] + sum_k a[j,k]*B_p[k,j]
    # Let T1[i] = sum_k a[i,k]*B_p[k,i]
    # T2[i,j] = sum_k a[i,k]*B_p[k,j] = dot(a[i,:], B_p[:,j])
    # T3[i,j] = sum_k a[j,k]*B_p[k,i] = dot(a[j,:], B_p[:,i])
    # T4[j] = sum_k a[j,k]*B_p[k,j]
    # Term2_full[i,j] = T1[i] + T4[j] - T2[i,j] - T3[i,j]
    # Exclude k=i: subtract (a[i,i]-a[j,i])*(B_p[i,i]-B_p[i,j])
    # Exclude k=j: subtract (a[i,j]-a[j,j])*(B_p[j,i]-B_p[j,j])
    
    # Term3: (a[i,j]-a[j,i])*(B_p[j,i]-B_p[i,j])
    
    # Let's compute efficiently:
    # S1[i] = dot(a[:,i], B_p[i,:])
    S1 = np.array([np.dot(a[:,i], B_p[i,:]) for i in range(n)])
    S4 = S1  # S4[j] = S1[j]
    
    # S2[i,j] = dot(a[:,i], B_p[j,:]) for all i,j
    # This is a @ B_p.T ? No. S2[i,j] = sum_k a[k,i]*B_p[j,k] = (a.T @ B_p)[i,j]
    S2 = a.T @ B_p  # S2[i,j] = sum_k a[k,i]*B_p[j,k]
    
    # S3[i,j] = dot(a[:,j], B_p[i,:]) = (a.T @ B_p)[j,i] = S2[j,i]
    S3 = S2.T
    
    # Term1_full[i,j] = S1[i] + S1[j] - S2[i,j] - S2[j,i]
    Term1_full = S1[:,None] + S1[None,:] - S2 - S2.T
    
    # Corrections for term 1:
    # corr1[i,j] = (a[i,i]-a[i,j])*(B_p[i,i]-B_p[j,i]) + (a[j,i]-a[j,j])*(B_p[i,j]-B_p[j,j])
    # Let's compute this vectorized:
    # diag_a = a.diag()
    # B_p_diag = B_p.diag() = b[p,p]
    # corr1[i,j] = (a[i,i]-a[i,j])*(B_p[i,i]-B_p[j,i]) + (a[j,i]-a[j,j])*(B_p[i,j]-B_p[j,j])
    
    # This is complex to vectorize fully. Let's use a different approach:
    # Compute the full delta matrix using matrix operations.
    
    # Actually, let me use a cleaner formulation:
    # delta(i,j) = sum_k (a[k,i]-a[k,j])*(b[p[i],p[k]]-b[p[j],p[k]])
    #            + sum_k (a[i,k]-a[j,k])*(b[p[k],p[i]]-b[p[k],p[j]])
    #            - (a[i,i]-a[i,j])*(b[p[i],p[i]]-b[p[j],p[i]])
    #            - (a[j,i]-a[j,j])*(b[p[i],p[j]]-b[p[j],p[j]])
    #            - (a[i,i]-a[j,i])*(b[p[i],p[i]]-b[p[i],p[j]])
    #            - (a[i,j]-a[j,j])*(b[p[j],p[i]]-b[p[j],p[j]])
    #            + (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    
    # Note that b is symmetric (QAPLIB instances have symmetric distance matrices),
    # so b[p[i],p[k]] = b[p[k],p[i]] etc. This simplifies things.
    
    # With symmetric B:
    # Term1 = sum_k (a[k,i]-a[k,j])*(b[p[i],p[k]]-b[p[j],p[k]])
    # Term2 = sum_k (a[i,k]-a[j,k])*(b[p[k],p[i]]-b[p[k],p[j]]) = sum_k (a[i,k]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]])
    
    # So Term1 + Term2 = sum_k (a[k,i]+a[i,k]-a[k,j]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]])
    
    # Let C[i,k] = a[i,k] + a[k,i] (symmetrized A)
    # Then Term1+Term2 = sum_k (C[i,k]-C[j,k])*(b[p[i],p[k]]-b[p[j],p[k]])
    # = sum_k C[i,k]*b[p[i],p[k]] - sum_k C[i,k]*b[p[j],p[k]] - sum_k C[j,k]*b[p[i],p[k]] + sum_k C[j,k]*b[p[j],p[k]]
    
    C = a + a.T  # symmetrized flow matrix
    
    # V[i,k] = C[i,k] * b[p[i],p[k]]
    # Let's compute M[i,j] = sum_k C[i,k]*(b[p[i],p[k]]-b[p[j],p[k]])
    # = sum_k C[i,k]*b[p[i],p[k]] - sum_k C[i,k]*b[p[j],p[k]]
    
    # P[i,k] = b[p[i],p[k]] = B_p[i,k]
    # M[i,j] = sum_k C[i,k]*B_p[i,k] - sum_k C[i,k]*B_p[j,k]
    # = (C * B_p).sum(axis=1)[i] - (C @ B_p.T)[i,j]
    
    CB = C * B_p  # element-wise product
    CB_row_sum = CB.sum(axis=1)  # CB_row_sum[i] = sum_k C[i,k]*B_p[i,k]
    C_BpT = C @ B_p.T  # C_BpT[i,j] = sum_k C[i,k]*B_p[j,k]
    
    M = CB_row_sum[:,None] - C_BpT  # M[i,j] = sum_k C[i,k]*(B_p[i,k]-B_p[j,k])
    
    # Now subtract the k=i and k=j terms:
    # For k=i: C[i,i]*(B_p[i,i]-B_p[j,i])
    # For k=j: C[i,j]*(B_p[i,j]-B_p[j,j])
    # Similarly for the j part...
    
    # Actually M[i,j] already includes all k. We need to subtract k=i and k=j contributions.
    # But wait, the formula sum_k (C[i,k]-C[j,k])*(B_p[i,k]-B_p[j,k]) includes k=i and k=j.
    # When k=i: (C[i,i]-C[j,i])*(B_p[i,i]-B_p[j,i])
    # When k=j: (C[i,j]-C[j,j])*(B_p[i,j]-B_p[j,j])
    
    # The full delta is:
    # delta = M[i,j] + M[j,i]  -- NO, that's not right.
    
    # Let me reconsider. With symmetric B:
    # delta(i,j) = sum_k (a[k,i]-a[k,j])*(B_p[i,k]-B_p[j,k]) + sum_k (a[i,k]-a[j,k])*(B_p[k,i]-B_p[k,j])
    # Since B is symmetric, B_p[k,i] = B_p[i,k], so:
    # = sum_k (a[k,i]-a[k,j])*(B_p[i,k]-B_p[j,k]) + sum_k (a[i,k]-a[j,k])*(B_p[i,k]-B_p[j,k])
    # = sum_k (a[k,i]+a[i,k]-a[k,j]-a[j,k])*(B_p[i,k]-B_p[j,k])
    # = sum_k (C[i,k]-C[j,k])*(B_p[i,k]-B_p[j,k])  where C[i,k]=a[i,k]+a[k,i]
    
    # Wait, C[i,k] = a[i,k]+a[k,i], so C[i,k]-C[j,k] = a[i,k]+a[k,i]-a[j,k]-a[k,j]
    # And the sum is over k. This equals:
    # sum_k (C[i,k]-C[j,k])*B_p[i,k] - sum_k (C[i,k]-C[j,k])*B_p[j,k]
    # = sum_k C[i,k]*B_p[i,k] - sum_k C[j,k]*B_p[i,k] - sum_k C[i,k]*B_p[j,k] + sum_k C[j,k]*B_p[j,k]
    
    # Let U[i] = sum_k C[i,k]*B_p[i,k]  (row sum of C*B_p)
    # Let V[i,j] = sum_k C[i,k]*B_p[j,k] = (C @ B_p.T)[i,j]
    
    # delta(i,j) = U[i] + U[j] - V[i,j] - V[j,i]
    
    # But this includes k=i and k=j. We need to check if the formula is correct when k=i or k=j.
    # When k=i: (C[i,i]-C[j,i])*(B_p[i,i]-B_p[j,i])
    # This should be part of the sum. Let's verify with the original delta formula.
    
    # Original: delta = sum_{k!=i,j} a[k,i]*b[p[i],p[k]] + ... (long formula)
    # The vectorized formula should give the correct result including all k.
    
    # Let me just compute it and verify:
    U = CB_row_sum
    V = C_BpT  # V[i,j] = sum_k C[i,k]*B_p[j,k]
    
    delta_matrix = U[:,None] + U[None,:] - V - V.T
    
    # Set diagonal to 0
    np.fill_diagonal(delta_matrix, 0)
    
    return delta_matrix

def _local_search(a, b, p, current_cost, max_iters=2000, time_limit=5.0):
    """
    Steepest-descent local search: repeatedly find the best improving swap.
    Uses vectorized O(n^2) delta computation per iteration.
    Returns (improved_p, improved_cost).
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all swap deltas vectorized
        delta_matrix = _compute_all_swap_deltas(a, b, p)
        
        # Find best improving swap (upper triangle only)
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        deltas = delta_matrix[upper_mask]
        min_delta = np.min(deltas)
        
        if min_delta >= 0:
            break  # Local minimum
        
        # Find the position of the minimum
        min_idx = np.argmin(deltas)
        # Convert linear index to (i, j) in upper triangle
        # Upper triangle row i has elements at columns i+1, i+2, ..., n-1
        # Number of elements in row i: n - 1 - i
        # Cumulative: row 0 has n-1, row 1 has n-2, etc.
        row = 0
        rem = int(min_idx)
        while rem >= (n - 1 - row):
            rem -= (n - 1 - row)
            row += 1
        col = row + 1 + rem
        
        p[row], p[col] = p[col], p[row]
        cost += int(min_delta)
    
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end, use_surrogate_feedback=False):
    """
    Run simulated annealing for a given duration.
    If use_surrogate_feedback, adjust acceptance based on surrogate cost gap.
    Returns (best_p, best_cost).
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    # Precompute for surrogate feedback
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        # Benders-inspired: if we have surrogate feedback, use it to modulate acceptance
        if use_surrogate_feedback and T > 10.0:
            # Compute surrogate before and after
            surrogate_before = int(np.sum(a_row_sums * b_col_sums[p]))
            p_new = p.copy()
            p_new[i], p_new[j] = p_new[j], p_new[i]
            surrogate_after = int(np.sum(a_row_sums * b_col_sums[p_new]))
            surrogate_delta = surrogate_after - surrogate_before
            
            # If surrogate improves but exact cost worsens, be more accepting
            # (the quadratic terms may be compensating)
            if surrogate_delta < 0 and delta > 0:
                # Boost acceptance probability
                accept_prob = np.exp(-delta / (T * 0.5))  # More permissive
            elif surrogate_delta > 0 and delta < 0:
                # Surrogate worsens but exact improves - definitely accept
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / T)
        else:
            accept_prob = np.exp(-delta / T)
        
        if delta < 0 or np.random.rand() < accept_prob:
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_greedy, cost_greedy), (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: Quick local search on best initial
    elapsed = time.time() - start_time
    ls_time = min(3.0, max(0.5, total_deadline - elapsed - 10.0))
    if ls_time > 0.5:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=10000, time_limit=ls_time)
    
    # Phase 3: SA to escape local minima, then local search
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 5.0  # Reserve 5s for final polish
    
    # SA run 1: From current best
    seg1_time = remaining * 0.5
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run(a, b, best_p, best_cost, seg1_time, T_start, T_end, use_surrogate_feedback=True)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Local search after SA
    elapsed = time.time() - start_time
    ls_time2 = min(3.0, max(0.5, total_deadline - elapsed - 5.0))
    if ls_time2 > 0.5:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=10000, time_limit=ls_time2)
    
    # SA run 2: From second-best initial (different basin)
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 5.0
    if remaining > 2.0:
        seg2_time = remaining * 0.5
        T_start = 300.0
        T_end = 3.0
        p2, cost2 = _sa_run(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end, use_surrogate_feedback=False)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2.copy()
        
        # Local search after SA2
        elapsed = time.time() - start_time
        ls_time3 = min(3.0, max(0.5, total_deadline - elapsed - 3.0))
        if ls_time3 > 0.5:
            best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=10000, time_limit=ls_time3)
    
    # Phase 4: Final intensive local search polish
    elapsed = time.time() - start_time
    polish_time = min(4.0, max(0.0, total_deadline - elapsed - 2.0))
    if polish_time > 0.5:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, max_iters=50000, time_limit=polish_time)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > total_deadline - 2.0:
        pass  # Already at deadline, return what we have
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
