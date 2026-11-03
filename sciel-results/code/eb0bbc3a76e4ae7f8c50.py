"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _linear_relaxation_init(a, b):
    """
    Initialize using the linear relaxation of the QAP.
    
    The linear relaxation assigns each facility to a location minimizing
    sum_i sum_j a[i,j] * b[pi, pj] where pi is the assignment.
    
    We solve two assignment problems:
    1. Assign facilities to locations based on row sums of a weighted by b
    2. Use the result as a starting point
    
    A common approach: minimize sum_i sum_j a[i,j] * b[pi, j] for fixed j,
    which decomposes into an assignment problem with cost matrix
    C[i, k] = sum_j a[i, j] * b[k, j]
    """
    n = len(a)
    # Cost matrix for the linear relaxation:
    # C[i, k] = sum_j a[i, j] * b[k, j]
    # This is a[i,:] @ b[k,:]^T = (a @ b.T)[i, k]
    c = a @ b.T
    # Solve assignment: facility i -> location row of b
    # linear_sum_assignment returns row_ind, col_ind such that cost is min
    row_ind, col_ind = linear_sum_assignment(c)
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities i and j.
    
    Current: facility i at p[i], facility j at p[j]
    After swap: facility i at p[j], facility j at p[i]
    
    Delta = cost_after - cost_before
    Only terms involving i or j change.
    """
    pi = p[i]
    pj = p[j]
    
    # Terms involving i or j in the sum sum_{k,l} a[k,l] * b[p[k], p[l]]
    # When we swap p[i] and p[j], the affected terms are:
    # - a[i,j] * (b[pj,pi] - b[pi,pj])  -- wait, let me think more carefully
    
    # The cost is sum_{k,l} a[k,l] * b[p[k], p[l]]
    # After swapping positions of i and j:
    # p'[i] = pj, p'[j] = pi, p'[k] = p[k] for k != i,j
    #
    # Changed terms:
    # For k=i, l=j: a[i,j] * (b[pj,pi] - b[pi,pj])
    # For k=j, l=i: a[j,i] * (b[pi,pj] - b[pj,pi])
    # For k=i, l=k (k!=i,j): a[i,k] * (b[pj,p[k]] - b[pi,p[k]])
    # For k=j, l=k (k!=i,j): a[j,k] * (b[pi,p[k]] - b[pj,p[k]])
    # For k=k (k!=i,j), l=i: a[k,i] * (b[p[k],pj] - b[p[k],pi])
    # For k=k (k!=i,j), l=j: a[k,j] * (b[p[k],pi] - b[p[k],pj])
    
    # Let's compute this efficiently using vectorized ops
    # Row i of a: a[i, :] and we need b[pj, :] vs b[pi, :]
    # Row j of a: a[j, :]
    # Column i of a: a[:, i]
    # Column j of a: a[:, j]
    
    # delta = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #         + sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    #         + sum_{k!=i,j} a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    #         + sum_{k!=i,j} a[k,i]*(b[p[k],pj]-b[p[k],pi])
    #         + sum_{k!=i,j} a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # Let's define:
    # diff_row = b[pj, :] - b[pi, :]  (shape n)
    # diff_col = b[:, pj] - b[:, pi]  (shape n)
    
    diff_row = b[pj, :] - b[pi, :]  # b[pj,k] - b[pi,k] for all k
    diff_col = b[:, pj] - b[:, pi]  # b[k,pj] - b[k,pi] for all k
    
    # sum_k a[i,k] * diff_row[k] for all k (including k=i,j, but we'll adjust)
    s1 = np.dot(a[i, :], diff_row)
    # sum_k a[j,k] * (-diff_row[k]) for all k
    s2 = -np.dot(a[j, :], diff_row)
    # sum_k a[k,i] * diff_col[k] for all k
    s3 = np.dot(a[:, i], diff_col)
    # sum_k a[k,j] * (-diff_col[k]) for all k
    s4 = -np.dot(a[:, j], diff_col)
    
    # Now subtract the double-counted terms for k=i and k=j:
    # In s1, when k=i: a[i,i]*diff_row[i] should be excluded (p[i] changes)
    # Actually, this is getting complex. Let me use a simpler approach.
    
    # Simpler: just compute the delta by considering all terms that change.
    # The change in cost is:
    # delta = sum_{k,l} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    # where p' differs from p only at indices i and j.
    
    # Terms where k=i or l=i or k=j or l=j are affected.
    # Let's enumerate:
    # (i,i): a[i,i]*(b[pj,pj]-b[pi,pi])
    # (i,j): a[i,j]*(b[pj,pi]-b[pi,pj])
    # (j,i): a[j,i]*(b[pi,pj]-b[pj,pi])
    # (j,j): a[j,j]*(b[pi,pi]-b[pj,pj])
    # (i,k), k!=i,j: a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    # (j,k), k!=i,j: a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    # (k,i), k!=i,j: a[k,i]*(b[p[k],pj]-b[p[k],pi])
    # (k,j), k!=i,j: a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # Vectorized:
    # For rows i and j:
    # row_i_contrib = a[i,:] @ (b[pj,:] - b[pi,:])  but we need to handle the fact that
    # when k=j in this sum, p'[j]=pi not p[j]=pj... 
    
    # This is tricky. Let me just do it directly for the affected terms.
    
    # Affected pairs: (i,i), (i,j), (i,k for k!=i,j), (j,i), (j,j), (j,k for k!=i,j),
    #                 (k,i for k!=i,j), (k,j for k!=i,j)
    
    # Let's compute using the full expression but mask out i and j appropriately.
    
    # Actually, the cleanest vectorized way:
    # delta = a[i,:].dot(b[pj,:]-b[pi,:]) + a[j,:].dot(b[pi,:]-b[pj,:])
    #         + a[:,i].dot(b[:,pj]-b[:,pi]) + a[:,j].dot(b[:,pi]-b[:,pj])
    #         - a[i,i]*(b[pj,pj]-b[pi,pi])  # remove double count
    #         - a[i,j]*(b[pj,pi]-b[pi,pj])  # remove double count  
    #         - a[j,i]*(b[pi,pj]-b[pj,pi])  # remove double count
    #         - a[j,j]*(b[pi,pi]-b[pj,pj])  # remove double count
    #         - a[i,i]*(b[pj,pj]-b[pi,pi])  # from column term
    #         - a[i,j]*(b[pj,pi]-b[pi,pj])  # from column term
    #         - a[j,i]*(b[pi,pj]-b[pj,pi])  # from column term
    #         - a[j,j]*(b[pi,pi]-b[pj,pj])  # from column term
    # 
    # Wait, this is getting messy. Let me just compute it directly.
    
    # Direct computation:
    # The change comes from terms where at least one index is i or j.
    
    # Let's use the formula:
    # delta = sum_{k} a[i,k]*(b[pj,p'[k]] - b[pi,p[k]]) 
    #       + sum_{k} a[j,k]*(b[pi,p'[k]] - b[pj,p[k]])
    #       + sum_{k} a[k,i]*(b[p'[k],pj] - b[p[k],pi])
    #       + sum_{k} a[k,j]*(b[p'[k],pi] - b[p[k],pj])
    # where p'[i]=pj, p'[j]=pi, p'[k]=p[k] for k!=i,j
    #
    # But we're double counting the (i,i), (i,j), (j,i), (j,j) terms.
    # Each of these appears in two of the four sums above.
    # So we need to subtract them once.
    
    # Let me just compute it as:
    # Create p_prime
    # delta = sum_{k,l} a[k,l]*(b[p'[k],p'[l]] - b[p[k],p[l]])
    # Only compute for affected (k,l) pairs.
    
    # For efficiency with n=150, let's just do it with a few vectorized ops.
    
    p_prime = p.copy()
    p_prime[i] = pj
    p_prime[j] = pi
    
    # The difference is nonzero only when k in {i,j} or l in {i,j}
    # So we compute:
    # delta = sum_{k in {i,j}} sum_l a[k,l]*(b[p'[k],p'[l]] - b[p[k],p[l]])
    #       + sum_{k not in {i,j}} sum_{l in {i,j}} a[k,l]*(b[p'[k],p'[l]] - b[p[k],p[l]])
    
    # First part: rows i and j
    row_diff_i = b[p_prime[i], p_prime] - b[p[i], p]  # b[pj, p'] - b[pi, p]
    row_diff_j = b[p_prime[j], p_prime] - b[p[j], p]  # b[pi, p'] - b[pj, p]
    
    # Second part: columns i and j
    # For k not in {i,j}: b[p'[k], p'[i]] - b[p[k], p[i]] = b[p[k], pj] - b[p[k], pi]
    #                    b[p'[k], p'[j]] - b[p[k], p[j]] = b[p[k], pi] - b[p[k], pj]
    
    # This is still complex. For n=150, let's just compute the full cost difference
    # using a smarter approach.
    
    # Actually for n=150, a direct swap delta computation using the formula:
    # delta = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #         + sum_{k not in {i,j}} [a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) 
    #                                 + a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    #                                 + a[k,i]*(b[p[k],pj]-b[p[k],pi])
    #                                 + a[k,j]*(b[p[k],pi]-b[p[k],pj])]
    
    # Let me compute this:
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff_row = b[pj, :] - b[pi, :]  # shape (n,)
    diff_col = b[:, pj] - b[:, pi]  # shape (n,)
    
    # For k not in {i,j}:
    # a[i,k]*diff_row[k] + a[j,k]*(-diff_row[k]) + a[k,i]*diff_col[k] + a[k,j]*(-diff_col[k])
    # = (a[i,k] - a[j,k]) * diff_row[k] + (a[k,i] - a[k,j]) * diff_col[k]
    
    s = np.sum((a[i, mask] - a[j, mask]) * diff_row[mask] + 
               (a[mask, i] - a[mask, j]) * diff_col[mask])
    
    # Plus the (i,j) and (j,i) terms:
    s += a[i, j] * (b[pj, pi] - b[pi, pj])
    s += a[j, i] * (b[pi, pj] - b[pj, pi])
    
    # Note: the (i,i) and (j,j) terms:
    # When k=i in the sum above, we excluded it. But in the original sum, 
    # a[i,i]*b[p'[i],p'[i]] - a[i,i]*b[p[i],p[i]] = a[i,i]*(b[pj,pj]-b[pi,pi])
    # This term is NOT captured by the mask sum. Similarly for (j,j).
    # Also, the cross terms (i,j) and (j,i) are partially captured.
    
    # Let me verify: in the mask sum, for k not in {i,j}:
    # We have a[i,k]*diff_row[k] which is a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    # This is correct for the (i,k) term.
    # We have a[k,i]*diff_col[k] which is a[k,i]*(b[p[k],pj]-b[p[k],pi])
    # This is correct for the (k,i) term.
    # But we're missing:
    # (i,i): a[i,i]*(b[pj,pj]-b[pi,pi])
    # (j,j): a[j,j]*(b[pi,pi]-b[pj,pj])
    # (i,j): a[i,j]*(b[pj,pi]-b[pi,pj]) -- partially in the formula
    # (j,i): a[j,i]*(b[pi,pj]-b[pj,pi]) -- partially in the formula
    
    # Wait, I think my formula is correct if I include the (i,j) and (j,i) terms separately
    # and the (i,i) and (j,j) terms separately.
    
    s += a[i, i] * (b[pj, pj] - b[pi, pi])
    s += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    return s


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Step 1: Linear relaxation initialization
    p = _linear_relaxation_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Step 2: 2-opt local search with swap deltas
    # Use a randomized 2-opt: try all pairs, accept improving moves
    # To speed up, use a greedy approach: repeatedly find the best improving swap
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_pair = None
        
        # Sample a subset of pairs for efficiency if n is large
        # For n=150, 150*149/2 = 11175 pairs, which is manageable
        # But computing delta for each is O(n), so total O(n^2) per pass = ~2.25M ops
        # That's fast enough.
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        
        if best_pair is not None and best_delta < 0:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Step 3: If time remains, try random restarts with perturbation
    if time.time() < deadline - 20.0:
        # Try a few perturbed restarts
        for restart in range(5):
            if time.time() >= deadline - 10.0:
                break
            
            # Perturb: do a random number of random swaps
            p2 = best_p.copy()
            n_swaps = n // 10
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p2[i], p2[j] = p2[j], p2[i]
            
            cost2 = _qap_cost(a, b, p2)
            
            # Local search from perturbed solution
            improved2 = True
            while improved2 and time.time() < deadline - 5.0:
                improved2 = False
                best_delta2 = 0
                best_pair2 = None
                for i in range(n):
                    for j in range(i + 1, n):
                        delta2 = _compute_swap_delta(a, b, p2, i, j)
                        if delta2 < best_delta2:
                            best_delta2 = delta2
                            best_pair2 = (i, j)
                if best_pair2 is not None and best_delta2 < 0:
                    i, j = best_pair2
                    p2[i], p2[j] = p2[j], p2[i]
                    cost2 += best_delta2
                    improved2 = True
                    if cost2 < best_cost:
                        best_cost = cost2
                        best_p = p2.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
