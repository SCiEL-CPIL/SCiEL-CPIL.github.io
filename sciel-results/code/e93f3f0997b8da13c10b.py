"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(a, b, p):
    """Exact integer cost for assignment p."""
    # p: list/array of length n, facility i -> location p[i]
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each pair (i,j), add a[i,j]*b[p[i],p[j]]
    # Efficient way: create permuted b matrix
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _sa_cost_delta(a, b, p, i, j):
    """
    Compute the change in total cost if we swap facilities i and j in assignment p.
    Currently: facility i is at location p[i], facility j is at location p[j].
    After swap: facility i is at location p[j], facility j is at location p[i].
    
    The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    Only terms involving i or j change.
    
    Before swap, terms involving i or j:
      sum_l a[i,l] * b[p[i], p[l]]  +  sum_l a[j,l] * b[p[j], p[l]]
      + sum_k a[k,i] * b[p[k], p[i]] + sum_k a[k,j] * b[p[k], p[j]]
      - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]  (to avoid double counting the (i,j) and (j,i) terms)
    
    Wait, let me be more careful. The full cost is:
      C = sum_{k=0}^{n-1} sum_{l=0}^{n-1} a[k,l] * b[p[k], p[l]]
    
    Terms that change when swapping p[i] and p[j]:
      For k=i: a[i,l] * b[p[i], p[l]] changes to a[i,l] * b[p[j], p[l]] for all l
      For k=j: a[j,l] * b[p[j], p[l]] changes to a[j,l] * b[p[i], p[l]] for all l
      For l=i: a[k,i] * b[p[k], p[i]] changes to a[k,i] * b[p[k], p[j]] for all k
      For l=j: a[k,j] * b[p[k], p[j]] changes to a[k,j] * b[p[k], p[i]] for all k
    
    But we need to be careful about double counting when (k,l) = (i,j) or (j,i).
    
    Let's compute delta directly:
    delta = C_after - C_before
    
    C_before terms involving i or j (as first or second index):
      T_before = sum_l a[i,l]*b[p[i],p[l]] + sum_l a[j,l]*b[p[j],p[l]] 
               + sum_k a[k,i]*b[p[k],p[i]] + sum_k a[k,j]*b[p[k],p[j]]
               - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]
    
    C_after terms:
      T_after = sum_l a[i,l]*b[p[j],p[l]] + sum_l a[j,l]*b[p[i],p[l]] 
              + sum_k a[k,i]*b[p[k],p[j]] + sum_k a[k,j]*b[p[k],p[i]]
              - a[i,j]*b[p[j],p[i]] - a[j,i]*b[p[i],p[j]]
    
    delta = T_after - T_before
    """
    pi = p[i]
    pj = p[j]
    
    # sum_l a[i,l]*b[pi, p[l]]
    s1_before = np.dot(a[i, :], b[pi, p])
    s1_after = np.dot(a[i, :], b[pj, p])
    
    # sum_l a[j,l]*b[pj, p[l]]
    s2_before = np.dot(a[j, :], b[pj, p])
    s2_after = np.dot(a[j, :], b[pi, p])
    
    # sum_k a[k,i]*b[p[k], pi]
    s3_before = np.dot(a[:, i], b[p, pi])
    s3_after = np.dot(a[:, i], b[p, pj])
    
    # sum_k a[k,j]*b[p[k], pj]
    s4_before = np.dot(a[:, j], b[p, pj])
    s4_after = np.dot(a[:, j], b[p, pi])
    
    # Correction for double-counted (i,j) and (j,i)
    corr_before = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    corr_after = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    
    delta = (s1_after + s2_after + s3_after + s4_after - corr_after) - \
            (s1_before + s2_before + s3_before + s4_before - corr_before)
    
    return int(delta)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = _total_cost(a, b, p)
    
    # Simulated annealing
    T = 1000.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute row and column sums for faster delta computation
    # Actually, let's use a different approach: maintain the cost incrementally
    
    # For SA, we need efficient swap evaluation.
    # Let's precompute some helper arrays.
    
    # b_rows: b[pi, :] for current p[i] = pi
    # We can compute delta more efficiently if we maintain:
    #   row_sums_b_at_p[i] = sum_l b[p[i], p[l]]  -- but this changes with swaps
    # Let's just use the dot product approach but optimize it.
    
    # Alternative: maintain the full cost and compute delta via:
    # delta = sum_l a[i,l]*(b[pj,p[l]] - b[pi,p[l]]) + sum_l a[j,l]*(b[pi,p[l]] - b[pj,p[l]])
    #        + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi]) + sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
    #        - (a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi])
    
    # Let's define:
    #   diff_ij = b[pj, p] - b[pi, p]  (vector of length n)
    #   diff_ji = b[pi, p] - b[pj, p] = -diff_ij
    #   diff_col_i = b[p, pj] - b[p, pi]  (vector of length n)
    #   diff_col_j = b[p, pi] - b[p, pj] = -diff_col_i
    
    # delta = a[i,:] @ diff_ij + a[j,:] @ diff_ji + a[:,i] @ diff_col_i + a[:,j] @ diff_col_j
    #        - (a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi])
    
    # This is O(n) per swap evaluation using dot products.
    
    current_cost = best_cost
    
    # Number of iterations
    # We'll run until deadline
    iter_count = 0
    while time.time() < deadline:
        # Pick two random facilities to swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        pi = p[i]
        pj = p[j]
        
        # Compute delta
        # diff_row = b[pj, p] - b[pi, p]
        diff_row = b[pj, p] - b[pi, p]
        diff_col = b[p, pj] - b[p, pi]
        
        delta = (np.dot(a[i, :], diff_row) + np.dot(a[j, :], -diff_row) +
                 np.dot(a[:, i], diff_col) + np.dot(a[:, j], -diff_col) -
                 (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj] -
                  a[i, j] * b[pi, pj] - a[j, i] * b[pj, pi]))
        
        delta = int(delta)
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            # Accept the move
            p[i], p[j] = pj, pi
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        iter_count += 1
        # Check time periodically
        if iter_count % 10000 == 0:
            if time.time() > deadline - 5.0:
                break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
