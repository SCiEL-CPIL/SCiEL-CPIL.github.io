"""QAP tai256c solver: iterated local search with vectorized 2-opt and Hungarian initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(a)
    total = 0
    for i in range(n):
        total += np.dot(a[i, :], b[p[i], p])
    return int(total)


def _hungarian_init(a, b):
    """Use Hungarian algorithm on a reduced cost matrix for initial assignment."""
    n = len(a)
    # Create a cost matrix where cost[i][j] represents assigning facility i to location j
    # We use row sums of a and column sums of b as a proxy
    # Actually, for QAP, a common heuristic is to use the rank-matching or
    # solve a linear assignment based on aggregated costs
    
    # Compute for each facility i and location j: 
    #   c[i,j] = sum_k a[i,k] * b[j, k] + sum_k a[k,i] * b[k, j]
    # This captures the "interaction" cost if facility i is at location j
    # But this is O(n^3) to compute. For n=256, that's ~16M ops, feasible.
    
    # Actually, let's use a simpler approach:
    # c[i,j] = sum_k a[i,k] * b[j, k]  (out-flow contribution)
    # This is O(n^3) = n * n * n = 256^3 = 16.7M, which is fine.
    
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # c[i, :] = a[i, :] @ b  -- this gives sum_k a[i,k]*b[j,k] for each j
        c[i, :] = a[i, :] @ b
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(c)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _compute_out_matrix(a, b, p):
    """
    Compute out[i][l] = sum_j a[i,j] * b[l, p[j]] for all i, l.
    This is the contribution of facility i if placed at location l,
    considering the current assignment p for all other facilities.
    """
    n = len(a)
    # out[i, l] = sum_j a[i,j] * b[l, p[j]]
    # = (a[i, :] * b[:, p[j]].T) ... no
    # For fixed l: out[:, l] = a @ b[l, p]  where b[l, p] is a vector of length n
    # Actually: out[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Let b_l_p = b[l, p] which is a vector of length n (b[l, p[0]], b[l, p[1]], ..., b[l, p[n-1]])
    # Then out[:, l] = a @ b_l_p
    
    out = np.zeros((n, n), dtype=np.int64)
    for l in range(n):
        # b[l, p] gives b[l, p[0]], b[l, p[1]], ..., b[l, p[n-1]]
        b_lp = b[l, p]  # shape (n,)
        out[:, l] = a @ b_lp
    
    return out


def _local_search(a, b, p, deadline, max_iters=100000):
    """
    2-opt local search with efficient delta computation.
    
    The key insight: precompute out[i][l] = sum_j a[i,j] * b[l, p[j]].
    When we swap p[i] and p[j], we need to update out for all facilities.
    
    Actually, let's use a different approach. For a swap of positions i and j:
    - p'[i] = p[j], p'[j] = p[i], p'[k] = p[k] for k != i,j
    
    The cost change involves:
    1. Terms where facility i or j is involved
    2. Terms where location p[i] or p[j] is involved
    
    Let's compute the delta directly:
    delta = sum_{k != i,j} [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) 
           + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
           + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
           + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
          + a[i,j]*(b[p[j],p[j]] - b[p[i],p[i]])
          + a[j,i]*(b[p[i],p[i]] - b[p[j],p[j]])
    
    This can be computed in O(n) per swap.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    # Precompute some useful arrays
    # a_row[i] = a[i, :]
    # a_col[i] = a[:, i]
    
    # For fast delta computation, precompute:
    # For each position i and location l, we need b[l, p[k]] for various k
    # This is expensive. Instead, let's compute delta on the fly but vectorized.
    
    # Actually, let's precompute:
    # b_p[l, k] = b[l, p[k]] for all l, k
    # This is just b[:, p] which is O(n^2)
    b_p = b[:, p]  # shape (n, n), b_p[l, k] = b[l, p[k]]
    
    # Also precompute a rows and columns
    # a[i, k] and a[k, i]
    
    # For delta computation when swapping positions i and j:
    # Let li = p[i], lj = p[j]
    # delta = sum_{k != i,j} [a[i,k]*(b_p[lj,k] - b_p[li,k]) 
    #        + a[k,i]*(b_p[k,lj] - b_p[k,li])
    #        + a[j,k]*(b_p[li,k] - b_p[lj,k])
    #        + a[k,j]*(b_p[k,li] - b_p[k,lj])]
    #       + a[i,j]*(b[lj,lj] - b[li,li])
    #       + a[j,i]*(b[li,li] - b[lj,lj])
    
    # Note: b_p[lj,k] = b[lj, p[k]], b_p[li,k] = b[li, p[k]]
    # And b_p[k,lj] = b[k, p[lj]] = b[k, p[j]] = b[k, lj] (since p[j] = lj)
    # Wait, p[j] = lj, so b_p[k, j] = b[k, p[j]] = b[k, lj]
    # But we need b[k, lj] which is b[k, p[j]] = b_p[k, j]
    # Similarly b[k, li] = b[k, p[i]] = b_p[k, i]
    
    # So:
    # b_p[lj,k] - b_p[li,k] = b[lj, p[k]] - b[li, p[k]]
    # b_p[k,lj] - b_p[k,li] = b[k, p[lj]] - b[k, p[li]] = b[k, lj] - b[k, li]
    # But p[lj] is not lj. p is a permutation of locations, so p maps positions to locations.
    # p[i] = li means position i has location li.
    # p[j] = lj means position j has location lj.
    # So b[k, lj] = b[k, p[j]] = b_p[k, j]
    # And b[k, li] = b[k, p[i]] = b_p[k, i]
    
    # So delta = sum_{k != i,j} [a[i,k]*(b_p[lj,k] - b_p[li,k]) 
    #        + a[k,i]*(b_p[k,j] - b_p[k,i])
    #        + a[j,k]*(b_p[li,k] - b_p[lj,k])
    #        + a[k,j]*(b_p[k,i] - b_p[k,j])]
    #       + a[i,j]*(b[lj,lj] - b[li,li])
    #       + a[j,i]*(b[li,li] - b[lj,lj])
    
    # Let's define:
    # d1[k] = a[i,k] - a[j,k]  (for k != i,j)
    # d2[k] = a[k,i] - a[k,j]  (for k != i,j)
    # d3[k] = b_p[lj,k] - b_p[li,k]  (for k != i,j)
    # d4[k] = b_p[k,j] - b_p[k,i]  (for k != i,j)
    
    # Then delta = sum_{k != i,j} [d1[k]*d3[k] + d2[k]*d4[k]] 
    #            + a[i,j]*(b[lj,lj] - b[li,li])
    #            + a[j,i]*(b[li,li] - b[lj,lj])
    
    # This is O(n) per swap!
    
    # Precompute b_p = b[:, p]
    # We'll update b_p when we accept a swap
    
    for iteration in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try a random pair
        i, j = random.sample(range(n), 2)
        if i == j:
            continue
        
        li = p[i]
        lj = p[j]
        
        # Compute delta
        # d1[k] = a[i,k] - a[j,k] for k != i,j
        # d2[k] = a[k,i] - a[k,j] for k != i,j
        # d3[k] = b_p[lj,k] - b_p[li,k] for k != i,j
        # d4[k] = b_p[k,j] - b_p[k,i] for k != i,j
        
        # Vectorized computation
        d1 = a[i, :] - a[j, :]
        d2 = a[:, i] - a[:, j]
        d3 = b_p[lj, :] - b_p[li, :]
        d4 = b_p[:, j] - b_p[:, i]
        
        # Zero out the i and j positions
        d1[i] = 0
        d1[j] = 0
        d2[i] = 0
        d2[j] = 0
        d3[i] = 0
        d3[j] = 0
        d4[i] = 0
        d4[j] = 0
        
        delta_sum = np.dot(d1, d3) + np.dot(d2, d4)
        
        # Diagonal terms
        delta_diag = a[i, j] * (b[lj, lj] - b[li, li]) + a[j, i] * (b[li, li] - b[lj, lj])
        
        delta = delta_sum + delta_diag
        
        if delta < 0:
            # Accept the swap
            p[i], p[j] = lj, li
            current_cost += delta
            
            # Update b_p: b_p[l, k] = b[l, p[k]]
            # After swap, p[i] = lj, p[j] = li
            # So b_p[:, i] = b[:, lj] and b_p[:, j] = b[:, li]
            b_p[:, i] = b[:, lj]
            b_p[:, j] = b[:, li]
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, int(best_cost)


def _perturb(p, n, num_swaps):
    """Perturb a permutation by swapping random pairs."""
    p = p.copy()
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Initial solution using Hungarian
    p = _hungarian_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local search
    best_p, best_cost = _local_search(a, b, p, deadline, max_iters=100000)
    
    # Iterated local search with perturbations
    for restart in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Perturb the best solution
        num_swaps = max(2, n // 8)
        p = _perturb(best_p, n, num_swaps)
        
        current_cost = _qap_cost(a, b, p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Local search from this point
        p2, cost2 = _local_search(a, b, p, deadline, max_iters=50000)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2
    
    # Final polish: one more local search pass
    if time.time() - start_time < deadline - 5:
        best_p, best_cost = _local_search(a, b, best_p, deadline, max_iters=100000)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
