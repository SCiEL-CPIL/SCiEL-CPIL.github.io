import numpy as np
from scipy.optimize import linear_sum_assignment
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit

    # 1. Initial Solution via Linear Surrogate (Hungarian)
    # Approximate QAP cost sum_{i,j} A[i,j] B[p[i], p[j]]
    # By sum_{i} A_row_sum[i] * B_row_sum[p[i]]
    # This assumes high flow facilities should go to low distance locations (or vice versa, 
    # actually high flow * low distance is good, so we want to minimize product).
    # So we want to assign high A_row_sum to low B_row_sum.
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Cost matrix for linear assignment: C[i, j] = a_row_sums[i] * b_row_sums[j]
    # We want to minimize this sum.
    # linear_sum_assignment minimizes the sum of costs.
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Compute initial cost
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = trace(a @ b[p][:,p]) ? No.
    # b_pp = b[np.ix_(p, p)]
    # cost = np.sum(a * b_pp)
    b_pp = b[np.ix_(p, p)]
    current_cost = np.sum(a * b_pp)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Vectorized Simulated Annealing
    # We will perform swaps of locations u and v.
    # Delta cost calculation for swapping facilities at u and v:
    # Let i = p[u], j = p[v].
    # Old terms involving i or j:
    # sum_k (A[i,k] B[u, p[k]] + A[j,k] B[v, p[k]] + A[k,i] B[p[k], u] + A[k,j] B[p[k], v])
    # New terms (i at v, j at u):
    # sum_k (A[i,k] B[v, p[k]] + A[j,k] B[u, p[k]] + A[k,i] B[p[k], v] + A[k,j] B[p[k], u])
    # Note: terms where k=u or k=v are handled correctly by the sum if we just swap indices in the formula?
    # Let's check k=u.
    # Old: A[i,u]B[u,p[u]] + A[j,u]B[v,p[u]] + A[u,i]B[p[u],u] + A[u,j]B[p[u],v]
    #      Since p[u]=i, p[v]=j:
    #      A[i,u]B[u,i] + A[j,u]B[v,i] + A[u,i]B[i,u] + A[u,j]B[i,v]
    # New: A[i,u]B[v,p[u]] + A[j,u]B[u,p[u]] + A[u,i]B[p[u],v] + A[u,j]B[p[u],u]
    #      A[i,u]B[v,i] + A[j,u]B[u,i] + A[u,i]B[i,v] + A[u,j]B[i,u]
    # The vectorized sum approach:
    # old_cost_part = np.dot(a[i, :], b[u, p]) + np.dot(a[j, :], b[v, p]) + np.dot(a[:, i], b[p, u]) + np.dot(a[:, j], b[p, v])
    # new_cost_part = np.dot(a[i, :], b[v, p]) + np.dot(a[j, :], b[u, p]) + np.dot(a[:, i], b[p, v]) + np.dot(a[:, j], b[p, u])
    # delta = new_cost_part - old_cost_part
    
    # This is O(n) per step.
    
    T = 10000.0
    alpha = 0.995
    min_T = 1.0
    
    # Pre-compute views for speed?
    # a_rows = a
    # b_rows = b
    # b_cols = b.T
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick random distinct u, v
        u = np.random.randint(n)
        v = np.random.randint(n)
        while v == u:
            v = np.random.randint(n)
            
        i = p[u]
        j = p[v]
        
        # Calculate delta
        # We need b[u, p], b[v, p], b[p, u], b[p, v]
        # b[u, p] is row u of b, indexed by p.
        # b[p, u] is column u of b, indexed by p.
        
        # Optimization: 
        # row_u_b = b[u, p]
        # row_v_b = b[v, p]
        # col_u_b = b[p, u]
        # col_v_b = b[p, v]
        
        # old = a[i, :] @ row_u_b + a[j, :] @ row_v_b + a[:, i] @ col_u_b + a[:, j] @ col_v_b
        # new = a[i, :] @ row_v_b + a[j, :] @ row_u_b + a[:, i] @ col_v_b + a[:, j] @ col_u_b
        
        # delta = new - old
        
        # Let's try to vectorize this slightly more or just use np.dot
        # a[i, :] is a vector. b[u, p] is a vector.
        
        old_part1 = np.dot(a[i, :], b[u, p])
        old_part2 = np.dot(a[j, :], b[v, p])
        old_part3 = np.dot(a[:, i], b[p, u])
        old_part4 = np.dot(a[:, j], b[p, v])
        
        new_part1 = np.dot(a[i, :], b[v, p])
        new_part2 = np.dot(a[j, :], b[u, p])
        new_part3 = np.dot(a[:, i], b[p, v])
        new_part4 = np.dot(a[:, j], b[p, u])
        
        delta = (new_part1 + new_part2 + new_part3 + new_part4) - (old_part1 + old_part2 + old_part3 + old_part4)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept move
            p[u], p[v] = p[v], p[u]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= alpha

    return best_p.tolist()
