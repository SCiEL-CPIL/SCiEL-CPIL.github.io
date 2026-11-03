"""Frank-Wolfe QAP solver on Birkhoff polytope with Hungarian projection and vectorized 2-opt."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search_2opt(a, b, p, max_iters=100, time_budget=10.0):
    """Vectorized 2-opt local search."""
    p = np.array(p, dtype=np.int64)
    n = len(p)
    best_cost = _qap_cost(a, b, p)
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute delta for all pairs (i,j) with i<j
        # delta[i,j] = change in cost when swapping p[i] and p[j]
        # Using the formula:
        # Let pi = p[i], pj = p[j]
        # delta = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
        #       + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        #       + sum_{k!=i,j} [a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #                       + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])]
        
        # Vectorized approach:
        # For each pair (i,j), the "bulk" term (k not in {i,j}) can be computed as:
        # sum_k a[i,k]*b[pj,p[k]] - sum_k a[i,k]*b[pi,p[k]] 
        #   - a[i,i]*b[pj,pi] - a[i,j]*b[pj,pj]  (subtract k=i,j terms)
        # + similar for other terms
        # This is getting complex. Let me use a simpler but still vectorized approach.
        
        # Alternative: compute the full cost delta using matrix operations
        # For swap of i,j: 
        # new_b_matrix has rows/cols i,j swapped in terms of p
        # delta = sum_{k,l} a[k,l] * (b[p_new[k],p_new[l]] - b[p_old[k],p_old[l]])
        # where p_new differs from p_old only at positions i and j
        
        # This is equivalent to:
        # Let S = {i,j}
        # delta = sum_{k in S, l in S} a[k,l]*(b[p_new[k],p_new[l]] - b[p_old[k],p_old[l]])
        #       + sum_{k in S, l not in S} a[k,l]*(b[p_new[k],p_old[l]] - b[p_old[k],p_old[l]])
        #       + sum_{k not in S, l in S} a[k,l]*(b[p_old[k],p_new[l]] - b[p_old[k],p_old[l]])
        
        # For k=i, l=j: a[i,j]*(b[pj,pi] - b[pi,pj])
        # For k=j, l=i: a[j,i]*(b[pi,pj] - b[pj,pi])
        # For k=i, l=i: a[i,i]*(b[pj,pj] - b[pi,pi])
        # For k=j, l=j: a[j,j]*(b[pi,pi] - b[pj,pj])
        # For k=i, l not in {i,j}: a[i,l]*(b[pj,p[l]] - b[pi,p[l]])
        # For k=j, l not in {i,j}: a[j,l]*(b[pi,p[l]] - b[pj,p[l]])
        # For k not in {i,j}, l=i: a[k,i]*(b[p[k],pj] - b[p[k],pi])
        # For k not in {i,j}, l=j: a[k,j]*(b[p[k],pi] - b[p[k],pj])
        
        # Let's vectorize over all pairs at once using broadcasting
        # This will be an n x n matrix of deltas
        
        pi = p  # p[i] for each i
        # For all i,j pairs:
        # term_ii = a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]])
        # term_jj = a[j,j]*(b[p[i],p[i]] - b[p[j],p[j]])
        # term_ij = a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
        # term_ji = a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
        # bulk_i = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) - a[i,i]*(b[p[j],p[i]]-b[p[i],p[i]]) - a[i,j]*(b[p[j],p[j]]-b[p[i],p[j]])
        # bulk_j = sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) - a[j,j]*(b[p[i],p[j]]-b[p[j],p[j]]) - a[j,i]*(b[p[i],p[i]]-b[p[j],p[i]])
        # bulk_ki = sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) - a[i,i]*(b[p[i],p[j]]-b[p[i],p[i]]) - a[j,i]*(b[p[j],p[j]]-b[p[j],p[i]])
        # bulk_kj = sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]]) - a[i,j]*(b[p[i],p[i]]-b[p[i],p[j]]) - a[j,j]*(b[p[j],p[i]]-b[p[j],p[j]])
        
        # This is getting very complex. Let me just do a simple loop-based 2-opt but limit iterations.
        
        improved = False
        for i in range(n):
            if time.time() - start > time_budget:
                break
            for j in range(i + 1, n):
                pi_val, pj_val = p[i], p[j]
                
                # Compute delta
                delta = 0.0
                # Self terms
                delta += a[i, i] * (b[pj_val, pj_val] - b[pi_val, pi_val])
                delta += a[j, j] * (b[pi_val, pi_val] - b[pj_val, pj_val])
                # Cross i,j
                delta += a[i, j] * (b[pj_val, pi_val] - b[pi_val, pj_val])
                delta += a[j, i] * (b[pi_val, pj_val] - b[pj_val, pi_val])
                
                # Bulk terms: sum over k not in {i,j}
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                pk = p[mask]
                
                delta += np.sum(a[i, mask] * (b[pj_val, pk] - b[pi_val, pk]))
                delta += np.sum(a[mask, i] * (b[pk, pj_val] - b[pk, pi_val]))
                delta += np.sum(a[j, mask] * (b[pi_val, pk] - b[pj_val, pk]))
                delta += np.sum(a[mask, j] * (b[pk, pi_val] - b[pk, pj_val]))
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    best_cost += int(round(delta))
                    improved = True
        
        if not improved:
            break
    
    return p, best_cost

def _frank_wolfe(a, b, x0, max_iters=150, time_budget=30.0):
    """Frank-Wolfe on Birkhoff polytope."""
    n = len(a)
    X = x0.copy().astype(np.float64)
    
    # Get initial permutation
    row_ind, col_ind = linear_sum_assignment(-X)
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    prev_cost = np.inf
    start_time = time.time()
    
    for t in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Gradient of f(X) = tr(A X B X^T)
        # = A @ X @ B + A.T @ X @ B.T
        grad = a @ X @ b + a.T @ X @ b.T
        grad = (grad + grad.T) / 2.0
        
        # Solve linear assignment on grad
        row_ind, col_ind = linear_sum_assignment(grad)
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        # FW step size
        step = 2.0 / (t + 2)
        
        # Update
        X = (1 - step) * X + step * Y
        
        # Get permutation from Y
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # Convergence check
        if abs(prev_cost - cost) < 1e-10 * max(1, abs(cost)):
            break
        prev_cost = cost
    
    return best_p, best_cost

def _greedy_initial(a, b):
    """Greedy rank-matching initial solution."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = len(a)
    
    # Get initial solution from greedy + 2-opt
    p0 = _greedy_initial(a, b)
    p0, cost0 = _local_search_2opt(a, b, p0, max_iters=50, time_budget=15.0)
    
    best_p = p0.copy()
    best_cost = cost0
    
    # Try Frank-Wolfe from different starting points
    rng = np.random.RandomState(42)
    num_restarts = 0
    
    while time.time() < deadline - 20.0:
        # Create starting point
        if num_restarts == 0:
            X0 = np.zeros((n, n), dtype=np.float64)
            X0[np.arange(n), p0] = 1.0
        else:
            # Random permutation with smoothing
            p_rand = rng.permutation(n)
            X0 = np.zeros((n, n), dtype=np.float64)
            X0[np.arange(n), p_rand] = 1.0
            # Smooth: mix with uniform
            X0 = 0.5 * X0 + 0.5 / n
            # This is doubly stochastic
        
        # Run Frank-Wolfe
        remaining_fw = min(30.0, deadline - time.time() - 15.0)
        if remaining_fw < 5.0:
            break
        p_fw, cost_fw = _frank_wolfe(a, b, X0, max_iters=100, time_budget=remaining_fw)
        
        # Polish with 2-opt
        remaining_2opt = min(10.0, deadline - time.time() - 5.0)
        if remaining_2opt < 2.0:
            p_polished = p_fw
            cost_polished = cost_fw
        else:
            p_polished, cost_polished = _local_search_2opt(a, b, p_fw, max_iters=30, time_budget=remaining_2opt)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
        
        num_restarts += 1
    
    # Final polish if time allows
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 2.0:
            best_p, best_cost = _local_search_2opt(a, b, best_p, max_iters=50, time_budget=remaining)
    
    return best_p.tolist()
