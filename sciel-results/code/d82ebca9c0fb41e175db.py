"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _calc_cost(a, b, p):
    """Calculate the QAP cost for a given permutation."""
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, time_limit, start_time, max_passes=50):
    """
    Local search with 2-opt moves for QAP.
    Uses vectorized delta computation for efficiency.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute A_sym for delta calculations
    # Delta for swapping positions i and j in permutation p:
    # Let A = a, B = b
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap p[i] and p[j], the change in cost is:
    # delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #       + sum_k a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + sum_k a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    #       - a[i,i]*(b[p[i],p[i]] - b[p[j],p[j]]) - a[j,j]*(b[p[j],p[j]] - b[p[i],p[i]])
    #       - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]]) - a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
    
    # More cleanly:
    # delta = sum_k a[i,k]*(D[j,k] - D[i,k]) + sum_k a[k,i]*(D[k,j] - D[k,i])
    #       + sum_k a[j,k]*(D[i,k] - D[j,k]) + sum_k a[k,j]*(D[k,i] - D[k,j])
    #       + (a[i,i] - a[j,j])*(D[i,i] - D[j,j]) + (a[i,j] - a[j,i])*(D[i,j] - D[j,i])
    
    # where D = b[np.ix_(p,p)]
    
    A_sym = a + a.T  # Not directly useful but keep for reference
    
    passes = 0
    while time.time() - start_time < time_limit and passes < max_passes:
        improved = False
        passes += 1
        
        # Try all 2-opt moves in random order
        order = np.random.permutation(n)
        
        for idx_i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            i = order[idx_i]
            js = np.array([order[idx_j] for idx_j in range(idx_i + 1, n)])
            
            if len(js) == 0:
                continue
            
            # Compute delta for swapping i with each j in js
            # delta(i,j) = sum_k a[i,k]*(D[j,k] - D[i,k]) + sum_k a[k,i]*(D[k,j] - D[k,i])
            #            + sum_k a[j,k]*(D[i,k] - D[j,k]) + sum_k a[k,j]*(D[k,i] - D[k,j])
            #            + (a[i,i] - a[j,j])*(D[i,i] - D[j,j]) + (a[i,j] - a[j,i])*(D[i,j] - D[j,i])
            
            # Vectorize over j:
            # term1: a[i,:] @ (D[js,:] - D[i,:]) -> shape (len(js),)
            # term2: a[:,i] @ (D[:,js] - D[:,i]) -> shape (len(js),)
            # term3: for each j in js: a[j,:] @ (D[i,:] - D[j,:]) -> need row-wise dot
            # term4: for each j in js: a[:,j] @ (D[:,i] - D[:,j]) -> need col-wise dot
            
            # term1 = a[i, :] @ (D[js, :] - D[i, :])
            term1 = a[i, :] @ (D[js, :] - D[i, :])
            
            # term2 = a[:, i] @ (D[:, js] - D[:, i])
            term2 = a[:, i] @ (D[:, js] - D[:, i])
            
            # term3: for each j in js, a[j, :] dot (D[i, :] - D[j, :])
            # = (a[js, :] * (D[i, :] - D[js, :])).sum(axis=1)
            term3 = (a[js, :] * (D[i, :] - D[js, :])).sum(axis=1)
            
            # term4: for each j in js, a[:, j] dot (D[:, i] - D[:, j])
            # = (a[:, js] * (D[:, i] - D[:, js])).sum(axis=0)
            term4 = (a[:, js] * (D[:, i] - D[:, js])).sum(axis=0)
            
            # term5 = (a[i,i] - a[js,js]) * (D[i,i] - D[js,js])
            term5 = (a[i, i] - a[js, js]) * (D[i, i] - D[js, js])
            
            # term6 = (a[i,js] - a[js,i]) * (D[i,js] - D[js,i])
            term6 = (a[i, js] - a[js, i]) * (D[i, js] - D[js, i])
            
            deltas = term1 + term2 + term3 + term4 + term5 + term6
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Swap i and j in p
                p[i], p[j] = p[j], p[i]
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b, n):
    """
    Initialize using Hungarian algorithm on a derived cost matrix.
    """
    # Create a cost matrix that approximates the QAP objective
    # Use the product of row/col sums as a heuristic
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    
    # Cost matrix: c[i,j] = -a_row_sums[i] * b_col_sums[j]
    # We want to minimize this, so use negative
    cost_matrix = -np.outer(a_row_sums, b_col_sums)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.zeros(n, dtype=int)
    for i, j in zip(row_ind, col_ind):
        p[i] = j
    
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer for safety
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Hungarian-based initialization
    p_hung = _hungarian_init(a, b, n)
    p_ls, cost_ls = _local_search(a, b, p_hung, 25, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # Phase 2: Multiple random restarts with local search
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    
    if remaining > 15:
        num_restarts = min(4, int(remaining / 12))
        for r in range(num_restarts):
            if time.time() - start_time > time_budget - 10:
                break
            p_init = np.random.permutation(n)
            p_ls, cost_ls = _local_search(a, b, p_init, 12, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Phase 3: Iterated Local Search (ILS)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    
    if remaining > 10:
        consecutive_no_improve = 0
        base_perturb = 3
        
        while time.time() - start_time < time_budget - 3:
            if time.time() - start_time > time_budget - 3:
                break
            
            if consecutive_no_improve > 8:
                perturb_k = min(12, base_perturb + consecutive_no_improve // 2)
            else:
                perturb_k = base_perturb
            
            p_pert = _perturb(best_p, perturb_k)
            
            duration = (time_budget - 3) - (time.time() - start_time)
            if duration < 1:
                break
            
            p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
            
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
    
    # Final verification
    if best_p is None:
        best_p = np.random.permutation(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
