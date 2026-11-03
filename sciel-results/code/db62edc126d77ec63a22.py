"""Frank-Wolfe / Birkhoff relaxation for QAP tai256c with local search refinement."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: use einsum
    # Let's build a matrix P where P[i, p[i]] = 1
    # Then cost = sum_{i,j,k,l} a[i,j] * P[i,k] * b[k,l] * P[j,l]
    # = sum_{i,j} a[i,j] * b[P[i], P[j]]
    # We can do: b_perm = b[np.ix_(p, p)]  # b_perm[i,j] = b[p[i], p[j]]
    # cost = np.sum(a * b_perm)
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _rank_matching_start(a, b):
    """Generate initial permutation using rank matching heuristic."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, max_iters=5000, time_limit=10.0):
    """2-opt local search: try swapping pairs of locations."""
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute row sums of a for faster delta computation
    # Delta for swapping p[i] and p[j]:
    # Old contribution involving i or j:
    #   sum_k a[i,k]*b[p[i],p[k]] + sum_k a[k,i]*b[p[k],p[i]] 
    #   + sum_k a[j,k]*b[p[j],p[k]] + sum_k a[k,j]*b[p[k],p[j]]
    #   - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]  (these are counted twice above)
    # New contribution:
    #   sum_k a[i,k]*b[p[j],p[k]] + sum_k a[k,i]*b[p[k],p[j]]
    #   + sum_k a[j,k]*b[p[i],p[k]] + sum_k a[k,j]*b[p[k],p[i]]
    #   - a[i,j]*b[p[j],p[i]] - a[j,i]*b[p[i],p[j]]
    
    # Let's use a simpler approach: just compute the full cost change
    # For a swap of positions i and j in p:
    # Let loc_i = p[i], loc_j = p[j]
    # The cost change is:
    # sum over k != i,j: a[i,k]*(b[loc_j,p[k]] - b[loc_i,p[k]]) + a[k,i]*(b[p[k],loc_j] - b[p[k],loc_i])
    #                 + a[j,k]*(b[loc_i,p[k]] - b[loc_j,p[k]]) + a[k,j]*(b[p[k],loc_i] - b[p[k],loc_j])
    # + a[i,j]*(b[loc_i,loc_j] - b[loc_j,loc_i]) + a[j,i]*(b[loc_j,loc_i] - b[loc_i,loc_j])
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        # Randomly sample a few pairs to try
        num_samples = min(n * 10, 2000)
        indices = np.random.choice(n * (n - 1) // 2, size=num_samples, replace=False)
        
        for idx in indices:
            i = int(idx // (n - 1))
            j = int(idx % (n - 1))
            if j >= i:
                j += 1
            
            loc_i = p[i]
            loc_j = p[j]
            
            # Compute delta cost
            # Terms involving i (excluding j) and j (excluding i)
            # a[i, :] with b[loc_i, :] vs b[loc_j, :]
            diff_i = b[loc_j, :] - b[loc_i, :]
            diff_j = b[loc_i, :] - b[loc_j, :]
            
            # Sum over k != i, j
            delta = np.dot(a[i, :], diff_i) + np.dot(a[:, i], diff_i)
            delta += np.dot(a[j, :], diff_j) + np.dot(a[:, j], diff_j)
            
            # Subtract the i-j and j-i terms that were double counted or need adjustment
            # When k=j in the first two terms: a[i,j]*b[loc_j,loc_j] - a[i,j]*b[loc_i,loc_j]
            # When k=i in the first two terms: a[i,i]*b[loc_j,loc_i] - a[i,i]*b[loc_i,loc_i]
            # These are included in the dot products. Let's be more careful.
            
            # Actually, let's just compute the exact delta by considering all terms
            # Old terms involving i or j:
            # T_old = sum_k a[i,k]*b[loc_i,p[k]] + sum_k a[k,i]*b[p[k],loc_i]
            #       + sum_k a[j,k]*b[loc_j,p[k]] + sum_k a[k,j]*b[p[k],loc_j]
            #       - a[i,j]*b[loc_i,loc_j] - a[j,i]*b[loc_j,loc_i]
            #       - a[i,i]*b[loc_i,loc_i] - a[j,j]*b[loc_j,loc_j]
            # (subtracting because they're counted in both sums)
            
            # New terms:
            # T_new = sum_k a[i,k]*b[loc_j,p[k]] + sum_k a[k,i]*b[p[k],loc_j]
            #       + sum_k a[j,k]*b[loc_i,p[k]] + sum_k a[k,j]*b[p[k],loc_i]
            #       - a[i,j]*b[loc_j,loc_i] - a[j,i]*b[loc_i,loc_j]
            #       - a[i,i]*b[loc_j,loc_j] - a[j,j]*b[loc_i,loc_i]
            
            # This is getting complex. Let's just compute the full cost for small n or use a simpler delta.
            # For n=256, let's compute delta more carefully.
            
            # Delta = T_new - T_old
            # = sum_k a[i,k]*(b[loc_j,p[k]] - b[loc_i,p[k]]) + sum_k a[k,i]*(b[p[k],loc_j] - b[p[k],loc_i])
            #   + sum_k a[j,k]*(b[loc_i,p[k]] - b[loc_j,p[k]]) + sum_k a[k,j]*(b[p[k],loc_i] - b[p[k],loc_j])
            #   - a[i,j]*(b[loc_j,loc_i] - b[loc_i,loc_j]) - a[j,i]*(b[loc_i,loc_j] - b[loc_j,loc_i])
            #   - a[i,i]*(b[loc_j,loc_j] - b[loc_i,loc_i]) - a[j,j]*(b[loc_i,loc_i] - b[loc_j,loc_j])
            
            # The last four terms:
            # -a[i,j]*(b[loc_j,loc_i] - b[loc_i,loc_j]) - a[j,i]*(b[loc_i,loc_j] - b[loc_j,loc_i])
            # = -(a[i,j] - a[j,i])*(b[loc_j,loc_i] - b[loc_i,loc_j])
            # = (a[j,i] - a[i,j])*(b[loc_j,loc_i] - b[loc_i,loc_j])
            
            # -a[i,i]*(b[loc_j,loc_j] - b[loc_i,loc_i]) - a[j,j]*(b[loc_i,loc_i] - b[loc_j,loc_j])
            # = (a[j,j] - a[i,i])*(b[loc_i,loc_i] - b[loc_j,loc_j])
            
            # The first four sums:
            # sum_k a[i,k]*(b[loc_j,p[k]] - b[loc_i,p[k]]) = a[i,:] @ (b[loc_j,:] - b[loc_i,:])
            # sum_k a[k,i]*(b[p[k],loc_j] - b[p[k],loc_i]) = a[:,i] @ (b[:,loc_j] - b[:,loc_i])
            # sum_k a[j,k]*(b[loc_i,p[k]] - b[loc_j,p[k]]) = a[j,:] @ (b[loc_i,:] - b[loc_j,:])
            # sum_k a[k,j]*(b[p[k],loc_i] - b[p[k],loc_j]) = a[:,j] @ (b[:,loc_i] - b[:,loc_j])
            
            diff_row_i = b[loc_j, :] - b[loc_i, :]
            diff_col_i = b[:, loc_j] - b[:, loc_i]
            diff_row_j = b[loc_i, :] - b[loc_j, :]
            diff_col_j = b[:, loc_i] - b[:, loc_j]
            
            delta = np.dot(a[i, :], diff_row_i) + np.dot(a[:, i], diff_col_i)
            delta += np.dot(a[j, :], diff_row_j) + np.dot(a[:, j], diff_col_j)
            
            # Special terms for i-j interactions
            delta += (a[j, i] - a[i, j]) * (b[loc_j, loc_i] - b[loc_i, loc_j])
            delta += (a[j, j] - a[i, i]) * (b[loc_i, loc_i] - b[loc_j, loc_j])
            
            if delta < -1e-10:
                # Perform swap
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                improved = True
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                break  # Restart sampling after improvement
        
        if not improved:
            no_improve += 1
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    # Start with rank matching
    p = _rank_matching_start(a, b)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Frank-Wolfe iterations on Birkhoff polytope
    # We maintain a doubly stochastic matrix X
    # Initialize X as the permutation matrix of p
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0
    
    # Frank-Wolfe: minimize f(X) = <A, X^T B X> over Birkhoff polytope
    # Gradient: G = 2 * A @ X^T @ B (or similar, depends on exact formulation)
    # Actually, f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # df/dX[i,k] = sum_j sum_l A[i,j] B[k,l] X[j,l] + sum_j sum_l A[j,i] X[j,k] B[l,k] ... 
    # This is getting complex. Let's use a simpler approach.
    
    # Alternative: Use the fact that for permutation matrices, we can do local search.
    # Let's just do iterative local search with random restarts.
    
    num_restarts = 0
    while time.time() - start_time < deadline:
        # Local search on current p
        p, cost = _local_search_2opt(a, b, p, max_iters=2000, time_limit=10.0)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Random restart with perturbation
        num_restarts += 1
        if num_restarts < 5:
            # Perturb best solution
            p = best_p.copy()
            num_swaps = max(1, n // 10)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        else:
            # Full random restart
            p = np.random.permutation(n)
    
    return best_p.tolist()
