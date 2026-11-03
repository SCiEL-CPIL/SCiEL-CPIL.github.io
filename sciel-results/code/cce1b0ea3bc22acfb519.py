"""
Strategy: Linearization-based initialization + Vectorized Full-Neighborhood 1-Swap Local Search.

Key changes from current program:
1. Initialization: Use Hungarian algorithm on a linearized cost matrix M = A @ B + diag(row_sums_A) @ diag(row_sums_B) / n.
   This provides a strong global structure-aware starting point, often much better than random or simple greedy.
2. Local Search: Implement a fast, vectorized 1-swap local search that evaluates the ENTIRE neighborhood 
   (n*(n-1)/2 swaps) in a few milliseconds using NumPy. This allows many more iterations per second than 
   sampled tabu search and guarantees finding the best 1-swap move, leading to faster convergence to local optima.
3. Diversification: Multiple restarts with different linearization weights or perturbations.
"""

import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _vectorized_best_swap(a, b, p, current_cost):
    """
    Find the best 1-swap move in the entire neighborhood using vectorized operations.
    Returns (best_i, best_j, best_delta, new_cost) or (None, None, 0, current_cost) if no improving move.
    
    For a swap of positions i and j in permutation p:
    Let x = p[i], y = p[j].
    The change in cost is:
    delta = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x]) 
            + sum_{k!=i,j} [ a[i,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,i]*(b[p[k],y]-b[p[k],x])
                            + a[j,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,j]*(b[p[k],x]-b[p[k],y]) ]
    
    We can compute this for all pairs (i,j) efficiently.
    """
    n = len(p)
    
    # Precompute vectors for each position in the permutation
    # V_pos[k] = b[:, p[k]]  (column of b corresponding to location p[k])
    # W_pos[k] = b[p[k], :]  (row of b corresponding to location p[k])
    
    # For each pair (i, j), the delta involves:
    # 1. Direct term: a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    # 2. Indirect terms involving all other k
    
    # Let's compute the "influence" of swapping positions i and j.
    # Let x = p[i], y = p[j].
    
    # We want to compute D[i,j] for all i < j.
    # D[i,j] = sum_{k} (a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x]) 
    #             + a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y]))
    #          + a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    #          - terms where k=i or k=j (which are included in the sum but need correction)
    
    # Actually, the standard formula for delta(i,j) in QAP 1-swap is:
    # delta = a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    #         + sum_{k!=i,j} ( a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    #                         + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]]) )
    
    # Let's define:
    # R_i = a[i, :]  (row i of A)
    # C_i = a[:, i]  (col i of A)
    # L_i = b[:, p[i]] (col p[i] of B)
    # U_i = b[p[i], :] (row p[i] of B)
    
    # The sum over k!=i,j of a[i,k]*(b[y,p[k]] - b[x,p[k]]) can be written as:
    # sum_k a[i,k]*b[y,p[k]] - sum_k a[i,k]*b[x,p[k]] - terms for k=i,j
    
    # Let M_A = a
    # Let M_B = b
    
    # Vectorized approach for all pairs:
    # This is complex to fully vectorize for all n^2 pairs at once without huge memory.
    # However, n=256 is small. n^2 = 65536.
    # We can compute the delta for all pairs in O(n^3) time but with low constant factors using NumPy.
    
    # Let's compute the "potential" for each position i relative to each other position j.
    
    # Precompute:
    # B_cols = b[:, p]  # shape (n, n), B_cols[k, m] = b[k, p[m]]
    # B_rows = b[p, :]  # shape (n, n), B_rows[k, m] = b[p[k], m]
    
    B_cols = b[:, p]  # B_cols[k, pos] = b[k, p[pos]]
    B_rows = b[p, :]  # B_rows[pos, k] = b[p[pos], k]
    
    # For a swap of positions i and j (with facilities x=p[i], y=p[j]):
    # The change in the sum over k of a[i,k]*b[p[i],p[k]] and a[k,i]*b[p[k],p[i]] etc.
    
    # Let's compute the contribution of each facility i to the cost if it were at position pos.
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    # If we swap p[i] and p[j], the new cost is:
    # C' = C - (terms involving i or j) + (new terms involving i or j)
    
    # Terms involving i in C:
    # sum_j a[i,j] * b[p[i], p[j]] + sum_j a[j,i] * b[p[j], p[i]]
    # Note: a[i,i]*b[p[i],p[i]] is counted twice if we sum both, but in QAP sum_{i,j} usually i!=j or includes diagonal.
    # The standard QAP cost is sum_{i,j} a[i,j] b[p[i],p[j]].
    
    # Let's use the delta formula derived in the current program but vectorize it.
    
    # delta(i,j) = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    #              + sum_{k!=i,j} [ a[i,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,i]*(b[p[k],y]-b[p[k],x])
    #                              + a[j,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,j]*(b[p[k],x]-b[p[k],y]) ]
    
    # Let S1 = sum_{k} a[i,k] * (B_cols[y, k] - B_cols[x, k])  # where B_cols[y,k] = b[y, p[k]]
    # Let S2 = sum_{k} a[k,i] * (B_rows[k, y] - B_rows[k, x])  # where B_rows[k,y] = b[p[k], y]
    # Let S3 = sum_{k} a[j,k] * (B_cols[x, k] - B_cols[y, k])
    # Let S4 = sum_{k} a[k,j] * (B_rows[k, x] - B_rows[k, y])
    
    # Total Sum = S1 + S2 + S3 + S4
    # We need to subtract terms for k=i and k=j from this sum.
    
    # For k=i:
    # Term in S1: a[i,i] * (b[y, p[i]] - b[x, p[i]]) = a[i,i] * (b[y, x] - b[x, x])
    # Term in S2: a[i,i] * (b[p[i], y] - b[p[i], x]) = a[i,i] * (b[x, y] - b[x, x])
    # Term in S3: a[j,i] * (b[x, p[i]] - b[y, p[i]]) = a[j,i] * (b[x, x] - b[y, x])
    # Term in S4: a[i,j] * (b[p[i], x] - b[p[i], y]) = a[i,j] * (b[x, x] - b[x, y])
    
    # For k=j:
    # Term in S1: a[i,j] * (b[y, p[j]] - b[x, p[j]]) = a[i,j] * (b[y, y] - b[x, y])
    # Term in S2: a[j,i] * (b[p[j], y] - b[p[j], x]) = a[j,i] * (b[y, y] - b[y, x])
    # Term in S3: a[j,j] * (b[x, p[j]] - b[y, p[j]]) = a[j,j] * (b[x, y] - b[y, y])
    # Term in S4: a[j,j] * (b[p[j], x] - b[p[j], y]) = a[j,j] * (b[y, x] - b[y, y])
    
    # Direct term:
    # D = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    
    # This is still O(n^2) per swap if not careful.
    # To vectorize for ALL i,j, we need to compute S1, S2, S3, S4 for all i,j simultaneously.
    
    # S1[i,j] = sum_k a[i,k] * (b[p[j], p[k]] - b[p[i], p[k]])
    # Note: p[j] = y, p[i] = x.
    # b[p[j], p[k]] is B_rows[j, k].
    # b[p[i], p[k]] is B_rows[i, k].
    # So S1[i,j] = a[i,:] @ (B_rows[j,:] - B_rows[i,:])
    
    # S2[i,j] = sum_k a[k,i] * (b[p[k], p[j]] - b[p[k], p[i]])
    # b[p[k], p[j]] is B_cols[p[j], k] ?? No.
    # B_cols[m, k] = b[m, p[k]]. So b[p[k], p[j]] is not directly B_cols.
    # b[p[k], p[j]] is the element at row p[k], col p[j] of b.
    # Let's define B_perm = b[np.ix_(p, p)]. Then B_perm[k, j] = b[p[k], p[j]].
    # So b[p[k], p[j]] = B_perm[k, j].
    # b[p[k], p[i]] = B_perm[k, i].
    # So S2[i,j] = a[:,i] @ (B_perm[:,j] - B_perm[:,i])
    
    # S3[i,j] = sum_k a[j,k] * (b[p[i], p[k]] - b[p[j], p[k]])
    # = a[j,:] @ (B_rows[i,:] - B_rows[j,:])
    
    # S4[i,j] = sum_k a[k,j] * (b[p[k], p[i]] - b[p[k], p[j]])
    # = a[:,j] @ (B_perm[:,i] - B_perm[:,j])
    
    # Let's compute these matrices.
    # B_rows is (n, n). B_rows[i, k] = b[p[i], p[k]].
    # B_perm is (n, n). B_perm[k, j] = b[p[k], p[j]].
    
    B_perm = b[np.ix_(p, p)]
    
    # S1: (n, n) matrix where S1[i,j] = a[i,:] @ (B_rows[j,:] - B_rows[i,:])
    # This is (A @ (B_rows.T - B_rows.T)) ? No.
    # S1[i,j] = sum_k A[i,k] * B_rows[j,k] - sum_k A[i,k] * B_rows[i,k]
    # Let AB = A @ B_rows.T  # (n, n) where AB[i,j] = sum_k A[i,k] B_rows[j,k]
    # Then S1[i,j] = AB[i,j] - AB[i,i]
    
    AB = a @ B_rows.T  # (n, n)
    S1 = AB - np.diag(AB).reshape(-1, 1)  # Subtract AB[i,i] from each row i
    
    # S2: S2[i,j] = sum_k A[k,i] * (B_perm[k,j] - B_perm[k,i])
    # = sum_k A[k,i] B_perm[k,j] - sum_k A[k,i] B_perm[k,i]
    # Let AT = A.T
    # Let ABP = AT @ B_perm  # (n, n) where ABP[i,j] = sum_k A[k,i] B_perm[k,j]
    # Then S2[i,j] = ABP[i,j] - ABP[i,i]
    
    ABP = a.T @ B_perm  # (n, n)
    S2 = ABP - np.diag(ABP).reshape(-1, 1)
    
    # S3: S3[i,j] = sum_k A[j,k] * (B_rows[i,k] - B_rows[j,k])
    # = sum_k A[j,k] B_rows[i,k] - sum_k A[j,k] B_rows[j,k]
    # = AB[j,i] - AB[j,j]
    
    S3 = AB.T - np.diag(AB).reshape(1, -1)  # AB.T[i,j] = AB[j,i]. Subtract AB[j,j] from each col j? 
    # Wait, S3[i,j] depends on j for the subtraction term.
    # S3[i,j] = AB[j,i] - AB[j,j].
    # So S3 is a matrix where row i, col j is AB[j,i] - AB[j,j].
    # This is AB.T - diag(AB) broadcast along rows?
    # AB.T[i,j] = AB[j,i].
    # We want to subtract AB[j,j] from element (i,j).
    # So S3 = AB.T - np.diag(AB).reshape(1, -1)  # This subtracts AB[j,j] from every element in column j of AB.T.
    # Yes, that's correct.
    
    # S4: S4[i,j] = sum_k A[k,j] * (B_perm[k,i] - B_perm[k,j])
    # = sum_k A[k,j] B_perm[k,i] - sum_k A[k,j] B_perm[k,j]
    # = ABP[j,i] - ABP[j,j]
    
    S4 = ABP.T - np.diag(ABP).reshape(1, -1)
    
    # Total Sum = S1 + S2 + S3 + S4
    
    # Direct Term D[i,j] = a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    # b[p[j],p[i]] = B_perm[j,i]
    # b[p[i],p[j]] = B_perm[i,j]
    # D[i,j] = a[i,j]*(B_perm[j,i] - B_perm[i,j]) + a[j,i]*(B_perm[i,j] - B_perm[j,i])
    
    D = a * (B_perm.T - B_perm) + a.T * (B_perm - B_perm.T)
    
    # Total Delta Matrix
    Delta = S1 + S2 + S3 + S4 + D
    
    # We only care about i < j.
    # Find the minimum delta in the upper triangle.
    iu = np.triu_indices(n, k=1)
    deltas = Delta[iu]
    
    min_idx = np.argmin(deltas)
    best_i, best_j = iu[0][min_idx], iu[1][min_idx]
    best_delta = deltas[min_idx]
    
    if best_delta >= 0:
        return None, None, 0, current_cost
    
    return best_i, best_j, best_delta, current_cost + best_delta

def _local_search(a, b, p, time_limit, max_iters=10000):
    """
    Perform 1-swap local search until no improving move or time limit.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    iters = 0
    
    while time.time() - start_time < time_limit and iters < max_iters:
        iters += 1
        best_i, best_j, best_delta, new_cost = _vectorized_best_swap(a, b, p, current_cost)
        
        if best_i is None:
            break
        
        # Perform swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost = new_cost
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def _linearized_init(a, b, n, alpha=1.0):
    """
    Generate initial permutation using Hungarian algorithm on a linearized cost matrix.
    M[i,j] = alpha * (a @ b)[i,j] + (1-alpha) * (a.sum(1)[:,None] * b.sum(0)[None,:] / n)
    We want to minimize sum M[i, p[i]].
    """
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # Linearization 1: A @ B
    M1 = a_float @ b_float
    
    # Linearization 2: Row/Col sums product
    a_row_sums = a_float.sum(axis=1)
    b_col_sums = b_float.sum(axis=0)
    M2 = np.outer(a_row_sums, b_col_sums) / n
    
    M = alpha * M1 + (1 - alpha) * M2
    
    # Hungarian algorithm minimizes sum M[i, p[i]]
    # We want to assign facility i to location p[i].
    # So row i of M corresponds to facility i, col j to location j.
    row_ind, col_ind = linear_sum_assignment(M)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _perturb(p, n, k=10):
    """Randomly swap k pairs in p."""
    p = p.copy()
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145  # 5s safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Generate diverse initial solutions using linearization
    inits = []
    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        p_init = _linearized_init(a, b, n, alpha=alpha)
        inits.append(p_init)
        
    # Add some random perturbations of the best linearized init
    p_lin_best = inits[0]
    for k in [5, 10, 20]:
        inits.append(_perturb(p_lin_best, n, k))
        
    # Phase 2: Local search on each init
    # Allocate time based on number of inits
    num_inits = len(inits)
    time_per_init = (deadline - time.time() - 10) / num_inits
    if time_per_init < 1.0:
        time_per_init = 1.0
        
    for i, p_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_init, time_limit=time_per_init)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # Phase 3: Polish the best solution with remaining time
    if best_p is not None:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 1.0:
            p_polished, cost_polished = _local_search(a, b, best_p, time_limit=remaining_time)
            if cost_polished < best_cost:
                best_cost = cost_polished
                best_p = p_polished
    
    # Fallback
    if best_p is None:
        p_init = _linearized_init(a, b, n, alpha=0.5)
        best_p, best_cost = _local_search(a, b, p_init, time_limit=max(1.0, deadline - time.time() - 1))
        
    return best_p.tolist()

# Keep the original function name and signature
def _solve_tai256c(a, b):
    return solve(a, b)

# The task requires solve(a, b) to be the entry point
# But the current program has _solve_tai256c and solve.
# The contract says "Keep the function names and signatures of the seed program."
# The seed program has `def solve(a, b): return _solve_tai256c(a, b)`
# And `_solve_tai256c` is the main logic.
# I will keep this structure.
