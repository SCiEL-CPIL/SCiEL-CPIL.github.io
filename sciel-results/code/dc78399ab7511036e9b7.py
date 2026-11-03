"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    n = len(a)
    
    # Compute cost for a permutation
    def compute_cost(p):
        # p[i] is the location for facility i
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: for each i, sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i] @ b[p[i]])
        # More efficient: create permuted b and do matrix ops
        # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Let's use: for each i, the contribution is a[i,:] dot b[p[i],:]
        # But that's O(n^2). Let's think of it as:
        # cost = trace(a @ B_p) where B_p[i,j] = b[p[i], p[j]]
        # B_p = b[np.ix_(p, p)]
        # So cost = np.sum(a * b[np.ix_(p, p)])
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    # Initial heuristic: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # FISTA-style continuous relaxation to get a good starting point
    # We'll do a few iterations of projected gradient on the assignment matrix
    # X is n x n, X[i,j] = 1 if facility i at location j
    # Objective: f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    #            = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # This can be written as: f(X) = trace(X^T @ C @ X) where C[i,j,k,l] = a[i,j]*b[k,l]
    # Or more simply: f(X) = sum_{i,j} a[i,j] * (X @ b.T @ X.T)[i,j] ... let me re-derive
    # f(X) = sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    #      = sum_{i,j} a[i,j] * (X @ B @ X.T)[i,j]  where B = b
    #      = sum_{i,j} a[i,j] * (X @ b @ X.T)[i,j]
    #      = trace(A @ X @ b @ X.T) ... not quite
    # Let M = X @ b @ X.T, then f = sum_{i,j} a[i,j] * M[i,j] = trace(A.T @ M) = trace(A @ X @ b @ X.T)
    # Since A is symmetric (flows), f = trace(A @ X @ b @ X.T)
    
    # Gradient of f w.r.t. X:
    # f = trace(A @ X @ b @ X.T)
    # df/dX = A.T @ X @ b.T + A @ X @ b  (using matrix calculus)
    # If A is symmetric: df/dX = A @ X @ (b + b.T)
    # Actually let me be more careful.
    # f(X) = sum_{i,j} A[i,j] * (X @ b @ X.T)[i,j]
    # Let Y = X @ b @ X.T
    # df/dY[i,j] = A[i,j]
    # dY/dX: Y[i,j] = sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    # df/dX[i,m] = sum_j A[i,j] * dY[i,j]/dX[i,m] + sum_j A[j,i] * dY[j,i]/dX[i,m]
    # dY[i,j]/dX[i,m] = b[m,j]
    # dY[j,i]/dX[i,m] = b[j,m]
    # So df/dX[i,m] = sum_j A[i,j] * b[m,j] + sum_j A[j,i] * b[j,m]
    #               = (A @ b.T)[i,m] + (A.T @ b)[i,m]  ... wait
    # sum_j A[i,j] * b[m,j] = (A @ b.T)[i,m]
    # sum_j A[j,i] * b[j,m] = (A.T @ b)[i,m]
    # So grad = A @ b.T + A.T @ b
    # If A is symmetric: grad = A @ (b + b.T)
    
    # This gradient is constant w.r.t. X! That's because the objective is quadratic in X
    # but the "quadratic" form here leads to a linear gradient in X.
    # Wait, that can't be right. Let me recheck.
    # f(X) = sum_{i,j} A[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    # This is NOT quadratic in X in the usual sense. It's a 4-index contraction.
    # Let me write it as: f = sum_{i,j,k,l} A[i,j] * b[k,l] * X[i,k] * X[j,l]
    # = sum_{k,l} b[k,l] * sum_{i,j} A[i,j] * X[i,k] * X[j,l]
    # = sum_{k,l} b[k,l] * (X.T @ A @ X)[k,l]
    # = trace(b.T @ X.T @ A @ X)
    # = trace(A @ X @ b.T @ X.T)
    # So f(X) = trace(A @ X @ b.T @ X.T)
    
    # Gradient: d/dX trace(A @ X @ b.T @ X.T)
    # = A.T @ X @ b + A @ X @ b.T  ... let me use standard results
    # d/dX trace(M @ X @ N @ X.T) = M.T @ X @ N.T + M @ X @ N
    # Here M = A, N = b.T
    # So grad = A.T @ X @ (b.T).T + A @ X @ b.T = A.T @ X @ b + A @ X @ b.T
    # If A is symmetric: grad = A @ X @ (b + b.T)
    
    # So the gradient IS linear in X. This means the continuous relaxation is a quadratic form
    # and the gradient is linear. FISTA on this should work.
    
    # Actually, for the assignment problem relaxation, the standard approach is:
    # min trace(A @ X @ b.T @ X.T) s.t. X is doubly stochastic
    # The gradient is A @ X @ (b + b.T) if A is symmetric.
    
    # Let's do FISTA with projection onto doubly stochastic matrices (Sinkhorn)
    
    # Precompute
    grad_const = a @ (b + b.T)  # This is the "A @ (b + b.T)" part, grad = grad_const @ X if A symmetric
    # Wait, grad = A @ X @ (b + b.T). So it's not just grad_const @ X.
    # grad[i,m] = sum_j A[i,j] * (b+j.T)[j,m] ... no
    # grad = A @ X @ (b + b.T)
    # So for each X, we need to compute A @ X @ (b + b.T)
    
    b_sym = b + b.T
    
    def compute_gradient(X):
        return a @ X @ b_sym
    
    def project_doubly_stochastic(X, max_iter=200, tol=1e-6):
        # Sinkhorn-Knopp algorithm
        X = np.maximum(X, 1e-10)
        n = X.shape[0]
        for _ in range(max_iter):
            # Normalize rows
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            # Normalize columns
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            # Check convergence
            if _ % 10 == 9:
                diff = np.abs(X.sum(axis=1) - 1).max() + np.abs(X.sum(axis=0) - 1).max()
                if diff < tol:
                    break
        return X
    
    # FISTA
    X = np.ones((n, n)) / n  # start with uniform doubly stochastic
    X = project_doubly_stochastic(X, max_iter=50)
    
    # FISTA parameters
    step_size = 0.5  # will adapt
    t = 1.0
    Y = X.copy()
    best_X = X.copy()
    best_f = np.inf
    
    def compute_f(X):
        # f(X) = trace(A @ X @ b.T @ X.T)
        # = sum_{i,j} A[i,j] * (X @ b.T @ X.T)[i,j]
        # = sum_{i,j} A[i,j] * sum_{k,l} X[i,k] * b.T[k,l] * X[j,l]
        # = sum_{i,j,k,l} A[i,j] * b[l,k] * X[i,k] * X[j,l]
        # Let's just compute it directly
        M = X @ b.T @ X.T
        return np.sum(a * M)
    
    # Do a few FISTA iterations
    for it in range(200):
        if time.time() - start_time > deadline - 30:
            break
        
        # Compute gradient at Y
        G = compute_gradient(Y)
        
        # FISTA update
        Z = Y - step_size * G
        X_new = project_doubly_stochastic(Z, max_iter=100)
        
        # Update t
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Y = X_new + ((t - 1) / t_new) * (X_new - X)
        Y = np.maximum(Y, 0)  # keep non-negative
        X = X_new
        t = t_new
        
        # Track best
        f_val = compute_f(X)
        if f_val < best_f:
            best_f = f_val
            best_X = X.copy()
    
    # Recover permutation from best_X using Hungarian
    if time.time() - start_time < deadline - 25:
        # Use the relaxed solution to guide the initial permutation
        # Take argmax of each row as a starting point, then fix conflicts
        row_max = np.argmax(best_X, axis=1)
        # Resolve conflicts greedily
        p_hungarian = np.zeros(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        order = np.argsort(-best_X.sum(axis=1), kind="stable")
        for i in order:
            loc = row_max[i]
            if not used[loc]:
                p_hungarian[i] = loc
                used[loc] = True
            else:
                # Find best unused
                best_loc = np.argmax(best_X[i] * (~used))
                p_hungarian[i] = best_loc
                used[best_loc] = True
        
        cost_hung = compute_cost(p_hungarian)
        if cost_hung < best_cost:
            best_p = p_hungarian.copy()
            best_cost = cost_hung
    
    # Also try Hungarian on the full matrix for a baseline
    if time.time() - start_time < deadline - 20:
        # Hungarian on -best_X to get a good permutation
        row_ind, col_ind = linear_sum_assignment(-best_X)
        p_hung2 = np.zeros(n, dtype=np.int64)
        p_hung2[row_ind] = col_ind
        cost_hung2 = compute_cost(p_hung2)
        if cost_hung2 < best_cost:
            best_p = p_hung2.copy()
            best_cost = cost_hung2
    
    # Local search: 2-swap
    def compute_swap_cost(p, i, j):
        # Cost of swapping locations of facilities i and j
        # Original: b[p[i], p[j]] and b[p[j], p[i]]
        # New: b[p[j], p[i]] and b[p[i], p[j]]
        # The change in cost is:
        # sum_k a[i,k] * (b[p[j], p[k]] - b[p[i], p[k]]) + sum_k a[j,k] * (b[p[i], p[k]] - b[p[j], p[k]])
        # + sum_k a[k,i] * (b[p[k], p[j]] - b[p[k], p[i]]) + sum_k a[k,j] * (b[p[k], p[i]] - b[p[k], p[j]])
        # = sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
        #   + sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
        
        li = p[i]
        lj = p[j]
        
        # delta = sum_k (a[i,k] - a[j,k]) * (b[lj, k] - b[li, k])
        #       + sum_k (a[k,i] - a[k,j]) * (b[k, lj] - b[k, li])
        
        delta1 = np.sum((a[i] - a[j]) * (b[lj] - b[li]))
        delta2 = np.sum((a[:, i] - a[:, j]) * (b[:, lj] - b[:, li]))
        return delta1 + delta2
    
    # Vectorized 2-swap local search
    improved = True
    while improved and time.time() - start_time < deadline - 5:
        improved = False
        # Compute all swap deltas
        # For each pair (i, j), compute the delta
        # This is O(n^2) pairs, each O(n) work = O(n^3)
        # For n=150, that's 150^3 = 3.375M operations per pass, which is fast
        
        # Precompute for efficiency
        # delta(i,j) = sum_k (a[i,k] - a[j,k]) * (b[p[j],k] - b[p[i],k])
        #            + sum_k (a[k,i] - a[k,j]) * (b[k,p[j]] - b[k,p[i]])
        
        # Let's vectorize: for fixed i, compute deltas with all j
        # Actually, let's just do it in a loop but vectorize over j for each i
        
        for i in range(n):
            if time.time() - start_time > deadline - 5:
                break
            li = p[i]
            
            # For all j != i, compute delta
            # delta(i,j) = sum_k (a[i,k] - a[j,k]) * (b[p[j],k] - b[p[i],k])
            #            + sum_k (a[k,i] - a[k,j]) * (b[k,p[j]] - b[k,p[i]])
            
            # Term 1: sum_k (a[i,k] - a[j,k]) * (b[p[j],k] - b[p[i],k])
            # = sum_k a[i,k]*b[p[j],k] - sum_k a[i,k]*b[p[i],k] - sum_k a[j,k]*b[p[j],k] + sum_k a[j,k]*b[p[i],k]
            
            # Let's precompute: for each location l, a_col[l] = a[:, l] ... no
            # Let me just compute directly with numpy
            
            j_idx = np.array([j for j in range(n) if j != i])
            if len(j_idx) == 0:
                continue
            
            pj = p[j_idx]  # locations of all j
            bi = b[li]     # b[p[i], :]
            bj = b[pj]     # b[p[j], :] shape (n-1, n)
            
            ai = a[i]      # a[i, :]
            aj = a[j_idx]  # a[j, :] shape (n-1, n)
            
            # Term 1: sum_k (ai - aj) * (bj - bi)
            # = sum_k (ai - aj) * bj - sum_k (ai - aj) * bi
            # = (ai - aj) @ bj.T - (ai - aj) @ bi  ... but bi is a vector
            # term1 = np.sum((ai[None,:] - aj) * (bj - bi[None,:]), axis=1)
            
            term1 = np.sum((ai[None, :] - aj) * (bj - bi[None, :]), axis=1)
            
            # Term 2: sum_k (a[k,i] - a[k,j]) * (b[k,p[j]] - b[k,p[i]])
            # = sum_k a[k,i]*b[k,pj] - sum_k a[k,i]*b[k,li] - sum_k a[k,j]*b[k,pj] + sum_k a[k,j]*b[k,li]
            
            a_i_col = a[:, i]  # shape (n,)
            a_j_cols = a[:, j_idx]  # shape (n, n-1)
            
            b_li_col = b[:, li]  # shape (n,)
            b_pj_cols = b[:, pj]  # shape (n, n-1)
            
            term2 = np.sum((a_i_col[:, None] - a_j_cols) * (b_pj_cols - b_li_col[:, None]), axis=0)
            
            deltas = term1 + term2
            
            # Find the best improving swap
            best_delta = np.min(deltas)
            if best_delta < -1e-10:
                best_j = j_idx[np.argmin(deltas)]
                # Perform swap
                p[i], p[best_j] = p[best_j], p[i]
                best_cost += int(best_delta)
                improved = True
    
    # Final cost computation to be sure
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
