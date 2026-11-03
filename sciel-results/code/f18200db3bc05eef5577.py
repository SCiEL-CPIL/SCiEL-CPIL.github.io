"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead

    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    # Precompute some useful quantities
    # f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    #      = trace(X^T * A * X * B^T) ... let's verify
    # Actually: f(X) = sum_{k,l} b[k,l] * (X^T * A * X)[k,l]
    #           = trace(B^T * X^T * A * X)
    #           = trace(X * B^T * X^T * A)
    #           = sum_{i,j,k,l} A[i,j] * X[i,k] * B[k,l] * X[j,l] ✓
    
    # Gradient: grad[p,q] = (A @ X @ B.T)[p,q] + (B @ X.T @ A.T)[p,q]
    # Let's verify: f = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # df/dX[p,q]: terms where i=p,k=q: sum_{j,l} A[p,j] B[q,l] X[j,l] = (A X B^T)[p,q]
    #             terms where j=p,l=q: sum_{i,k} A[i,q] B[k,p] X[i,k] = (B X^T A^T)[p,q]
    # So grad = A @ X @ B.T + B @ X.T @ A.T ✓

    # Initial permutation: rank-matching
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central

    def perm_cost(p):
        """Compute cost of permutation p."""
        return int(np.sum(a_f * b_f[np.ix_(p, p)]))

    best_cost = perm_cost(p0)
    best_p = p0.copy()

    # Create initial doubly stochastic matrix from permutation
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p0[i]] = 1.0

    # Sinkhorn-Knopp for projection to doubly stochastic
    def sinkhorn(X, max_iter=100, tol=1e-7):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-10)
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            if np.max(np.abs(X.sum(axis=1) - 1.0)) < tol and np.max(np.abs(X.sum(axis=0) - 1.0)) < tol:
                break
        return X

    # Adaptive step size with backtracking
    base_step = 1.0 / (n * max(a_f.max(), b_f.max(), 1.0))
    step_size = base_step

    best_relaxed_cost = np.inf
    best_X = X.copy()

    for iteration in range(3000):
        if time.time() - t0 > time_limit:
            break

        # Compute gradient
        grad = a_f @ X @ b_f.T + b_f @ X.T @ a_f.T

        # Compute current relaxed cost
        C = X.T @ a_f @ X
        cost = np.sum(b_f * C)

        if cost < best_relaxed_cost:
            best_relaxed_cost = cost
            best_X = X.copy()

        # Try to improve with backtracking line search
        improved = False
        alpha = step_size
        for _ in range(5):
            X_new = X - alpha * grad
            X_new = sinkhorn(X_new, max_iter=30)
            
            C_new = X_new.T @ a_f @ X_new
            cost_new = np.sum(b_f * C_new)
            
            if cost_new < cost:
                X = X_new
                step_size = alpha
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Reduce step size if we can't improve
            step_size *= 0.5
            if step_size < 1e-10:
                break

    # Recover permutation from best_X using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p_relaxed = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p_relaxed[i] = j

    cost_relaxed = perm_cost(p_relaxed)
    if cost_relaxed < best_cost:
        best_cost = cost_relaxed
        best_p = p_relaxed.copy()

    # Local search: 2-opt style swaps
    def local_search_2opt(p, max_iter=100, time_limit_inner=15.0):
        """Improve permutation via pairwise swaps."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        
        for _ in range(max_iter):
            if time.time() - t_start > time_limit_inner:
                break
            
            improved = False
            # Try random pairs of swaps
            for _ in range(100):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                # Cost change from swapping p[i] and p[j]
                # Old contributions involving i or j
                pi, pj = p[i], p[j]
                
                # Compute delta cost efficiently
                # Cost = sum_{k,l} a[k,l] * b[p[k],p[l]]
                # Only terms involving i or j change
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Terms a[k,i]*b[p[k],pi] -> a[k,i]*b[p[k],pj]
                    # Terms a[k,j]*b[p[k],pj] -> a[k,j]*b[p[k],pi]
                    delta += a_f[k, i] * (b_f[p[k], pj] - b_f[p[k], pi])
                    delta += a_f[k, j] * (b_f[p[k], pi] - b_f[p[k], pj])
                
                # Terms a[i,i]*b[pi,pi] -> a[i,i]*b[pj,pj]
                delta += a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi])
                # Terms a[j,j]*b[pj,pj] -> a[j,j]*b[pi,pi]
                delta += a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
                # Terms a[i,j]*b[pi,pj] -> a[i,j]*b[pj,pi]
                delta += a_f[i, j] * (b_f[pj, pi] - b_f[pi, pj])
                # Terms a[j,i]*b[pj,pi] -> a[j,i]*b[pi,pj]
                delta += a_f[j, i] * (b_f[pi, pj] - b_f[pj, pi])
                
                if delta < -1e-10:
                    p[i], p[j] = pj, pi
                    current_cost += delta
                    improved = True
                    break
            
            if not improved:
                break
        
        return p, current_cost

    # Apply local search to both initial and relaxed permutations
    if time.time() - t0 < time_limit - 5:
        p_ls1, cost_ls1 = local_search_2opt(best_p, max_iter=50, time_limit_inner=10.0)
        if cost_ls1 < best_cost:
            best_cost = cost_ls1
            best_p = p_ls1

    if time.time() - t0 < time_limit - 5:
        p_ls2, cost_ls2 = local_search_2opt(p_relaxed, max_iter=50, time_limit_inner=10.0)
        if cost_ls2 < best_cost:
            best_cost = cost_ls2
            best_p = p_ls2

    # Final attempt: if time remains, do more local search
    if time.time() - t0 < time_limit - 3:
        p_final, cost_final = local_search_2opt(best_p, max_iter=100, time_limit_inner=10.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
