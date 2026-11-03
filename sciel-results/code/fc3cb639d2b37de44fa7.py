"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    # Precompute transposes
    a_T = a_f.T.copy()
    b_T = b_f.T.copy()

    # Initial permutation: rank matching
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    # Sort facilities by flow volume (descending) and locations by distance volume (ascending)
    facility_order = np.argsort(-a_row_sums, kind="stable")
    location_order = np.argsort(b_row_sums, kind="stable")
    
    p0 = np.empty(n, dtype=np.int64)
    for rank in range(n):
        p0[facility_order[rank]] = location_order[rank]
    
    # Build initial doubly stochastic matrix
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p0[i]] = 1.0

    def sinkhorn_project(X, max_iter=30, tol=1e-7):
        """Sinkhorn-Knopp projection to doubly stochastic."""
        X = np.maximum(X, 1e-12)
        for _ in range(max_iter):
            X /= X.sum(axis=1, keepdims=True)
            X /= X.sum(axis=0, keepdims=True)
            # Check convergence
            row_err = np.max(np.abs(X.sum(axis=1) - 1.0))
            col_err = np.max(np.abs(X.sum(axis=0) - 1.0))
            if row_err < tol and col_err < tol:
                break
        return X

    # Lipschitz constant estimate for gradient
    L = 2.0 * np.max(np.abs(a_f)) * np.max(np.abs(b_f)) * 2.0
    # Conservative initial step
    step_size = 1.0 / (L + 1e-10)
    max_step = 10.0 / (L + 1e-10)

    best_cost = np.inf
    best_X = X.copy()
    best_perm_cost = np.inf
    best_perm = p0.copy()

    def compute_permutation_cost(p):
        """Compute exact cost of permutation p."""
        cost = 0
        for i in range(n):
            pi = p[i]
            row_a = a_f[i, :]
            row_b = b_f[pi, :]
            # cost contribution: sum_j a[i,j] * b[p[i], p[j]]
            # = sum_j a[i,j] * b[pi, p[j]]
            cost += np.dot(row_a, row_b[p])
        return cost

    # Evaluate initial permutation
    best_perm_cost = compute_permutation_cost(p0)
    best_perm = p0.copy()

    for iteration in range(15000):
        if time.time() - t0 > time_limit - 5.0:
            break

        # Compute gradient: grad = A @ X @ B.T + B @ X.T @ A.T
        AX = a_f @ X
        term1 = AX @ b_T
        term2 = b_f @ AX.T
        grad = term1 + term2

        # Compute current relaxed cost
        C = X.T @ AX
        cost = np.sum(b_f * C)

        if cost < best_cost:
            best_cost = cost
            best_X = X.copy()

        # Gradient descent step
        X = X - step_size * grad

        # Project to doubly stochastic
        X = sinkhorn_project(X, max_iter=20)

        # Adaptive step size
        AX_new = a_f @ X
        C_new = X.T @ AX_new
        cost_new = np.sum(b_f * C_new)

        if cost_new > cost * 1.005:
            step_size *= 0.5
        elif cost_new < cost * 0.995:
            step_size = min(step_size * 1.2, max_step)

        # Every 100 iterations, try to extract a permutation and evaluate
        if (iteration + 1) % 100 == 0 or iteration == 14999:
            row_ind, col_ind = linear_sum_assignment(-X)
            p_candidate = np.empty(n, dtype=np.int64)
            for i, j in zip(row_ind, col_ind):
                p_candidate[i] = j
            
            perm_cost = compute_permutation_cost(p_candidate)
            if perm_cost < best_perm_cost:
                best_perm_cost = perm_cost
                best_perm = p_candidate.copy()

    # Final extraction from best_X
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p_final = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p_final[i] = j
    
    final_cost = compute_permutation_cost(p_final)
    if final_cost < best_perm_cost:
        best_perm = p_final

    return best_perm.tolist()
# EVOLVE-BLOCK-END
