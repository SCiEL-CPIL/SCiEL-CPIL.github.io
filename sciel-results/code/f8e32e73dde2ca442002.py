"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    Uses a smooth optimization approach with gradient descent, attempting to find
    lower-rank decompositions than the schoolbook rank of 60.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    def compute_error_fast(U, V, W, rank):
        """Compute max absolute error using vectorized ops."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def compute_errors_vectorized(U, V, W, rank):
        """Compute all errors as a 3D array."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
        return errors

    def try_rank(rank, num_trials, max_iterations, time_budget):
        """Try to find a decomposition of given rank using smooth optimization."""
        if time.time() > start_time + time_budget:
            return None

        for trial in range(num_trials):
            if time.time() > start_time + time_budget:
                break

            np.random.seed(rank * 7919 + trial * 104729)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5

            # Use log-sum-exp style smooth max for the objective
            # Minimize sum(exp(beta * err^2)) which approximates max err^2
            beta = 50.0
            prev_loss = float('inf')
            stuck_count = 0

            for iteration in range(max_iterations):
                if time.time() > start_time + time_budget:
                    break

                errors = compute_errors_vectorized(U, V, W, rank)
                max_err = np.max(np.abs(errors))
                
                # Smooth surrogate: sum of exp(beta * err^2 / (1+beta*max_err^2))
                # This is a soft-max approximation
                scaled = errors**2
                max_scaled = np.max(scaled)
                if max_scaled < 1e-12:
                    break
                # Normalize to prevent overflow
                norm = max_scaled
                exp_terms = np.exp(beta * (scaled - norm) / (1 + beta))
                loss = np.sum(exp_terms) / (1 + beta)
                
                if loss < 1e-12:
                    break

                if loss > prev_loss * 1.01:
                    stuck_count += 1
                else:
                    stuck_count = 0
                prev_loss = loss

                if stuck_count > 30:
                    # Restart with different random seed
                    np.random.seed(rank * 100003 + trial * 104729 + iteration)
                    U = np.random.randn(n*m, rank) * 0.3
                    V = np.random.randn(m*p, rank) * 0.3
                    W = np.random.randn(p*n, rank) * 0.3
                    prev_loss = float('inf')
                    stuck_count = 0
                    continue

                # Adaptive learning rate based on error magnitude
                lr = 0.02 / (1.0 + max_err * 10)

                # Gradient for U: for each (i,j), grad_U[i*m+j,c] = sum_k errors[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
                for i in range(n):
                    for j in range(m):
                        grad = np.zeros(rank)
                        for k in range(p):
                            grad += errors[i,j,k] * V[j*p+k] * W[k*n+i]
                        U[i*m+j] -= lr * grad

                # Gradient for V: for each (j,k), grad_V[j*p+k,c] = sum_i errors[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
                for j in range(m):
                    for k in range(p):
                        grad = np.zeros(rank)
                        for i in range(n):
                            grad += errors[i,j,k] * U[i*m+j] * W[k*n+i]
                        V[j*p+k] -= lr * grad

                # Gradient for W: for each (i,k), grad_W[k*n+i,c] = sum_j errors[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
                for i in range(n):
                    for k in range(p):
                        grad = np.zeros(rank)
                        for j in range(m):
                            grad += errors[i,j,k] * U[i*m+j] * V[j*p+k]
                        W[k*n+i] -= lr * grad

            # Check convergence
            err = compute_error_fast(U, V, W, rank)
            if err < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2

                err_rounded = compute_error_fast(U_rounded, V_rounded, W_rounded, rank)
                if err_rounded < 1e-8:
                    # Convert to list of lists
                    U_list = [[float(U_rounded[i, c]) for c in range(rank)] for i in range(n*m)]
                    V_list = [[float(V_rounded[i, c]) for c in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_rounded[i, c]) for c in range(rank)] for i in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}

            # Also try with integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            err_int = compute_error_fast(U_int, V_int, W_int, rank)
            if err_int < 1e-8:
                U_list = [[float(U_int[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V_int[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W_int[i, c]) for c in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from high to low, with more trials for lower ranks
    # Focus on ranks that might be achievable: 45-55
    for rank in range(55, 44, -1):
        if time.time() > start_time + 105:
            break

        # Allocate more time for lower ranks (harder to find)
        if rank <= 47:
            time_budget = 100
            num_trials = 8
            max_iterations = 250
        elif rank <= 50:
            time_budget = 90
            num_trials = 10
            max_iterations = 200
        else:
            time_budget = 80
            num_trials = 12
            max_iterations = 180

        result = try_rank(rank, num_trials, max_iterations, time_budget)
        if result is not None:
            best_result = result
            best_rank = rank
            break

    return best_result
# EVOLVE-BLOCK-END
