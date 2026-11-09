import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def solve_W(U, V, n, m, p, r):
    W = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            # For fixed (k, i), solve for W[k*n+i, :]
            # Equations: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Coefficient matrix: m x r, entry (j, c) = U[i*m+j, c] * V[j*p+k, c]
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i * m + j, c] * V[j * p + k, c]
            b = np.ones(m)
            # Solve A @ W_row = b
            W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = W_row
    return W

def solve_U(V, W, n, m, p, r):
    U = np.zeros((n * m, r))
    for i in range(n):
        for j in range(m):
            # For fixed (i, j), solve for U[i*m+j, :]
            # Equations: for each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Coefficient matrix: p x r, entry (k, c) = V[j*p+k, c] * W[k*n+i, c]
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j * p + k, c] * W[k * n + i, c]
            b = np.ones(p)
            U_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            U[i * m + j, :] = U_row
    return U

def solve_V(U, W, n, m, p, r):
    V = np.zeros((m * p, r))
    for j in range(m):
        for k in range(p):
            # For fixed (j, k), solve for V[j*p+k, :]
            # Equations: for each i, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Coefficient matrix: n x r, entry (i, c) = U[i*m+j, c] * W[k*n+i, c]
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i * m + j, c] * W[k * n + i, c]
            b = np.ones(n)
            V_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            V[j * p + k, :] = V_row
    return V

def round_to_half_int(X):
    return np.round(2 * X) / 2

def verify_decomposition(U, V, W, n, m, p, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def try_rank(r, n, m, p, max_iters=100, n_trials=10, time_limit=10):
    start_time = time.time()
    for trial in range(n_trials):
        if time.time() - start_time > time_limit:
            break
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            W = solve_W(U, V, n, m, p, r)
            U = solve_U(V, W, n, m, p, r)
            V = solve_V(U, W, n, m, p, r)
        # Round to half-integers
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        if verify_decomposition(U_r, V_r, W_r, n, m, p, r):
            return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    for r in range(85, 49, -1):
        result = try_rank(r, n, m, p, max_iters=50, n_trials=5, time_limit=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to naive decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
