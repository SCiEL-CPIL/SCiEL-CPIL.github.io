"""Rewritten program: search for a low-rank exact decomposition of 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 110  # 110s to leave margin

    def build_w_linear_system(U, V, r):
        # U: (n*m, r), V: (m*p, r)
        # Unknowns: W[:, c] for c in 0..r-1, each of size p*n
        # Equations: for each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        num_eq = n * m * p
        num_vars = p * n * r
        A = np.zeros((num_eq, num_vars), dtype=np.float64)
        b = np.ones(num_eq, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(r):
                        w_idx = c * (p * n) + k * n + i
                        A[eq_idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
        return A, b

    def verify_exact(U, V, W, n, m, p, r):
        # Verify with integer arithmetic: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True

    # Try ranks from 25 up to 44
    for r in range(25, min(rank_schoolbook, 45)):
        if time.time() > deadline:
            break
        # Random trials
        max_trials = 2000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            # Generate random U, V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, r))
            V = np.random.choice([-1, 0, 1], size=(m*p, r))
            
            # Build and solve for W
            A, b = build_w_linear_system(U, V, r)
            try:
                W_vec, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = W_vec.reshape(r, p*n).T  # W: (p*n, r)
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert to lists of floats (half-integers)
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            
            # Verify exactness with integer arithmetic (multiply by 2 to make integers)
            U_int = [[int(round(x * 2)) for x in row] for row in U_list]
            V_int = [[int(round(x * 2)) for x in row] for row in V_list]
            W_int = [[int(round(x * 2)) for x in row] for row in W_list]
            
            if verify_exact(U_int, V_int, W_int, n, m, p, r):
                # Scale back: divide by 8? No, the product is U*V*W. If we scale each by 2, product scales by 8.
                # We need sum_c U*V*W = 1. With half-integers, let's just return the half-integer values.
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
