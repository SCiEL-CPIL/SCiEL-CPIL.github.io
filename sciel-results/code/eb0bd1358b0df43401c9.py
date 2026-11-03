"""Rewritten program: search for a low-rank exact decomposition of 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

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
        # Verify with integer arithmetic: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 8
        # U, V, W are lists of lists of integers (scaled by 2 for half-integers)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 8:  # 2*2*2 = 8 scaling factor
                        return False
        return True

    def try_random_decomposition(r, U_init, V_init):
        """Try to find W such that the decomposition is exact."""
        U = U_init.copy()
        V = V_init.copy()
        
        # Build and solve for W
        A, b = build_w_linear_system(U, V, r)
        try:
            W_vec, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None
        
        W = W_vec.reshape(r, p*n).T  # W: (p*n, r)
        
        # Round to nearest integer (since we're working with integers now)
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        # Verify exactness
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        
        if verify_exact(U_list, V_list, W_list, n, m, p, r):
            # Convert to half-integers for the output
            U_out = [[float(x) / 2 for x in row] for row in U_list]
            V_out = [[float(x) / 2 for x in row] for row in V_list]
            W_out = [[float(x) / 2 for x in row] for row in W_list]
            return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    def schoolbook_decomposition():
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

    # Try ranks from 25 up to 44
    best_result = None
    for r in range(25, min(rank_schoolbook, 45)):
        if time.time() > deadline:
            break
        # Random trials with different distributions
        max_trials = 5000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random U, V with various distributions
            trial_type = trial % 4
            
            if trial_type == 0:
                # Sparse {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n*m, r))
                V = np.random.choice([-1, 0, 1], size=(m*p, r))
            elif trial_type == 1:
                # Denser {-1, 0, 1} with higher density
                U = np.where(np.random.rand(n*m, r) < 0.7, np.random.choice([-1, 1], size=(n*m, r)), 0)
                V = np.where(np.random.rand(m*p, r) < 0.7, np.random.choice([-1, 1], size=(m*p, r)), 0)
            elif trial_type == 2:
                # Small integers {-2, -1, 0, 1, 2}
                U = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, r))
                V = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, r))
            else:
                # Very sparse with larger values
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for _ in range(n*m*r//3):
                    i, j = np.random.randint(0, n*m), np.random.randint(0, r)
                    U[i, j] = np.random.choice([-2, -1, 1, 2])
                for _ in range(m*p*r//3):
                    i, j = np.random.randint(0, m*p), np.random.randint(0, r)
                    V[i, j] = np.random.choice([-2, -1, 1, 2])
            
            result = try_random_decomposition(r, U, V)
            if result is not None:
                return result
    
    # Fallback: schoolbook
    return schoolbook_decomposition()
