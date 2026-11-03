"""Rewritten program: search for a low-rank exact decomposition of 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = time.time() + 135  # 135s to leave margin for 150s limit

    def verify_exact(U, V, W, n, m, p, r):
        """Verify with exact integer arithmetic that the decomposition is exact.
        U, V, W are lists of lists of integers (scaled by 2 for half-integers).
        For each (i,j,k): sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 8
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 8:
                        return False
        return True

    def build_and_solve_W(U, V, r, n, m, p):
        """Build the linear system for W given U, V and solve it.
        Returns W as a 2D array (p*n, r) or None if no exact solution.
        """
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
        
        try:
            W_vec, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        
        if len(residuals) > 0 and residuals[0] > 1e-4:
            return None
        
        W = W_vec.reshape(r, p * n).T  # W: (p*n, r)
        return W

    def try_integer_solution(U, V, r, n, m, p):
        """Given integer U, V, try to find integer W that satisfies the equations exactly.
        Returns (U_list, V_list, W_list) if found, else None.
        """
        W = build_and_solve_W(U, V, r, n, m, p)
        if W is None:
            return None
        
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        U_list = U_int.tolist()
        V_list = V_int.tolist()
        W_list = W_int.tolist()
        
        if verify_exact(U_list, V_list, W_list, n, m, p, r):
            return (U_list, V_list, W_list)
        
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

    def to_half_integer_output(U_list, V_list, W_list):
        """Convert integer lists (scaled by 2) to half-integer float lists."""
        U_out = [[float(x) / 2 for x in row] for row in U_list]
        V_out = [[float(x) / 2 for x in row] for row in V_list]
        W_out = [[float(x) / 2 for x in row] for row in W_list]
        return {"U": U_out, "V": V_out, "W": W_out}

    # Strategy: Try ranks from 25 up to 44
    # Use multiple random initialization strategies
    best_result = None

    for r in range(25, min(rank_schoolbook, 45)):
        if time.time() > deadline:
            break
        
        # Number of trials per rank - adjust based on time
        max_trials = 2000
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            trial_type = trial % 8
            
            if trial_type == 0:
                # Sparse {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n*m, r))
                V = np.random.choice([-1, 0, 1], size=(m*p, r))
            elif trial_type == 1:
                # Denser {-1, 0, 1}
                U = np.where(np.random.rand(n*m, r) < 0.6, np.random.choice([-1, 1], size=(n*m, r)), 0)
                V = np.where(np.random.rand(m*p, r) < 0.6, np.random.choice([-1, 1], size=(m*p, r)), 0)
            elif trial_type == 2:
                # Small integers {-2, -1, 0, 1, 2}
                U = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, r))
                V = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, r))
            elif trial_type == 3:
                # Very sparse with larger values
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for _ in range(n*m*r//4):
                    i, j = np.random.randint(0, n*m), np.random.randint(0, r)
                    U[i, j] = np.random.choice([-2, -1, 1, 2])
                for _ in range(m*p*r//4):
                    i, j = np.random.randint(0, m*p), np.random.randint(0, r)
                    V[i, j] = np.random.choice([-2, -1, 1, 2])
            elif trial_type == 4:
                # Binary {-1, 1} dense
                U = np.random.choice([-1, 1], size=(n*m, r))
                V = np.random.choice([-1, 1], size=(m*p, r))
            elif trial_type == 5:
                # Structured: block patterns
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                # Create blocks of 3
                for c in range(r):
                    block_start = (c // 3) * 3
                    if block_start < n*m:
                        for offset in range(3):
                            if block_start + offset < n*m:
                                U[block_start + offset, c] = np.random.choice([-1, 1])
                    if block_start < m*p:
                        for offset in range(3):
                            if block_start + offset < m*p:
                                V[block_start + offset, c] = np.random.choice([-1, 1])
            elif trial_type == 6:
                # Random with some structure: reuse patterns
                U_base = np.random.choice([-1, 0, 1], size=(n*m, min(r, 9)))
                V_base = np.random.choice([-1, 0, 1], size=(m*p, min(r, 9)))
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                for c in range(r):
                    src = c % min(r, 9)
                    U[:, c] = U_base[:, src] * np.random.choice([-1, 1])
                    V[:, c] = V_base[:, src] * np.random.choice([-1, 1])
            else:
                # Gaussian thresholded
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                U = np.where(np.abs(U) > 0.4, np.sign(U).astype(int), 0)
                V = np.where(np.abs(V) > 0.4, np.sign(V).astype(int), 0)
            
            result = try_integer_solution(U, V, r, n, m, p)
            if result is not None:
                U_list, V_list, W_list = result
                return to_half_integer_output(U_list, V_list, W_list)
        
        # If we get here without finding a solution for this rank, continue to next rank

    # Fallback: schoolbook
    return schoolbook_decomposition()
